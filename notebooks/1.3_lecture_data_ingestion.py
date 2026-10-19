# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Parse the raw lecture files in the volume into the lecture assistant's Delta tables."""

# MAGIC %md
# MAGIC # Lecture data ingestion
# MAGIC
# MAGIC Modelled on the course's `1.3_arxiv_data_ingestion.py`. It reads the raw lecture files
# MAGIC uploaded to `<volume>/raw/<lecture_id>/` and writes four tables to your schema:
# MAGIC
# MAGIC | Table | One row per | Parsed with |
# MAGIC |---|---|---|
# MAGIC | `lectures` | lecture | transcript header |
# MAGIC | `lecture_transcript` | speaker turn | `arxiv_curator.lecture_parsing` |
# MAGIC | `lecture_chat` | chat message | `arxiv_curator.lecture_parsing` |
# MAGIC | `lecture_slides` | PDF page (slides and materials) | `ai_parse_document` |
# MAGIC
# MAGIC Every run overwrites the four tables.

# COMMAND ----------

from datetime import UTC, datetime
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from arxiv_curator.config import get_env, load_config
from arxiv_curator.lecture_parsing import parse_chat, parse_transcript

spark = DatabricksSession.builder.getOrCreate()

# The environment comes from the job's base parameters; ENV or "dev" when run elsewhere.
try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)
RAW = f"/Volumes/{cfg.catalog}/{cfg.schema}/{cfg.volume}/raw"
CHAT_TZ = "Europe/Amsterdam"
INGESTED_AT = datetime.now(UTC)
logger.info(f"Reading {RAW}, writing to {cfg.full_schema_name}")

# COMMAND ----------
# Transcripts and chats: parsed in Python, one lecture at a time.


def read_files(pattern: str) -> dict[str, tuple[str, bytes]]:
    """Read files from the volume, keyed by lecture id.

    Args:
        pattern: A glob under `RAW`, such as `*/transcript.docx`.

    Returns:
        A map from lecture id to the file's path and content.

    """
    rows = spark.read.format("binaryFile").load(f"{RAW}/{pattern}").select("path", "content").collect()
    return {PurePosixPath(r.path).parent.name: (r.path, bytes(r.content)) for r in rows}


transcripts = read_files("*/transcript.docx")
chats = read_files("*/chat.txt")
logger.info(f"Transcripts: {sorted(transcripts)}; chats: {sorted(chats)}")

lecture_rows, turn_rows, message_rows = [], [], []
for lecture_id, (path, content) in sorted(transcripts.items()):
    header, turns = parse_transcript(content, lecture_id)
    lecture_date = header.recording_start_utc.astimezone(ZoneInfo(CHAT_TZ)).date()
    lecture_rows.append(
        {
            "lecture_id": lecture_id,
            "lecture_date": lecture_date,
            "recording_start_utc": header.recording_start_utc,
            "duration_seconds": header.duration_seconds,
            "source_path": path,
        }
    )
    turn_rows += [{**t, "source_path": path} for t in turns]
    if lecture_id in chats:
        chat_path, chat_content = chats[lecture_id]
        messages = parse_chat(
            chat_content.decode("utf-8"), lecture_id, lecture_date, header.recording_start_utc, CHAT_TZ
        )
        message_rows += [{**m, "source_path": chat_path} for m in messages]

missing = sorted(set(chats) - set(transcripts))
if missing:
    logger.warning(f"Chats without a transcript are skipped, as their start time is unknown: {missing}")

# COMMAND ----------
# Slides and materials: ai_parse_document, then one row per page.

slides_df = spark.sql(f"""
    WITH parsed AS (
        SELECT path, ai_parse_document(content, map('version', '2.0')) AS doc
        FROM READ_FILES('{RAW}/*/*.pdf', format => 'binaryFile')
    ),
    elements AS (
        SELECT path, pos, el
        FROM parsed
        LATERAL VIEW posexplode(variant_get(doc, '$.document.elements', 'ARRAY<VARIANT>')) e AS pos, el
    )
    SELECT
        regexp_extract(path, '/raw/([^/]+)/', 1) AS lecture_id,
        regexp_extract(path, '/([^/]+)[.]pdf$', 1) AS kind,
        variant_get(el, '$.bbox[0].page_id', 'INT') + 1 AS page,
        array_join(
            transform(
                array_sort(collect_list(struct(
                    pos,
                    coalesce(variant_get(el, '$.content', 'STRING'),
                             variant_get(el, '$.description', 'STRING')) AS txt
                ))),
                x -> x.txt
            ),
            char(10)
        ) AS text,
        path AS source_path
    FROM elements
    GROUP BY path, page
""")

# COMMAND ----------
# Write the four tables.

SCHEMAS = {
    "lectures": StructType(
        [
            StructField("lecture_id", StringType(), False),
            StructField("lecture_date", DateType(), False),
            StructField("recording_start_utc", TimestampType(), False),
            StructField("duration_seconds", IntegerType(), False),
            StructField("source_path", StringType(), False),
        ]
    ),
    "lecture_transcript": StructType(
        [
            StructField("lecture_id", StringType(), False),
            StructField("turn", IntegerType(), False),
            StructField("speaker", StringType(), False),
            StructField("offset_seconds", IntegerType(), False),
            StructField("ts_utc", TimestampType(), False),
            StructField("text", StringType(), True),
            StructField("source_path", StringType(), False),
        ]
    ),
    "lecture_chat": StructType(
        [
            StructField("lecture_id", StringType(), False),
            StructField("message", IntegerType(), False),
            StructField("sender", StringType(), True),
            StructField("ts_utc", TimestampType(), False),
            StructField("offset_seconds", IntegerType(), False),
            StructField("text_truncated", BooleanType(), False),
            StructField("text", StringType(), True),
            StructField("source_path", StringType(), False),
        ]
    ),
}


def write_table(df: DataFrame, name: str) -> str:
    """Overwrite a table in the target schema, stamping each row with the run time.

    Args:
        df: The rows to write.
        name: The table name, without catalog and schema.

    Returns:
        The fully qualified table name.

    """
    table = f"{cfg.full_schema_name}.{name}"
    df = df.withColumn("ingested_at", F.lit(INGESTED_AT))
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table)
    logger.info(f"Wrote {table}")
    return table


rows = {"lectures": lecture_rows, "lecture_transcript": turn_rows, "lecture_chat": message_rows}
tables = [write_table(spark.createDataFrame(rows[name], schema), name) for name, schema in SCHEMAS.items()]
tables.append(write_table(slides_df, "lecture_slides"))

# COMMAND ----------
# Verify the data: schema, counts per lecture, and sample records.

for table in tables:
    df = spark.table(table)
    logger.info(f"{table}: {df.count()} rows")
    df.printSchema()
    df.groupBy("lecture_id").count().orderBy("lecture_id").show()
    df.drop("source_path", "ingested_at").show(5, truncate=60)
