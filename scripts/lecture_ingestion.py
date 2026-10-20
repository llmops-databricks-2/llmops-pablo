"""Parse the raw lecture files in the volume into the lecture assistant's Delta tables.

The job version of `notebooks/1.3_lecture_data_ingestion.py`: the same parsing and writes, without
the sample-row checks. It reads `<volume>/raw/<lecture_id>/` and overwrites four tables in the
target's schema: `lectures`, `lecture_transcript`, `lecture_chat` and `lecture_slides`.

Usage:
    python scripts/lecture_ingestion.py --env dev
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
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

from arxiv_curator.config import ProjectConfig, load_config
from arxiv_curator.lecture_parsing import parse_chat, parse_transcript

CHAT_TZ = "Europe/Amsterdam"
PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"

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


def read_files(spark: SparkSession, raw: str, pattern: str) -> dict[str, tuple[str, bytes]]:
    """Read files from the volume, keyed by lecture id.

    Args:
        spark: The Spark session.
        raw: The volume's `raw` directory.
        pattern: A glob under `raw`, such as `*/transcript.docx`.

    Returns:
        A map from lecture id to the file's path and content.

    """
    rows = spark.read.format("binaryFile").load(f"{raw}/{pattern}").select("path", "content").collect()
    return {PurePosixPath(r.path).parent.name: (r.path, bytes(r.content)) for r in rows}


def parse_text_sources(spark: SparkSession, raw: str) -> dict[str, list[dict]]:
    """Parse every lecture's transcript and chat into table rows.

    Args:
        spark: The Spark session.
        raw: The volume's `raw` directory.

    Returns:
        Rows for the `lectures`, `lecture_transcript` and `lecture_chat` tables.

    """
    transcripts = read_files(spark, raw, "*/transcript.docx")
    chats = read_files(spark, raw, "*/chat.txt")
    logger.info(f"Transcripts: {sorted(transcripts)}; chats: {sorted(chats)}")

    rows: dict[str, list[dict]] = {"lectures": [], "lecture_transcript": [], "lecture_chat": []}
    for lecture_id, (path, content) in sorted(transcripts.items()):
        header, turns = parse_transcript(content, lecture_id)
        lecture_date = header.recording_start_utc.astimezone(ZoneInfo(CHAT_TZ)).date()
        rows["lectures"].append(
            {
                "lecture_id": lecture_id,
                "lecture_date": lecture_date,
                "recording_start_utc": header.recording_start_utc,
                "duration_seconds": header.duration_seconds,
                "source_path": path,
            }
        )
        rows["lecture_transcript"] += [{**t, "source_path": path} for t in turns]
        if lecture_id in chats:
            chat_path, chat_content = chats[lecture_id]
            messages = parse_chat(
                chat_content.decode("utf-8"), lecture_id, lecture_date, header.recording_start_utc, CHAT_TZ
            )
            rows["lecture_chat"] += [{**m, "source_path": chat_path} for m in messages]

    missing = sorted(set(chats) - set(transcripts))
    if missing:
        logger.warning(f"Chats without a transcript are skipped, as their start time is unknown: {missing}")
    return rows


def parse_pdfs(spark: SparkSession, raw: str) -> DataFrame:
    """Parse the slide and materials PDFs with ai_parse_document, one row per page.

    Args:
        spark: The Spark session.
        raw: The volume's `raw` directory.

    Returns:
        Rows for the `lecture_slides` table.

    """
    return spark.sql(f"""
        WITH parsed AS (
            SELECT path, ai_parse_document(content, map('version', '2.0')) AS doc
            FROM READ_FILES('{raw}/*/*.pdf', format => 'binaryFile')
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


def write_table(df: DataFrame, cfg: ProjectConfig, name: str, ingested_at: datetime) -> None:
    """Overwrite a table in the target schema, stamping each row with the run time.

    Args:
        df: The rows to write.
        cfg: The project configuration for the target environment.
        name: The table name, without catalog and schema.
        ingested_at: The run's timestamp.

    """
    table = f"{cfg.full_schema_name}.{name}"
    df = df.withColumn("ingested_at", F.lit(ingested_at))
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table)
    logger.info(f"Wrote {table}: {df.count()} rows")


def main(env: str) -> None:
    """Run the ingestion for one environment.

    Args:
        env: The target environment (dev, acc or prd).

    """
    spark = DatabricksSession.builder.getOrCreate()
    cfg = load_config(str(PROJECT_CONFIG), env)
    raw = f"/Volumes/{cfg.catalog}/{cfg.schema}/{cfg.volume}/raw"
    ingested_at = datetime.now(UTC)
    logger.info(f"Reading {raw}, writing to {cfg.full_schema_name}")

    rows = parse_text_sources(spark, raw)
    for name, schema in SCHEMAS.items():
        write_table(spark.createDataFrame(rows[name], schema), cfg, name, ingested_at)
    write_table(parse_pdfs(spark, raw), cfg, "lecture_slides", ingested_at)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dev", choices=["dev", "acc", "prd"])
    main(parser.parse_args().env)
