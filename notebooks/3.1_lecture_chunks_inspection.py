# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Inspect the processed lecture tables and the chunks built from them."""

# MAGIC %md
# MAGIC # Lecture chunks inspection
# MAGIC
# MAGIC Reads the tables that the `lecture_insights_assistant` job writes to your schema and checks
# MAGIC them before the chunks are embedded and synced to Lakebase:
# MAGIC
# MAGIC | Table | One row per | Written by |
# MAGIC |---|---|---|
# MAGIC | `lecture_transcript` | speaker turn | `lecture_ingestion` task |
# MAGIC | `lecture_slides` | PDF page (slides and materials) | `lecture_ingestion` task |
# MAGIC | `lecture_chunks` | chunk: a transcript window of up to 300 words, or one page | `lecture_chunking` task |
# MAGIC
# MAGIC It writes nothing. Run the job first (`databricks bundle run lecture_insights_assistant`).

# COMMAND ----------

# MAGIC %md
# MAGIC Install the package wheel that `databricks bundle deploy` uploaded next to the bundle files.
# MAGIC Update the version in the path when `version.txt` changes.

# COMMAND ----------

# MAGIC %pip install ../../artifacts/.internal/arxiv_curator-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import functions as F

from arxiv_curator.config import get_env, load_config

spark = DatabricksSession.builder.getOrCreate()

# The environment comes from the env widget; ENV or "dev" when run elsewhere.
try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)
tables = {name: f"{cfg.full_schema_name}.{name}" for name in ["lecture_transcript", "lecture_slides", "lecture_chunks"]}

# COMMAND ----------
# Row counts, schemas and sample rows of each table.

for name, table in tables.items():
    df = spark.table(table)
    logger.info(f"{table}: {df.count()} rows")
    df.printSchema()
    df.drop("source_path", "ingested_at").show(5, truncate=80)

# COMMAND ----------
# Chunks and words per lecture and source.

chunks = spark.table(tables["lecture_chunks"])
chunks.groupBy("lecture_id", "source").agg(
    F.count("*").alias("chunks"),
    F.sum("word_count").alias("words"),
    F.min("word_count").alias("min_words"),
    F.expr("percentile(word_count, 0.5)").alias("median_words"),
    F.max("word_count").alias("max_words"),
).orderBy("lecture_id", "source").show()

# COMMAND ----------
# Empty chunks: no text after cleaning. The chunking skips these, so the count should be 0.

empty = chunks.where(F.trim(F.col("text")) == "")
logger.info(f"Empty chunks: {empty.count()}")
empty.select("chunk_id", "source_path").show(truncate=False)

# COMMAND ----------
# Duplicate chunks: a repeated chunk_id breaks the synced table's primary key; repeated text
# (such as a slide shown twice) is stored and embedded twice.

dup_ids = chunks.groupBy("chunk_id").count().where("count > 1")
logger.info(f"Duplicate chunk ids: {dup_ids.count()}")
dup_ids.show(truncate=False)

dup_text = (
    chunks.groupBy("text")
    .agg(F.count("*").alias("copies"), F.collect_list("chunk_id").alias("chunk_ids"))
    .where("copies > 1")
)
logger.info(f"Texts stored more than once: {dup_text.count()}")
dup_text.select("copies", "chunk_ids", F.substring("text", 1, 80).alias("text")).show(truncate=False)
