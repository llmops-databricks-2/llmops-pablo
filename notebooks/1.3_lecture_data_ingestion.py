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
# MAGIC Every run overwrites the four tables. The parsing and writes live in
# MAGIC `arxiv_curator.utils.lecture_ingestion`, which the job's `scripts/lecture_ingestion.py` also runs;
# MAGIC this notebook adds the schema and sample-row checks.

# COMMAND ----------

# MAGIC %md
# MAGIC Install the package wheel that `databricks bundle deploy` uploaded next to the bundle files.
# MAGIC Update the version in the path when `version.txt` changes. Locally, these magic cells are
# MAGIC comments and the package comes from the uv environment.

# COMMAND ----------

# MAGIC %pip install ../../artifacts/.internal/arxiv_curator-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks.connect import DatabricksSession
from loguru import logger

from arxiv_curator.config import get_env, load_config
from arxiv_curator.utils.lecture_ingestion import ingest_lectures

spark = DatabricksSession.builder.getOrCreate()

# The environment comes from the env widget; ENV or "dev" when run elsewhere.
try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
# Parse the raw files and write the four tables (arxiv_curator.utils.lecture_ingestion, shared with
# the job's scripts/lecture_ingestion.py).

tables = ingest_lectures(spark, cfg)

# COMMAND ----------
# Verify the data: schema, counts per lecture, and sample records.

for table in tables:
    df = spark.table(table)
    logger.info(f"{table}: {df.count()} rows")
    df.printSchema()
    df.groupBy("lecture_id").count().orderBy("lecture_id").show()
    df.drop("source_path", "ingested_at").show(5, truncate=60)
