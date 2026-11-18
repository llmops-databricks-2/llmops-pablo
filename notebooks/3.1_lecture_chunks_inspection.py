# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Inspect the processed lecture tables and the chunks built from them."""

# MAGIC %md
# MAGIC # Lecture chunks inspection
# MAGIC
# MAGIC Reads the tables that the `lecture_insights_assistant` job writes to your schema and checks
# MAGIC them before the chunks are synced to Lakebase:
# MAGIC
# MAGIC | Table | One row per | Written by |
# MAGIC |---|---|---|
# MAGIC | `lecture_transcript` | speaker turn | `lecture_ingestion` task |
# MAGIC | `lecture_slides` | PDF page (slides and materials) | `lecture_ingestion` task |
# MAGIC | `lecture_chunks` | chunk: up to 300 words of transcript sentences, or one page, with its embedding | `lecture_chunking` task |
# MAGIC | `lecture_chunks_synced` (Lakebase Postgres) | chunk, with `embedding` as `vector(1024)` | `lecture_sync` task |
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
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import functions as F

from arxiv_curator.config import get_env, load_config
from arxiv_curator.utils.lakebase import SYNCED_TABLE, check_synced_table, connect, postgres_table

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

for table in tables.values():
    df = spark.table(table)
    logger.info(f"{table}: {df.count()} rows")
    df.printSchema()
    df.drop("source_path", "ingested_at", "embedding").show(5, truncate=80)

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

# COMMAND ----------
# Embeddings: every chunk needs a 1,024-float vector for the Lakebase vector(1024) column, and change data
# feed must be on for the synced table to follow the table.

bad_vectors = chunks.where(F.col("embedding").isNull() | (F.size("embedding") != 1024))
logger.info(f"Chunks without a 1,024-float embedding: {bad_vectors.count()}")
bad_vectors.select("chunk_id", F.size("embedding").alias("size")).show(truncate=False)
chunks.select(F.min(F.size("embedding")).alias("min_size"), F.max(F.size("embedding")).alias("max_size")).show()

properties = {r.key: r.value for r in spark.sql(f"SHOW TBLPROPERTIES {tables['lecture_chunks']}").collect()}
logger.info(f"Change data feed: {properties.get('delta.enableChangeDataFeed', 'false')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase synced table
# MAGIC
# MAGIC The `lecture_sync` task copies `lecture_chunks` into Postgres. This cell repeats the task's checks by
# MAGIC hand: the sync state, the vector type, the row count against `lecture_chunks` and the indexes. It
# MAGIC raises if one is wrong. Connecting wakes the Lakebase compute if it is suspended.

# COMMAND ----------

w = WorkspaceClient()
synced = w.postgres.get_synced_table(name=f"synced_tables/{cfg.full_schema_name}.{SYNCED_TABLE}").status
logger.info(f"Sync state: {synced.detailed_state} | {synced.message}")

with connect(cfg, w) as conn:
    check_synced_table(spark, cfg, conn)
    sample = conn.execute(
        f"SELECT chunk_id, lecture_id, source, word_count, left(text, 80) FROM {postgres_table(cfg)} "
        "ORDER BY chunk_id LIMIT 5"
    ).fetchall()
for row in sample:
    logger.info(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Source comparison
# MAGIC
# MAGIC A few chunks next to what they were built from, to see which structure parsing and chunking lose.
# MAGIC
# MAGIC - **Page chunks** are compared with the page's `ai_parse_document` elements and their types, which
# MAGIC   `lecture_slides` does not keep. The cell parses only the sampled PDFs again, which is billable.
# MAGIC   Open each PDF at the same page to check the parse itself.
# MAGIC - **Transcript chunks** are compared with the turns in their offset range. Speaker names stay out of
# MAGIC   the output.

# COMMAND ----------
# The sample: the longest and the shortest chunk of each page source, the first two transcript chunks
# built from the longest turn, and the transcript chunk that spans the most turns.

pages = chunks.where(F.col("page").isNotNull())
page_sample = []
for source in sorted(r.source for r in pages.select("source").distinct().collect()):
    for order in [F.desc("word_count"), F.asc("word_count")]:
        page_sample.append(pages.where(F.col("source") == source).orderBy(order, "chunk_id").first().chunk_id)

transcript = spark.table(tables["lecture_transcript"]).withColumn(
    "words", F.size(F.split(F.trim(F.col("text")), r"\s+"))
)
c, t = chunks.where(F.col("source") == "transcript").alias("c"), transcript.alias("t")
# A chunk spans from its first turn's offset up to the next turn's offset; the last chunk of a lecture
# starts and ends on the same offset.
covered = c.join(
    t,
    (F.col("c.lecture_id") == F.col("t.lecture_id"))
    & (F.col("t.offset_seconds") >= F.col("c.start_offset_seconds"))
    & (
        (F.col("t.offset_seconds") < F.col("c.end_offset_seconds"))
        | (F.col("t.offset_seconds") == F.col("c.start_offset_seconds"))
    ),
).select("c.chunk_id", "t.turn", "t.offset_seconds", "t.words", "t.text")

longest = transcript.orderBy(F.desc("words")).first()
from_longest = [
    r.chunk_id
    for r in c.where(
        (F.col("lecture_id") == longest.lecture_id) & (F.col("start_offset_seconds") == longest.offset_seconds)
    )
    .orderBy("chunk_id")
    .limit(2)
    .collect()
]
most_turns = covered.groupBy("chunk_id").count().orderBy(F.desc("count"), "chunk_id").first().chunk_id
transcript_sample = sorted({*from_longest, most_turns})

logger.info(f"Page chunks: {page_sample}")
logger.info(f"Longest turn: {longest.lecture_id} turn {longest.turn}, {longest.words} words")
logger.info(f"Transcript chunks: {transcript_sample}")

# COMMAND ----------
# Pages that produced no chunk: their text, if any, is lost to search.

slides = spark.table(tables["lecture_slides"])
skipped = slides.join(
    chunks.select("lecture_id", F.col("source").alias("kind"), "page"), ["lecture_id", "kind", "page"], "left_anti"
)
logger.info(f"Pages without a chunk: {skipped.count()}")
skipped.select("lecture_id", "kind", "page", "text").orderBy("lecture_id", "kind", "page").show(truncate=False)

# COMMAND ----------
# Parse the sampled PDFs again to recover each element's type, then show every sampled page chunk next to
# its elements. Content and description are kept apart: `lecture_slides` takes the content when it is not
# null, so a figure's description only reaches the chunk when the figure has no content at all.

sampled_pages = chunks.where(F.col("chunk_id").isin(page_sample)).orderBy("chunk_id").collect()
elements = (
    spark.read.format("binaryFile")
    .load(sorted({p.source_path for p in sampled_pages}))
    .selectExpr("path", "ai_parse_document(content, map('version', '2.0')) AS doc")
    .selectExpr("path", "posexplode(variant_get(doc, '$.document.elements', 'ARRAY<VARIANT>')) AS (pos, el)")
    .selectExpr(
        "path AS source_path",
        "variant_get(el, '$.bbox[0].page_id', 'INT') + 1 AS page",
        "pos",
        "variant_get(el, '$.type', 'STRING') AS type",
        "variant_get(el, '$.content', 'STRING') AS content",
        "variant_get(el, '$.description', 'STRING') AS description",
    )
    .collect()
)

# Element types across the sampled PDFs: every type other than text is flattened into the page's text.
types = {}
for e in elements:
    types[e.type] = types.get(e.type, 0) + 1
logger.info(f"Element types in the sampled PDFs: {dict(sorted(types.items()))}")

for chunk in sampled_pages:
    page_elements = sorted(
        (e for e in elements if e.source_path == chunk.source_path and e.page == chunk.page), key=lambda e: e.pos
    )
    logger.info(f"{chunk.chunk_id}: {chunk.word_count} words, {len(page_elements)} elements")
    for e in page_elements:
        logger.info(f"  [{e.type}] {(e.content or '')[:150]!r}")
        if e.description:
            logger.info(f"    description: {e.description[:150]!r}")
    logger.info(f"  chunk: {chunk.text[:600]!r}")

# COMMAND ----------
# Every sampled transcript chunk next to the turns in its offset range.

for chunk in chunks.where(F.col("chunk_id").isin(transcript_sample)).orderBy("chunk_id").collect():
    logger.info(
        f"{chunk.chunk_id}: {chunk.word_count} words, {chunk.start_offset_seconds}s to {chunk.end_offset_seconds}s"
    )
    logger.info(f"  chunk: {chunk.text[:600]!r}")
    covered.where(F.col("chunk_id") == chunk.chunk_id).drop("chunk_id").orderBy("turn").show(truncate=120)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lost structure
# MAGIC
# MAGIC Observed on 2026-10-27 over kick-off, lecture-1 and lecture-2 (291 chunks).
# MAGIC
# MAGIC **Slides and materials**
# MAGIC
# MAGIC - **Figure descriptions are dropped.** `ai_parse_document` describes every figure, but `lecture_slides`
# MAGIC   keeps a figure's content whenever it is not null. A figure with text keeps that text. A figure without
# MAGIC   text has an empty string as content, so its description is lost: 5 figures in lecture-2's slides.
# MAGIC - **The logo becomes a letter.** The logo on each slide is read as the content `"C"`, so short slides end
# MAGIC   in noise: `lecture-2:slides:p036` is `Homework C 36`.
# MAGIC - **Headers, footers and page numbers stay in the text.** Every page chunk repeats the same boilerplate,
# MAGIC   such as `CAUCHY · LLMOPS Lecture 2 · October 2026` and `Context engineering and RAG 06 / 15`.
# MAGIC - **Tables arrive as HTML.** Rows and columns survive, but the `<table><tr><td>` markup is embedded with
# MAGIC   the text.
# MAGIC - **Headings run into the body.** Element types are dropped and newlines are collapsed, so titles and
# MAGIC   section headers become part of the prose.
# MAGIC - **Screenshots are read as content.** A slide that shows a paper's first page (`lecture-2:slides:p010`,
# MAGIC   642 words) is mostly that paper's text, not the slide's point. Pages are never split, so it stays one
# MAGIC   chunk.
# MAGIC
# MAGIC **Transcripts**
# MAGIC
# MAGIC - **Split turns lose their timing.** Every piece of a long turn carries the whole turn's span. lecture-2
# MAGIC   turn 8 runs 2,217 words, so its chunks all cite 26 s to 1008 s.
# MAGIC - **Speakers are dropped by design.** A student's question and the lecturer's answer read as one voice.
# MAGIC - **Transcription noise is kept.** Short fragments such as greetings or misheard words remain in the
# MAGIC   chunks.
