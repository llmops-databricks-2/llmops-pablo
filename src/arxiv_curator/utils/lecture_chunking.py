"""Chunk the parsed lecture tables into `lecture_chunks`.

Shared by the job's `scripts/lecture_chunking.py` and any notebook that rebuilds the chunks. Reads
`lecture_transcript` and `lecture_slides` from the target's schema, embeds every chunk with the
configured embedding endpoint and overwrites `lecture_chunks`, with change data feed on so a Lakebase
synced table can follow it. Chat messages are left out:
they are student questions, not course content.
"""

from collections import defaultdict
from datetime import UTC, datetime

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from arxiv_curator.config import ProjectConfig
from arxiv_curator.lecture_chunking import chunk_page, chunk_transcript, split_sentences
from arxiv_curator.schemas.lecture_tables import LECTURE_CHUNKS
from arxiv_curator.utils.lecture_ingestion import write_table

CHUNKS_TABLE = "lecture_chunks"
# The output size of databricks-gte-large-en; the Lakebase synced table stores it as vector(1024).
EMBEDDING_DIM = 1024


def build_chunks(spark: SparkSession, cfg: ProjectConfig) -> list[dict]:
    """Build the chunk rows from the transcript and slide tables.

    The tables hold a few hundred rows, so they are collected and chunked on the driver.

    Args:
        spark: The Spark session.
        cfg: The project configuration for the target environment.

    Returns:
        The chunk rows, each with its `source_path`.

    """
    schema = cfg.full_schema_name
    turns = defaultdict(list)
    for row in spark.table(f"{schema}.lecture_transcript").collect():
        turns[row.lecture_id].append(row.asDict())

    chunks = []
    for lecture_id, lecture_turns in sorted(turns.items()):
        path = lecture_turns[0]["source_path"]
        chunks += [
            {**c, "source_path": path} for c in chunk_transcript(lecture_id, lecture_turns, split=split_sentences)
        ]

    empty_pages = 0
    for row in spark.table(f"{schema}.lecture_slides").orderBy("lecture_id", "kind", "page").collect():
        chunk = chunk_page(row.lecture_id, row.kind, row.page, row.text)
        if chunk is None:
            empty_pages += 1
            continue
        chunks.append({**chunk, "source_path": row.source_path})

    logger.info(f"Built {len(chunks)} chunks; skipped {empty_pages} pages with no text")
    return chunks


def embed_chunks(df: DataFrame, endpoint: str) -> DataFrame:
    """Add an `embedding` column with the endpoint's vector for each chunk's text.

    The embeddings are self-managed: they are computed here, stored in Delta and searched with pgvector
    in Lakebase, so queries must be embedded with the same endpoint.

    Args:
        df: Chunk rows with a `text` column.
        endpoint: The embedding model serving endpoint, such as `databricks-gte-large-en`.

    Returns:
        The rows with `embedding` as ARRAY<FLOAT>.

    """
    quoted = endpoint.replace("'", "''")
    return df.withColumn("embedding", F.expr(f"CAST(ai_query('{quoted}', text) AS ARRAY<FLOAT>)"))


def chunk_lectures(spark: SparkSession, cfg: ProjectConfig) -> str:
    """Overwrite `lecture_chunks` from the parsed lecture tables, with an embedding per chunk.

    Every chunk is embedded again on each run; at a few hundred chunks that costs little.

    Args:
        spark: The Spark session.
        cfg: The project configuration for the target environment.

    Returns:
        The fully qualified name of the chunk table.

    Raises:
        RuntimeError: If a chunk has no embedding or one of the wrong size.

    """
    df = embed_chunks(spark.createDataFrame(build_chunks(spark, cfg), LECTURE_CHUNKS), cfg.embedding_endpoint)
    table = write_table(df, cfg, CHUNKS_TABLE, datetime.now(UTC))
    spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    bad = spark.table(table).where(F.col("embedding").isNull() | (F.size("embedding") != EMBEDDING_DIM)).count()
    if bad:
        raise RuntimeError(f"{bad} chunks in {table} have no {EMBEDDING_DIM}-float embedding")
    logger.info(f"Embedded every chunk with {cfg.embedding_endpoint} ({EMBEDDING_DIM} floats)")
    return table
