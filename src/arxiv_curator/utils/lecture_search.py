"""Search the lecture chunks in Lakebase with pgvector: semantic, filtered and hybrid.

Adapted from the course reference's `vector_search.py`. All three modes run in Postgres on the synced table
`lecture_chunks_synced` and its self-managed embeddings, so a query must be embedded with the same endpoint
as the chunks (`embedding_endpoint`). Semantic search ranks by cosine distance with the HNSW index; filtered
search adds `lecture_id` and `source` constraints; hybrid search adds PostgreSQL full-text ranking on the
expression the GIN index covers.
"""

from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.lakebase import postgres_table

_COLUMNS = "chunk_id, lecture_id, source, page, start_offset_seconds, end_offset_seconds, left(text, 200) AS excerpt"
# The expression the GIN index covers; queries must use it verbatim for Postgres to use the index.
_TSVECTOR = "to_tsvector('english', text)"


def embed_query(w: WorkspaceClient, endpoint: str, query: str) -> list[float]:
    """Embed a question with the endpoint that embedded the chunks.

    Args:
        w: The workspace client.
        endpoint: The embedding serving endpoint, such as `databricks-gte-large-en`.
        query: The question text.

    Returns:
        The query vector.

    Raises:
        ValueError: If the endpoint returns no vector.

    """
    response = w.serving_endpoints.get_open_ai_client().embeddings.create(model=endpoint, input=query)
    if not response.data or not response.data[0].embedding:
        raise ValueError(f"{endpoint} returned no vector")
    return list(response.data[0].embedding)


def vector_literal(vector: list[float]) -> str:
    """Format a vector in pgvector's input syntax.

    Args:
        vector: The vector's values.

    Returns:
        The vector as `[v1,v2,...]`, to cast with `%s::vector`.

    """
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


def filter_clause(lecture_id: str | None = None, source: str | None = None) -> tuple[str, list[Any]]:
    """Build the WHERE condition and parameters for the optional metadata filters.

    Args:
        lecture_id: Keep only chunks of this lecture, such as `lecture-2`.
        source: Keep only chunks of this source: `transcript`, `slides` or `materials`.

    Returns:
        The condition, `TRUE` without filters, and its parameters in order.

    """
    conditions, params = [], []
    if lecture_id is not None:
        conditions.append("lecture_id = %s")
        params.append(lecture_id)
    if source is not None:
        conditions.append("source = %s")
        params.append(source)
    return " AND ".join(conditions) or "TRUE", params


def _fetch(conn: psycopg.Connection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    """Run a query and return its rows as dictionaries.

    Args:
        conn: An open Postgres connection.
        sql: The query, with `%s` placeholders.
        params: The placeholder values.

    Returns:
        One dictionary per row, keyed by column name.

    """
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [d.name for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def semantic_search(
    conn: psycopg.Connection,
    cfg: ProjectConfig,
    vector: list[float],
    k: int = 5,
    lecture_id: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return the k chunks nearest to the query vector by cosine distance.

    With a filter, pgvector's iterative HNSW scan keeps reading the index until k matching rows are found,
    instead of filtering a fixed candidate list and returning fewer.

    Args:
        conn: An open Postgres connection.
        cfg: The project configuration for the target environment.
        vector: The query vector, from `embed_query`.
        k: The number of chunks to return.
        lecture_id: Keep only chunks of this lecture.
        source: Keep only chunks of this source.

    Returns:
        The chunks, nearest first, each with `score` as cosine similarity.

    """
    where, params = filter_clause(lecture_id, source)
    if params:
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
    literal = vector_literal(vector)
    sql = (
        f"SELECT {_COLUMNS}, 1 - (embedding <=> %s::vector) AS score FROM {postgres_table(cfg)} "
        f"WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s"
    )
    return _fetch(conn, sql, [literal, *params, literal, k])


def filtered_search(
    conn: psycopg.Connection,
    cfg: ProjectConfig,
    vector: list[float],
    k: int = 5,
    *,
    lecture_id: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return the k nearest chunks within a lecture, a source, or both.

    Args:
        conn: An open Postgres connection.
        cfg: The project configuration for the target environment.
        vector: The query vector, from `embed_query`.
        k: The number of chunks to return.
        lecture_id: Keep only chunks of this lecture.
        source: Keep only chunks of this source.

    Returns:
        The chunks, nearest first, each with `score` as cosine similarity.

    Raises:
        ValueError: If neither filter is given.

    """
    if lecture_id is None and source is None:
        raise ValueError("filtered_search needs lecture_id, source or both")
    return semantic_search(conn, cfg, vector, k, lecture_id=lecture_id, source=source)


def hybrid_search(
    conn: psycopg.Connection,
    cfg: ProjectConfig,
    vector: list[float],
    query: str,
    k: int = 5,
    vector_weight: float = 0.7,
    lecture_id: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Rank chunks by a weighted sum of cosine similarity and full-text rank.

    The course's formula: `vector_weight * vector_score + (1 - vector_weight) * text_score`, where the text
    score is `ts_rank_cd` of the question's words. The two scores have different scales, so the weight is a
    starting point to tune, not a calibrated mix. Every eligible row is scored, which is fine at a few
    hundred chunks.

    Args:
        conn: An open Postgres connection.
        cfg: The project configuration for the target environment.
        vector: The query vector, from `embed_query`.
        query: The question text, for the full-text part.
        k: The number of chunks to return.
        vector_weight: The weight of the cosine similarity, between 0 and 1.
        lecture_id: Keep only chunks of this lecture.
        source: Keep only chunks of this source.

    Returns:
        The chunks, best first, each with `score`, `vector_score` and `text_score`.

    """
    where, params = filter_clause(lecture_id, source)
    sql = f"""
        WITH scored AS (
            SELECT {_COLUMNS},
                   1 - (embedding <=> %s::vector) AS vector_score,
                   ts_rank_cd({_TSVECTOR}, plainto_tsquery('english', %s)) AS text_score
            FROM {postgres_table(cfg)}
            WHERE {where}
        )
        SELECT *, %s * vector_score + %s * text_score AS score
        FROM scored
        ORDER BY score DESC
        LIMIT %s
    """
    return _fetch(conn, sql, [vector_literal(vector), query, *params, vector_weight, 1 - vector_weight, k])
