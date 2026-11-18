"""Tests for the Lakebase sync's naming and index DDL, without a database."""

from pathlib import Path

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.lakebase import SYNCED_TABLE, index_statements, postgres_table

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def test_synced_table_lands_in_the_schema_named_after_unity_catalog() -> None:
    """The Postgres table is quoted and sits in the schema named after the Unity Catalog schema."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")
    assert SYNCED_TABLE == "lecture_chunks_synced"
    assert postgres_table(cfg) == '"pablo"."lecture_chunks_synced"'


def test_index_statements_build_hnsw_cosine_and_gin_full_text() -> None:
    """Both indexes are idempotent, and the HNSW index uses cosine distance."""
    hnsw, gin = index_statements(ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev"))
    assert hnsw.startswith("CREATE INDEX IF NOT EXISTS lecture_chunks_synced_embedding_hnsw")
    assert "USING hnsw (embedding vector_cosine_ops)" in hnsw
    assert gin.startswith("CREATE INDEX IF NOT EXISTS lecture_chunks_synced_text_gin")
    assert "USING gin (to_tsvector('english', text))" in gin
