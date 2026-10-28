"""Sync `lecture_chunks` into Lakebase Postgres and index it for pgvector search.

Adapted from the course reference's notebook 2.5 and `vector_search.py`, without Lakebase Search: the
embeddings are self-managed (computed by the chunking task), a synced table copies them into Postgres as
`vector(1024)`, and the indexes and queries are ours. Delta stays the source of truth; the synced table
is read-only. The synced table lands in a Postgres schema named after the Unity Catalog schema.
"""

import time

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.pipelines import UpdateInfoState
from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.lecture_chunking import CHUNKS_TABLE, EMBEDDING_DIM

SYNCED_TABLE = f"{CHUNKS_TABLE}_synced"
# dev only for now: acc and prd share the Lakebase project and need their own database first.
DATABASE = "databricks_postgres"
SYNC_TIMEOUT_SECONDS = 30 * 60

_ONLINE = {SyncedTableState.SYNCED_TABLE_ONLINE, SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE}
_FAILED = {SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED, SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED}


def postgres_table(cfg: ProjectConfig) -> str:
    """Return the synced table's quoted Postgres name.

    Args:
        cfg: The project configuration for the target environment.

    Returns:
        The table as `"<schema>"."lecture_chunks_synced"`.

    """
    return f'"{cfg.db_schema}"."{SYNCED_TABLE}"'


def index_statements(cfg: ProjectConfig) -> list[str]:
    """Return the idempotent index DDL for the synced table.

    The HNSW index serves cosine-distance search on `embedding`; the GIN index serves full-text ranking on
    `text`, and queries must use the same `to_tsvector('english', text)` expression to use it.

    Args:
        cfg: The project configuration for the target environment.

    Returns:
        The `CREATE INDEX IF NOT EXISTS` statements.

    """
    table = postgres_table(cfg)
    return [
        f"CREATE INDEX IF NOT EXISTS {SYNCED_TABLE}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS {SYNCED_TABLE}_text_gin ON {table} USING gin (to_tsvector('english', text))",
    ]


def connect(cfg: ProjectConfig, w: WorkspaceClient) -> psycopg.Connection:
    """Open a TLS connection to the project's Postgres with a short-lived database credential.

    Opening a connection wakes a suspended compute. No password is stored: the credential comes from the
    Databricks SDK for the current identity, a user in dev or a service principal in a deployment.

    Args:
        cfg: The project configuration for the target environment.
        w: The workspace client.

    Returns:
        An open connection to the `databricks_postgres` database.

    """
    branch = f"projects/{cfg.lakebase_project_id}/branches/production"
    endpoint = next(iter(w.postgres.list_endpoints(parent=branch)))
    token = w.postgres.generate_database_credential(endpoint=endpoint.name).token
    return psycopg.connect(
        host=endpoint.status.hosts.host,
        port=5432,
        dbname=DATABASE,
        user=w.current_user.me().user_name,
        password=token,
        sslmode="require",
    )


def _create_synced_table(cfg: ProjectConfig, w: WorkspaceClient, name: str) -> None:
    """Create the synced table, which runs its first sync.

    Args:
        cfg: The project configuration for the target environment.
        w: The workspace client.
        name: The synced table's Unity Catalog name.

    """
    spec = SyncedTableSyncedTableSpec(
        source_table_full_name=f"{cfg.full_schema_name}.{CHUNKS_TABLE}",
        branch=f"projects/{cfg.lakebase_project_id}/branches/production",
        postgres_database=DATABASE,
        primary_key_columns=["chunk_id"],
        scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
        create_database_objects_if_missing=True,
        new_pipeline_spec=NewPipelineSpec(
            budget_policy_id=cfg.usage_policy_id, storage_catalog=cfg.catalog, storage_schema=cfg.db_schema
        ),
        # Without the override a Delta ARRAY lands in Postgres as JSONB, and vector operators do not apply.
        type_overrides=[
            SyncedTableSyncedTableSpecTypeOverride(
                column_name="embedding",
                pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                size=EMBEDDING_DIM,
            )
        ],
    )
    w.postgres.create_synced_table(synced_table=SyncedTable(spec=spec), synced_table_id=name).wait()
    logger.info(f"Created synced table {name}")


def _wait_for_sync(w: WorkspaceClient, name: str, update_id: str | None) -> None:
    """Wait until the started update completes, or a new table is online.

    Args:
        w: The workspace client.
        name: The synced table's Unity Catalog name.
        update_id: The pipeline update started for this sync, or None after a create.

    Raises:
        RuntimeError: If the sync fails, is canceled or runs past `SYNC_TIMEOUT_SECONDS`.

    """
    deadline = time.monotonic() + SYNC_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = w.postgres.get_synced_table(name=f"synced_tables/{name}").status
        if status.detailed_state in _FAILED:
            raise RuntimeError(f"Synced table {name} failed: {status.detailed_state} | {status.message}")
        if update_id:
            # Wait for the sync this run started, not just for the table to be online.
            update = w.pipelines.get_update(pipeline_id=status.pipeline_id, update_id=update_id).update
            if update.state == UpdateInfoState.COMPLETED:
                return
            if update.state in {UpdateInfoState.FAILED, UpdateInfoState.CANCELED}:
                raise RuntimeError(f"Sync update {update_id} of {name} ended in state {update.state}")
        elif status.detailed_state in _ONLINE:
            return
        logger.info(f"Waiting for {name}: {status.detailed_state}")
        time.sleep(30)
    raise RuntimeError(f"Sync of {name} did not finish within {SYNC_TIMEOUT_SECONDS} seconds")


def sync_chunks(cfg: ProjectConfig, w: WorkspaceClient) -> None:
    """Create the synced table on the first run, or start a sync of the existing one, and wait for it.

    Args:
        cfg: The project configuration for the target environment.
        w: The workspace client.

    """
    name = f"{cfg.full_schema_name}.{SYNCED_TABLE}"
    try:
        pipeline_id = w.postgres.get_synced_table(name=f"synced_tables/{name}").status.pipeline_id
    except NotFound:
        _create_synced_table(cfg, w, name)
        update_id = None
    else:
        # A synced table runs on a Lakeflow pipeline; starting an update is "Sync now".
        update_id = w.pipelines.start_update(pipeline_id=pipeline_id).update_id
        logger.info(f"Started sync of {name}: pipeline {pipeline_id}, update {update_id}")
    _wait_for_sync(w, name, update_id)


def check_synced_table(spark: SparkSession, cfg: ProjectConfig, conn: psycopg.Connection) -> None:
    """Log and check the synced table's vector type, row count and indexes.

    Args:
        spark: The Spark session.
        cfg: The project configuration for the target environment.
        conn: An open Postgres connection.

    Raises:
        RuntimeError: If `embedding` is not `vector(1024)`, the row counts differ or an index is missing.

    """
    table = postgres_table(cfg)
    rows = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    pg_type = (conn.execute(f"SELECT pg_typeof(embedding)::text FROM {table} LIMIT 1").fetchone() or [None])[0]
    column_type = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = %s::regclass AND attname = %s",
        (table, "embedding"),
    ).fetchone()[0]
    indexes = [
        r[0]
        for r in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = %s ORDER BY indexname",
            (cfg.db_schema, SYNCED_TABLE),
        ).fetchall()
    ]
    source_rows = spark.table(f"{cfg.full_schema_name}.{CHUNKS_TABLE}").count()
    logger.info(f"{table}: {rows} rows (source {source_rows}), embedding {pg_type} as {column_type}")
    logger.info(f"{table} indexes: {indexes}")

    expected = {f"{SYNCED_TABLE}_embedding_hnsw", f"{SYNCED_TABLE}_text_gin"}
    problems = []
    if column_type != f"vector({EMBEDDING_DIM})":
        problems.append(f"embedding is {column_type}, not vector({EMBEDDING_DIM})")
    if rows != source_rows:
        problems.append(f"{rows} rows against {source_rows} in the source")
    if missing := expected - set(indexes):
        problems.append(f"missing indexes {sorted(missing)}")
    if problems:
        raise RuntimeError(f"{table}: " + "; ".join(problems))


def sync_lecture_chunks(spark: SparkSession, cfg: ProjectConfig, w: WorkspaceClient | None = None) -> str:
    """Enable pgvector, sync `lecture_chunks` into Lakebase, index the copy and check it.

    pgvector is enabled first, because creating a synced table installs no extensions. The indexes are
    created after every sync: a full refresh recreates the table without them.

    Args:
        spark: The Spark session.
        cfg: The project configuration for the target environment.
        w: The workspace client; a new one when omitted.

    Returns:
        The synced table's quoted Postgres name.

    """
    w = w or WorkspaceClient()
    with connect(cfg, w) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    logger.info("pgvector extension ready")

    sync_chunks(cfg, w)

    with connect(cfg, w) as conn:
        for statement in index_statements(cfg):
            conn.execute(statement)
        check_synced_table(spark, cfg, conn)
    return postgres_table(cfg)
