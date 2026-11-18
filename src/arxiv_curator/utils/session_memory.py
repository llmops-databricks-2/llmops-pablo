"""Keep the agent's conversation in Lakebase Postgres, one session per lecturer thread.

Adapted from the course's `memory.py` and notebook 3.3, with the problems the lecture names fixed:

- Messages are ordered by `seq`, their position in the session, not by timestamp.
- A save writes each position once (`ON CONFLICT DO NOTHING`), so a retried save adds no duplicates.
- Each session has an owner, and only that caller may read or extend it. A session ID alone grants nothing.
- A failed load raises instead of returning an empty history that would hide the failure.

Messages are stored whole as JSONB, including the assistant's tool calls and each tool result's call ID, so a
reloaded conversation is valid input for the model. The tables sit in the Postgres schema named after the Unity
Catalog schema, next to the synced chunk table. Each call opens its own connection with a fresh credential for
the workspace client's identity: the developer in dev, the service principal once deployed.
"""

from collections.abc import Callable
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger
from psycopg.types.json import Jsonb

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.lakebase import connect

SESSIONS_TABLE = "sessions"
MESSAGES_TABLE = "session_messages"


def schema_statements(schema: str) -> list[str]:
    """Return the idempotent DDL for the session tables.

    The primary key of `session_messages` is `(session_id, seq)`; its index leads with `session_id`, so it
    serves loading one session, and it is what makes a repeated save a no-op.

    Args:
        schema: The Postgres schema, named after the Unity Catalog schema.

    Returns:
        The `CREATE TABLE IF NOT EXISTS` statements, sessions first.

    """
    return [
        f"""CREATE TABLE IF NOT EXISTS "{schema}".{SESSIONS_TABLE} (
            session_id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        f"""CREATE TABLE IF NOT EXISTS "{schema}".{MESSAGES_TABLE} (
            session_id TEXT NOT NULL REFERENCES "{schema}".{SESSIONS_TABLE} (session_id),
            seq INT NOT NULL,
            message JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, seq)
        )""",
    ]


class SessionMemory:
    """Load and save a session's messages, and check who owns the session."""

    def __init__(
        self,
        cfg: ProjectConfig,
        w: WorkspaceClient,
        connect_fn: Callable[[], psycopg.Connection] | None = None,
    ) -> None:
        """Bind the helper to the project's Lakebase.

        Args:
            cfg: The project configuration for the target environment.
            w: The workspace client whose identity connects to Postgres.
            connect_fn: Opens a connection; `lakebase.connect` for this project when omitted.

        """
        self._schema = cfg.db_schema
        self._connect = connect_fn or (lambda: connect(cfg, w))

    def ensure_schema(self) -> None:
        """Create the session tables if they do not exist."""
        with self._connect() as conn:
            for statement in schema_statements(self._schema):
                conn.execute(statement)
        logger.info(f'Session tables ready in "{self._schema}"')

    def claim(self, session_id: str, caller: str) -> None:
        """Create the session for the caller, or check that the caller owns it.

        Args:
            session_id: The session ID.
            caller: The authenticated caller's name. Never take it from the request body.

        Raises:
            PermissionError: If another caller owns the session.

        """
        with self._connect() as conn:
            conn.execute(
                f'INSERT INTO "{self._schema}".{SESSIONS_TABLE} (session_id, owner) VALUES (%s, %s) '
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id, caller),
            )
            owner = conn.execute(
                f'SELECT owner FROM "{self._schema}".{SESSIONS_TABLE} WHERE session_id = %s', (session_id,)
            ).fetchone()[0]
        if owner != caller:
            raise PermissionError(f"Session {session_id} belongs to another caller")

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session's messages in the order they were said.

        Call `claim` first: this method does not check the owner.

        Args:
            session_id: The session ID.

        Returns:
            The messages ordered by `seq`; an empty list for a session with none.

        """
        with self._connect() as conn:
            rows = conn.execute(
                f'SELECT message FROM "{self._schema}".{MESSAGES_TABLE} WHERE session_id = %s ORDER BY seq',
                (session_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def save(self, session_id: str, start_seq: int, messages: list[dict[str, Any]]) -> int:
        """Save a turn's new messages at positions `start_seq`, `start_seq + 1`, and so on.

        A position already taken is skipped, so saving the same turn twice writes nothing the second time.

        Args:
            session_id: The session ID; `claim` must have created it.
            start_seq: The position of the first new message, the length of the loaded history.
            messages: The turn's new messages.

        Returns:
            The number of messages written.

        """
        written = 0
        with self._connect() as conn:
            for offset, message in enumerate(messages):
                cursor = conn.execute(
                    f'INSERT INTO "{self._schema}".{MESSAGES_TABLE} (session_id, seq, message) '
                    "VALUES (%s, %s, %s) ON CONFLICT (session_id, seq) DO NOTHING",
                    (session_id, start_seq + offset, Jsonb(message)),
                )
                written += cursor.rowcount
        if written < len(messages):
            logger.warning(
                f"Session {session_id}: {len(messages) - written} of {len(messages)} messages were already "
                f"saved from position {start_seq}; a retried save, or two turns at once"
            )
        return written
