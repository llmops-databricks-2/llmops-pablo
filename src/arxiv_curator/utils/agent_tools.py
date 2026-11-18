"""Custom function tools for the lecture insights agent, with their specs and a registry.

Adapted from the course's notebook 3.1, which pairs each tool's OpenAI function spec with the callable that
runs it. `search_course_content` is the v1 tool set's one custom function (proposal/AGENT_TOOLS.md): it
embeds the question and runs semantic pgvector search over `lecture_chunks_synced`. The registry turns
every failure into an error result, so a bad call never crashes the agent loop.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.lecture_search import embed_query, semantic_search

SOURCES = ("transcript", "slides", "materials")
MAX_K = 10

SEARCH_COURSE_CONTENT_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_course_content",
        "description": (
            "Search the course's lecture transcripts, slides and readings by meaning. Use it for questions about "
            "what was taught, explained or said in a lecture. Returns chunks with the lecture, the source, the "
            "slide page or transcript offsets in seconds, an excerpt and a similarity score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain language."},
                "lecture_id": {
                    "type": "string",
                    "description": "Only search this lecture, such as 'lecture-2'. Omit to search every lecture.",
                },
                "source": {
                    "type": "string",
                    "enum": list(SOURCES),
                    "description": "Only search this kind of material. Omit to search all of them.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_K,
                    "description": "How many chunks to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class ToolInfo:
    """A tool the agent can call: its name, its OpenAI function spec and the callable that runs it."""

    name: str
    spec: dict[str, Any]
    exec_fn: Callable[..., Any]


class ToolRegistry:
    """The tools available to the agent, executed by name with JSON arguments."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """Add a tool, replacing any tool with the same name.

        Args:
            tool: The tool to add.

        """
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        """Return the specs of every registered tool, in the format the chat API's `tools` expects.

        Returns:
            One OpenAI function spec per tool.

        """
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, arguments: str) -> str:
        """Run a tool and return its result as JSON, or a JSON error instead of raising.

        Args:
            name: The tool name the model asked for.
            arguments: The model's arguments, as a JSON object string.

        Returns:
            The tool's result as JSON, or `{"error": ...}` for an unknown tool, arguments that are not a JSON
            object, invalid arguments or a failure inside the tool.

        """
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"Unknown tool: {name}")
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return _error(f"Arguments are not valid JSON: {e}")
        if not isinstance(args, dict):
            return _error("Arguments must be a JSON object")
        try:
            return json.dumps(tool.exec_fn(**args), default=str)
        except (TypeError, ValueError) as e:
            return _error(f"Invalid arguments: {e}")
        except Exception as e:  # noqa: BLE001 - any tool failure becomes an error result for the model
            return _error(f"{type(e).__name__}: {e}")


def _error(message: str) -> str:
    """Return an error result as JSON.

    Args:
        message: What went wrong.

    Returns:
        `{"error": message}` as a JSON string.

    """
    return json.dumps({"error": message})


def validate_search_args(query: str, source: str | None, k: int) -> None:
    """Check search arguments before any embedding or database call.

    Args:
        query: The search text.
        source: The optional source filter.
        k: The number of chunks requested.

    Raises:
        ValueError: If the query is empty, the source unknown, or k outside 1 to `MAX_K`.

    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if source is not None and source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}, not {source!r}")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be an integer from 1 to {MAX_K}, not {k!r}")


def make_search_course_content(
    conn: psycopg.Connection, cfg: ProjectConfig, w: WorkspaceClient
) -> Callable[..., list[dict[str, Any]]]:
    """Build the `search_course_content` callable over an open Lakebase connection.

    Args:
        conn: An open connection to the project's Postgres, from `arxiv_curator.utils.lakebase.connect`.
        cfg: The project configuration for the target environment.
        w: The workspace client, used to embed the query with `cfg.embedding_endpoint`.

    Returns:
        The tool's callable.

    """

    def search_course_content(
        query: str, lecture_id: str | None = None, source: str | None = None, k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the chunks nearest in meaning to the query.

        Args:
            query: What to look for.
            lecture_id: Only search this lecture.
            source: Only search this kind of material.
            k: How many chunks to return.

        Returns:
            The chunks, nearest first, with the score rounded to four decimals.

        """
        validate_search_args(query, source, k)
        vector = embed_query(w, cfg.embedding_endpoint, query)
        # One transaction per search: it scopes semantic_search's SET LOCAL and rolls back after a failure,
        # so the shared connection stays usable for the next call.
        with conn.transaction():
            rows = semantic_search(conn, cfg, vector, k=k, lecture_id=lecture_id, source=source)
        return [{**row, "score": round(float(row["score"]), 4)} for row in rows]

    return search_course_content


def build_registry(conn: psycopg.Connection, cfg: ProjectConfig, w: WorkspaceClient) -> ToolRegistry:
    """Register the custom function tools.

    Args:
        conn: An open connection to the project's Postgres.
        cfg: The project configuration for the target environment.
        w: The workspace client.

    Returns:
        A registry holding `search_course_content`.

    """
    registry = ToolRegistry()
    registry.register(
        ToolInfo(
            name="search_course_content",
            spec=SEARCH_COURSE_CONTENT_SPEC,
            exec_fn=make_search_course_content(conn, cfg, w),
        )
    )
    return registry
