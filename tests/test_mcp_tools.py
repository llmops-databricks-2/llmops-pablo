"""Tests for the MCP tool adapters and the UC function SQL, without a workspace."""

import json
from typing import Any

import pytest

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.agent_functions import transcript_function_sql
from arxiv_curator.utils.mcp_tools import MAX_ROWS, POLL_LIMIT_SECONDS, condense_genie, make_genie_exec

QUERY, POLL = "query_space_x", "poll_response_x"


def _message(status: str, **content: Any) -> str:  # noqa: ANN401
    """Build a Genie MCP result as the server returns it.

    Args:
        status: The message status.
        **content: The content fields, such as `textAttachments`.

    Returns:
        The result as a JSON string.

    """
    return json.dumps({"status": status, "conversationId": "c1", "messageId": "m1", "content": content})


class FakeGenie:
    """Replays Genie results and records every tool call."""

    def __init__(self, results: list[str]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._results = iter(results)

    def __call__(self, name: str, args: dict[str, Any]) -> str:
        """Record the call and return the next result.

        Args:
            name: The tool name.
            args: The arguments.

        Returns:
            The next scripted result.

        """
        self.calls.append((name, args))
        return next(self._results)


class FakeClock:
    """A clock that only moves when the code sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        """Advance the clock.

        Args:
            seconds: How far to advance.

        """
        self.now += seconds


def test_condense_genie_keeps_text_sql_and_capped_rows() -> None:
    """Plain-string and object text attachments are kept, and rows are capped at MAX_ROWS."""
    rows = [[i] for i in range(MAX_ROWS + 5)]
    result = json.loads(
        _message(
            "COMPLETED",
            textAttachments=["25 messages", {"content": "on 12 October"}],
            queryAttachments=[
                {
                    "query": "SELECT 1",
                    "description": "Count",
                    "statement_response": {
                        "manifest": {"schema": {"columns": [{"name": "n"}]}},
                        "result": {"data_array": rows},
                    },
                }
            ],
        )
    )
    condensed = condense_genie(result)
    assert condensed["text"] == ["25 messages", "on 12 October"]
    assert (condensed["sql"], condensed["columns"], condensed["row_count"]) == ("SELECT 1", ["n"], MAX_ROWS + 5)
    assert len(condensed["rows"]) == MAX_ROWS


def test_genie_asks_once_when_completed() -> None:
    """A completed answer is returned after one query call, and a follow-up passes its conversation ID."""
    genie = FakeGenie([_message("COMPLETED", textAttachments=["done"])])
    ask = make_genie_exec(genie, QUERY, POLL, sleep=FakeClock().sleep, clock=lambda: 0.0)
    assert ask("How many?", conversation_id="c1")["text"] == ["done"]
    assert genie.calls == [(QUERY, {"query": "How many?", "conversation_id": "c1"})]


def test_genie_polls_the_same_message_without_resubmitting() -> None:
    """A pending answer is polled by its IDs until done; the question is sent only once."""
    genie = FakeGenie([_message("ASKING_AI"), _message("EXECUTING_QUERY"), _message("COMPLETED")])
    clock = FakeClock()
    assert make_genie_exec(genie, QUERY, POLL, sleep=clock.sleep, clock=lambda: clock.now)("How many?")["status"] == (
        "COMPLETED"
    )
    assert [name for name, _ in genie.calls] == [QUERY, POLL, POLL]
    assert genie.calls[1][1] == {"conversation_id": "c1", "message_id": "m1"}


def test_genie_stops_polling_at_the_bound() -> None:
    """A message still pending after POLL_LIMIT_SECONDS raises TimeoutError instead of waiting forever."""
    genie = FakeGenie([_message("ASKING_AI")] * 100)
    clock = FakeClock()
    with pytest.raises(TimeoutError, match="still pending"):
        make_genie_exec(genie, QUERY, POLL, sleep=clock.sleep, clock=lambda: clock.now)("How many?")
    assert clock.now <= POLL_LIMIT_SECONDS
    assert [name for name, _ in genie.calls].count(QUERY) == 1


def test_transcript_function_leaves_out_speakers_and_caps_turns() -> None:
    """The UC function reads the schema's transcript table, returns no speaker and limits the turns."""
    cfg = ProjectConfig.model_validate(
        {
            "catalog": "llmops_dev",
            "schema": "pablo",
            "volume": "v",
            "llm_endpoint": "chat",
            "embedding_endpoint": "embed",
            "embedding_service": "embed",
            "experiment_path": "/x",
            "warehouse_id": "w",
            "github_org": "o",
            "github_secret_scope": "s",
            "github_secret_key": "k",
            "lakebase_project_id": "lb",
        }
    )
    sql = transcript_function_sql(cfg)
    assert "CREATE OR REPLACE FUNCTION llmops_dev.pablo.get_transcript_segment(" in sql
    assert "FROM llmops_dev.pablo.lecture_transcript" in sql
    assert "speaker" not in sql
    assert "LIMIT 100" in sql
