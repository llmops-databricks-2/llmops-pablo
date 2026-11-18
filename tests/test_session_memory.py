"""Tests for the session turn and the session table DDL, with a fake model and an in-memory store."""

import json
from typing import Any

import pytest
from test_agent_loop import FakeClient, _registry, _response, _tool_call

from arxiv_curator.utils.agent_loop import run_turn
from arxiv_curator.utils.session_memory import schema_statements


class FakeMemory:
    """An in-memory session store with the same interface as `SessionMemory`."""

    def __init__(self, fail_load: bool = False, fail_save: bool = False) -> None:
        self.owners: dict[str, str] = {}
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        self.fail_load, self.fail_save = fail_load, fail_save

    def claim(self, session_id: str, caller: str) -> None:
        """Create the session for the caller, or check the caller owns it.

        Args:
            session_id: The session ID.
            caller: The caller.

        Raises:
            PermissionError: If another caller owns the session.

        """
        if self.owners.setdefault(session_id, caller) != caller:
            raise PermissionError(f"Session {session_id} belongs to another caller")

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return the session's messages ordered by position.

        Args:
            session_id: The session ID.

        Returns:
            The messages.

        Raises:
            ConnectionError: If the store was built to fail loads.

        """
        if self.fail_load:
            raise ConnectionError("Lakebase unavailable")
        return [message for (sid, _), message in sorted(self.rows.items()) if sid == session_id]

    def save(self, session_id: str, start_seq: int, messages: list[dict[str, Any]]) -> int:
        """Store messages at free positions only.

        Args:
            session_id: The session ID.
            start_seq: The first message's position.
            messages: The messages.

        Returns:
            The number written.

        Raises:
            ConnectionError: If the store was built to fail saves.

        """
        if self.fail_save:
            raise ConnectionError("Lakebase unavailable")
        written = 0
        for offset, message in enumerate(messages):
            if (session_id, start_seq + offset) not in self.rows:
                self.rows[(session_id, start_seq + offset)] = message
                written += 1
        return written


def _turn(memory: FakeMemory, responses: list, question: str, session_id: str = "a", caller: str = "me") -> Any:  # noqa: ANN401
    """Run one turn with a scripted model.

    Args:
        memory: The session store.
        responses: The model's scripted responses.
        question: The question.
        session_id: The session ID.
        caller: The caller.

    Returns:
        The fake client, to inspect its requests.

    """
    client = FakeClient(responses)
    run_turn(client, "m", _registry(), memory, session_id, caller, question)
    return client


def test_turn_saves_the_question_tool_messages_and_answer_in_order() -> None:
    """A turn with a tool round saves the question, the call, its result under the call ID, and the answer."""
    memory = FakeMemory()
    call = _tool_call("call_1", "search_course_content", json.dumps({"query": "chunking"}))
    _turn(memory, [_response(tool_calls=[call]), _response("Lecture 2, page 4.")], "Where is chunking?")
    saved = memory.load("a")
    assert [m["role"] for m in saved] == ["user", "assistant", "tool", "assistant"]
    assert saved[1]["tool_calls"][0]["id"] == saved[2]["tool_call_id"] == "call_1"
    assert saved[-1]["content"] == "Lecture 2, page 4."


def test_next_turn_sends_the_history_and_saves_only_new_messages() -> None:
    """The second turn's request carries the first turn, and only the second turn's messages are added."""
    memory = FakeMemory()
    _turn(memory, [_response("Chunking is in lecture 2.")], "Where is chunking?")
    client = _turn(memory, [_response("Page 4.")], "Which page?")
    sent = client.requests[0]["messages"]
    assert [m["content"] for m in sent[1:]] == ["Where is chunking?", "Chunking is in lecture 2.", "Which page?"]
    assert sorted(memory.rows) == [("a", 0), ("a", 1), ("a", 2), ("a", 3)]


def test_second_session_sees_nothing_of_the_first() -> None:
    """A turn in another session sends no message from the first session."""
    memory = FakeMemory()
    _turn(memory, [_response("Chunking is in lecture 2.")], "Where is chunking?")
    client = _turn(memory, [_response("Which topic do you mean?")], "Which page?", session_id="b")
    assert [m["content"] for m in client.requests[0]["messages"][1:]] == ["Which page?"]
    assert len(memory.load("b")) == 2


def test_another_caller_is_rejected_before_the_model_is_called() -> None:
    """Only the session's owner may continue it."""
    memory = FakeMemory()
    _turn(memory, [_response("Hi.")], "Hello")
    client = FakeClient([])
    with pytest.raises(PermissionError):
        run_turn(client, "m", _registry(), memory, "a", "someone-else", "What did they ask?")
    assert client.requests == []


def test_failed_load_stops_the_turn() -> None:
    """A history that cannot be loaded stops the turn instead of answering without it."""
    client = FakeClient([])
    with pytest.raises(ConnectionError):
        run_turn(client, "m", _registry(), FakeMemory(fail_load=True), "a", "me", "Which page?")
    assert client.requests == []


def test_failed_save_still_returns_the_answer() -> None:
    """The lecturer gets the answer even when the save fails."""
    run = run_turn(FakeClient([_response("Page 4.")]), "m", _registry(), FakeMemory(fail_save=True), "a", "me", "Q")
    assert run.answer == "Page 4."


def test_schema_keys_messages_by_session_and_position() -> None:
    """The message table's primary key is (session_id, seq), and sessions carry an owner."""
    sessions, messages = schema_statements("pablo")
    assert '"pablo".sessions' in sessions and "owner TEXT NOT NULL" in sessions
    assert '"pablo".session_messages' in messages and "PRIMARY KEY (session_id, seq)" in messages
