"""Tests for the agent loop with a scripted fake model, without the AI Gateway."""

import json
from types import SimpleNamespace
from typing import Any

from arxiv_curator.utils.agent_loop import run_agent
from arxiv_curator.utils.agent_tools import ToolInfo, ToolRegistry


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """Build a tool call the way the OpenAI client returns it.

    Args:
        call_id: The call ID.
        name: The tool name.
        arguments: The arguments as a JSON string.

    Returns:
        The fake tool call.

    """
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """Build a chat completion with one choice.

    Args:
        content: The message text.
        tool_calls: The tool calls, if any.

    Returns:
        The fake completion.

    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")]
    )


class FakeClient:
    """A chat client that replays scripted responses and records each request."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = iter(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        """Record the request and return the next scripted response.

        Args:
            **kwargs: The request.

        Returns:
            The next response.

        """
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(self._responses)


def _registry() -> ToolRegistry:
    """Build a registry with one search tool that returns a fixed chunk.

    Returns:
        The registry.

    """
    registry = ToolRegistry()
    registry.register(
        ToolInfo(
            name="search_course_content",
            spec={"type": "function", "function": {"name": "search_course_content"}},
            exec_fn=lambda query, k=5: [{"lecture_id": "lecture-1", "page": 3, "excerpt": query}],
        )
    )
    return registry


def test_answers_directly_without_tools() -> None:
    """A model that answers at once uses no rounds and is offered the tools."""
    client = FakeClient([_response("Hello")])
    run = run_agent(client, "chat", _registry(), "Hi")
    assert (run.answer, run.rounds, run.hit_limit, run.tool_calls) == ("Hello", 0, False, [])
    assert client.requests[0]["tools"][0]["function"]["name"] == "search_course_content"


def test_runs_a_tool_and_returns_the_result_under_its_call_id() -> None:
    """A tool call is executed, kept as an assistant message, and answered by a tool message with its ID."""
    client = FakeClient(
        [
            _response(tool_calls=[_tool_call("c1", "search_course_content", '{"query": "rag"}')]),
            _response("Lecture 1, page 3."),
        ]
    )
    run = run_agent(client, "chat", _registry(), "Where is RAG?")
    assert run.answer == "Lecture 1, page 3."
    assert run.rounds == 1
    second = client.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": json.dumps([{"lecture_id": "lecture-1", "page": 3, "excerpt": "rag"}]),
    }


def test_bad_arguments_become_an_error_result() -> None:
    """Malformed arguments do not crash the loop; the model sees an error and can answer."""
    client = FakeClient(
        [
            _response(tool_calls=[_tool_call("c1", "search_course_content", "{query: rag")]),
            _response("The search failed."),
        ]
    )
    run = run_agent(client, "chat", _registry(), "Where is RAG?")
    assert run.answer == "The search failed."
    assert "error" in json.loads(run.tool_calls[0]["result"])


def test_empty_answer_is_asked_again_without_tools() -> None:
    """A reply with neither tool calls nor text gets one more call with tools off, not an empty answer."""
    client = FakeClient([_response(""), _response("Lecture 2, slide 35.")])
    run = run_agent(client, "chat", _registry(), "Where is context quality?")
    assert (run.answer, run.rounds, run.hit_limit) == ("Lecture 2, slide 35.", 0, False)
    assert client.requests[-1]["tool_choice"] == "none"


def test_empty_final_answer_is_retried_once() -> None:
    """An empty tools-off answer is retried once; a second empty one is returned empty, not looped on."""
    client = FakeClient([_response(""), _response(""), _response("")])
    run = run_agent(client, "chat", _registry(), "Where is context quality?")
    assert run.answer == ""
    assert len(client.requests) == 3


def test_stops_after_max_rounds_with_a_tools_off_answer() -> None:
    """A model that keeps calling tools is stopped at the limit and asked once more with tool_choice none."""
    calls = [_response(tool_calls=[_tool_call(f"c{i}", "search_course_content", '{"query": "rag"}')]) for i in range(2)]
    client = FakeClient([*calls, _response("Best effort answer.")])
    run = run_agent(client, "chat", _registry(), "Where is RAG?", max_rounds=2)
    assert (run.answer, run.rounds, run.hit_limit) == ("Best effort answer.", 2, True)
    assert len(client.requests) == 3
    assert client.requests[-1]["tool_choice"] == "none"
