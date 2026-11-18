"""Tests for the custom function tools' spec, validation and registry, without a database."""

import json

import pytest

from arxiv_curator.utils.agent_tools import (
    SEARCH_COURSE_CONTENT_SPEC,
    SOURCES,
    ToolInfo,
    ToolRegistry,
    validate_search_args,
)


def _registry() -> ToolRegistry:
    """Build a registry with one echo tool and one tool that always fails.

    Returns:
        The registry.

    """

    def echo(query: str, k: int = 5) -> dict:
        """Return the arguments it was called with."""
        return {"query": query, "k": k}

    def broken(query: str) -> None:
        """Fail the way a database error would."""
        raise RuntimeError("connection lost")

    registry = ToolRegistry()
    registry.register(ToolInfo(name="echo", spec={"function": {"name": "echo"}}, exec_fn=echo))
    registry.register(ToolInfo(name="broken", spec={"function": {"name": "broken"}}, exec_fn=broken))
    return registry


def test_search_spec_matches_the_openai_function_format() -> None:
    """The spec names the tool, requires only the query and offers the known sources."""
    function = SEARCH_COURSE_CONTENT_SPEC["function"]
    assert SEARCH_COURSE_CONTENT_SPEC["type"] == "function"
    assert function["name"] == "search_course_content"
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["properties"]["source"]["enum"] == list(SOURCES)


@pytest.mark.parametrize(
    ("query", "source", "k", "message"),
    [
        ("", None, 5, "non-empty"),
        ("   ", None, 5, "non-empty"),
        ("guardrails", "emails", 5, "source must be one of"),
        ("guardrails", None, 0, "k must be an integer"),
        ("guardrails", None, 11, "k must be an integer"),
        ("guardrails", None, True, "k must be an integer"),
    ],
)
def test_validate_search_args_rejects_bad_input(query: str, source: str | None, k: int, message: str) -> None:
    """Empty queries, unknown sources and out-of-range k are refused before any embedding call."""
    with pytest.raises(ValueError, match=message):
        validate_search_args(query, source, k)


def test_validate_search_args_accepts_good_input() -> None:
    """A query with a known source and k in range passes."""
    validate_search_args("guardrails", "slides", 10)


def test_registry_returns_the_result_as_json() -> None:
    """A good call runs the tool with the parsed arguments and returns its result as JSON."""
    assert json.loads(_registry().execute("echo", '{"query": "rag", "k": 2}')) == {"query": "rag", "k": 2}


def test_registry_lists_the_specs() -> None:
    """Every registered tool's spec is offered to the model."""
    assert [spec["function"]["name"] for spec in _registry().specs()] == ["echo", "broken"]


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("search_papers", "{}", "Unknown tool"),
        ("echo", "{query: rag", "not valid JSON"),
        ("echo", '["rag"]', "must be a JSON object"),
        ("echo", '{"query": "rag", "year": 2024}', "Invalid arguments"),
        ("broken", '{"query": "rag"}', "RuntimeError: connection lost"),
    ],
)
def test_registry_turns_failures_into_error_results(name: str, arguments: str, message: str) -> None:
    """Unknown tools, bad JSON, wrong arguments and tool failures return an error instead of raising."""
    result = json.loads(_registry().execute(name, arguments))
    assert message in result["error"]
