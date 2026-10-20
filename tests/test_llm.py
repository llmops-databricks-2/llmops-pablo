"""Tests for the gateway helpers in arxiv_curator.utils.llm."""

from openai.types.chat import ChatCompletion

from arxiv_curator.utils.llm import response_text


def make_completion(content: object) -> ChatCompletion:
    """Build a minimal chat completion with the given message content.

    Args:
        content: The message content, a string or a list of parts.

    Returns:
        The chat completion.

    """
    return ChatCompletion.model_construct(
        choices=[
            type("Choice", (), {"message": type("Message", (), {"content": content})()})(),
        ]
    )


def test_response_text_keeps_only_text_parts_of_reasoning_models() -> None:
    """Drop GPT-OSS reasoning summaries and join the text parts."""
    content = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking."}]},
        {"type": "text", "text": "gateway "},
        {"type": "text", "text": "ok"},
    ]
    assert response_text(make_completion(content)) == "gateway ok"


def test_response_text_returns_plain_string_content() -> None:
    """Return string content as is, and an empty string for no content."""
    assert response_text(make_completion("plain answer")) == "plain answer"
    assert response_text(make_completion(None)) == ""
