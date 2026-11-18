"""The agent's tool-calling loop, following proposal/ORCHESTRATION.md without the session steps.

Code runs the loop; the model chooses whether to answer or call tools. Each assistant tool-call message is
kept, and every result goes back under its call ID. Failed calls return an error result the model can react
to. After `max_rounds` tool rounds, one last call with tools turned off makes the model answer from what it
has. Session history, its ownership check and Genie come with later issues.
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import OpenAI

from arxiv_curator.utils.agent_tools import ToolRegistry
from arxiv_curator.utils.llm import response_text

MAX_ROUNDS = 5
FINAL_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You help a lecturer understand their own course. Answer questions about what was taught, explained or "
    "said in the lectures by searching the course content with your tools, and read the transcript around a hit "
    "when you need more of what was said. Use the Genie tool for counts, dates and other data questions about "
    "lectures, the chat and GitHub activity; pass its conversation_id to ask a follow-up. Cite each claim with its lecture and "
    "its slide page or transcript time. If a tool returns an error, try again with corrected arguments or tell "
    "the lecturer what failed. If the search finds nothing relevant, say that the course material does not "
    "cover the question; do not answer from general knowledge."
)


@dataclass
class AgentRun:
    """The outcome of one question: the answer and every tool call made on the way."""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    hit_limit: bool = False


def run_agent(
    client: OpenAI,
    model: str,
    registry: ToolRegistry,
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_rounds: int = MAX_ROUNDS,
) -> AgentRun:
    """Answer one question, letting the model call the registry's tools for up to `max_rounds` rounds.

    Args:
        client: An OpenAI-compatible client for the AI Gateway, from `arxiv_curator.utils.llm.gateway_client`.
        model: The chat model service, such as `course_ops.gateway.chat`.
        registry: The tools the model may call.
        question: The lecturer's question.
        system_prompt: The instructions the model follows.
        max_rounds: The most tool rounds before the final tools-off answer.

    Returns:
        The answer, each tool call with its arguments and result, the rounds used and whether the limit was hit.

    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    run = AgentRun(answer="")
    for _ in range(max_rounds):
        response = client.chat.completions.create(model=model, messages=messages, tools=registry.specs())
        message = response.choices[0].message
        if not message.tool_calls:
            run.answer = response_text(response)
            if run.answer:
                return run
            # GPT-OSS sometimes stops with neither tool calls nor text; ask once more, without tools.
            logger.warning(
                f"Empty answer (finish_reason {response.choices[0].finish_reason}); asking again without tools"
            )
            break
        run.rounds += 1
        messages.append(
            {
                "role": "assistant",
                "content": response_text(response) or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            result = registry.execute(call.function.name, call.function.arguments)
            logger.info(f"Round {run.rounds}: {call.function.name}({call.function.arguments}) -> {result[:300]}")
            run.tool_calls.append(
                {
                    "round": run.rounds,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                    "result": result,
                }
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    else:
        run.hit_limit = True
        logger.warning(f"Reached {max_rounds} tool rounds; asking for a final answer without tools")

    # The limit is reached, or the answer came back empty: a call with tools turned off answers from what was
    # gathered. It is tried FINAL_ATTEMPTS times, because an empty reply can come back from any call.
    for attempt in range(1, FINAL_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=registry.specs(), tool_choice="none"
        )
        run.answer = response_text(response)
        if run.answer:
            break
        logger.warning(f"Empty final answer on attempt {attempt} of {FINAL_ATTEMPTS}")
    return run
