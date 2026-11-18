"""The agent's tool-calling loop and its session turn, following proposal/ORCHESTRATION.md.

Code runs the loop; the model chooses whether to answer or call tools. Each assistant tool-call message is
kept, and every result goes back under its call ID. Failed calls return an error result the model can react
to. After `max_rounds` tool rounds, one last call with tools turned off makes the model answer from what it
has. `run_turn` wraps the loop in the session steps: check the caller owns the session, load its history,
answer, and save only the new messages.
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import OpenAI

from arxiv_curator.utils.agent_tools import ToolRegistry
from arxiv_curator.utils.llm import response_text
from arxiv_curator.utils.session_memory import SessionMemory

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
    """The outcome of one question: the answer, every tool call made on the way, and the messages to save."""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    hit_limit: bool = False
    new_messages: list[dict[str, Any]] = field(default_factory=list)


def run_agent(
    client: OpenAI,
    model: str,
    registry: ToolRegistry,
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_rounds: int = MAX_ROUNDS,
    history: list[dict[str, Any]] | None = None,
) -> AgentRun:
    """Answer one question, letting the model call the registry's tools for up to `max_rounds` rounds.

    Args:
        client: An OpenAI-compatible client for the AI Gateway, from `arxiv_curator.utils.llm.gateway_client`.
        model: The chat model service, such as `course_ops.gateway.chat`.
        registry: The tools the model may call.
        question: The lecturer's question.
        system_prompt: The instructions the model follows.
        max_rounds: The most tool rounds before the final tools-off answer.
        history: The session's earlier messages, placed between the system prompt and the question.

    Returns:
        The answer, each tool call with its arguments and result, the rounds used, whether the limit was hit,
        and the new messages: the question, each tool-call message and result, and the answer.

    """
    history = history or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": question},
    ]
    first_new = 1 + len(history)
    run = AgentRun(answer="")
    for _ in range(max_rounds):
        response = client.chat.completions.create(model=model, messages=messages, tools=registry.specs())
        message = response.choices[0].message
        if not message.tool_calls:
            run.answer = response_text(response)
            if run.answer:
                messages.append({"role": "assistant", "content": run.answer})
                run.new_messages = messages[first_new:]
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
    messages.append({"role": "assistant", "content": run.answer})
    run.new_messages = messages[first_new:]
    return run


def run_turn(
    client: OpenAI,
    model: str,
    registry: ToolRegistry,
    memory: SessionMemory,
    session_id: str,
    caller: str,
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_rounds: int = MAX_ROUNDS,
) -> AgentRun:
    """Answer one question in a session: check the owner, load the history, answer, save the new messages.

    Args:
        client: An OpenAI-compatible client for the AI Gateway.
        model: The chat model service, such as `course_ops.gateway.chat`.
        registry: The tools the model may call.
        memory: The session store.
        session_id: The session ID.
        caller: The authenticated caller's name, the session's owner.
        question: The lecturer's question.
        system_prompt: The instructions the model follows.
        max_rounds: The most tool rounds before the final tools-off answer.

    Returns:
        The run, as from `run_agent`.

    Raises:
        PermissionError: If another caller owns the session; the model is not called.

    """
    memory.claim(session_id, caller)
    # A failed load raises and stops the turn: an empty history would hide the failure and break follow-ups.
    history = memory.load(session_id)
    run = run_agent(client, model, registry, question, system_prompt, max_rounds, history=history)
    try:
        memory.save(session_id, len(history), run.new_messages)
    except Exception as error:  # noqa: BLE001 - the lecturer still gets the answer; the next turn shows the gap
        logger.warning(f"Session {session_id}: the turn's messages were not saved: {error}")
    return run
