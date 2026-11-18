"""Test the agent's session memory in Lakebase: follow-ups, reload, no duplicates, isolation and ownership.

The week 3 check for session history. It creates the session tables, asks a question and a follow-up in
session A, reads A back through a new helper, saves a turn again to show nothing is duplicated, asks the same
follow-up in session B to show B sees nothing of A, and has another caller try to open A. It writes rows to
the session tables under `check-<uuid>` IDs. Run the `lecture_insights_assistant` job and deploy the bundle
first, then run it from a laptop with a CLI profile for the workspace; it runs as that profile's user.

Usage:
    DATABRICKS_CONFIG_PROFILE=<profile> python scripts/session_memory_check.py --env dev
"""

import argparse
import json
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import load_config
from arxiv_curator.utils.agent_functions import TRANSCRIPT_FUNCTION
from arxiv_curator.utils.agent_loop import AgentRun, run_turn
from arxiv_curator.utils.agent_tools import build_registry
from arxiv_curator.utils.lakebase import connect
from arxiv_curator.utils.llm import gateway_client
from arxiv_curator.utils.mcp_tools import functions_mcp_url, genie_mcp_url, mcp_tools
from arxiv_curator.utils.session_memory import MESSAGES_TABLE, SESSIONS_TABLE, SessionMemory

QUESTION = "What did the course say about how to split documents into chunks for retrieval?"
FOLLOW_UP = "Which lecture and slide page was that on? Answer in one sentence."


def _log_run(label: str, question: str, run: AgentRun) -> None:
    """Log a turn's question, tools and answer.

    Args:
        label: Which session and turn.
        question: The question.
        run: The turn's outcome.

    """
    tools = [call["name"] for call in run.tool_calls] or ["none"]
    logger.info(f"{label} Q: {question}")
    logger.info(f"{label}    tools: {', '.join(tools)}; new messages: {len(run.new_messages)}")
    logger.info(f"{label}    A: {run.answer}")


def _check_tool_pairs(messages: list[dict]) -> None:
    """Check every tool result follows an assistant message that made a call with its ID.

    Args:
        messages: The loaded session.

    Raises:
        RuntimeError: If a tool result has no matching call.

    """
    called: set[str] = set()
    for message in messages:
        called |= {call["id"] for call in message.get("tool_calls") or []}
        if message["role"] == "tool" and message["tool_call_id"] not in called:
            raise RuntimeError(f"Tool result {message['tool_call_id']} has no earlier call")


def main(env: str, config_path: str) -> None:
    """Run the session memory checks for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    Raises:
        RuntimeError: If a reload, the ordering, the duplicate check, isolation or the ownership check fails.

    """
    cfg = load_config(config_path, env)
    w = WorkspaceClient()
    caller = w.current_user.me().user_name
    memory = SessionMemory(cfg, w)
    session_a, session_b = f"check-{uuid4()}", f"check-{uuid4()}"

    logger.info("== 1. Session tables")
    memory.ensure_schema()
    with connect(cfg, w) as conn:
        indexes = conn.execute(
            "SELECT tablename, indexname FROM pg_indexes WHERE schemaname = %s AND tablename IN (%s, %s) ORDER BY 1",
            (cfg.db_schema, SESSIONS_TABLE, MESSAGES_TABLE),
        ).fetchall()
    logger.info(f"Indexes: {indexes}")

    with connect(cfg, w) as conn:
        registry = build_registry(conn, cfg, w)
        if cfg.genie_space_id:
            for tool in mcp_tools(w, genie_mcp_url(w, cfg.genie_space_id)):
                registry.register(tool)
        function_name = f"{cfg.catalog}__{cfg.db_schema}__{TRANSCRIPT_FUNCTION}"
        for tool in mcp_tools(w, functions_mcp_url(w, cfg), names={function_name}):
            registry.register(tool)
        client = gateway_client(w)

        logger.info(f"== 2. Session A ({session_a}): a question, then a follow-up that needs it")
        first = run_turn(client, cfg.llm_endpoint, registry, memory, session_a, caller, QUESTION)
        _log_run("A1", QUESTION, first)
        second = run_turn(client, cfg.llm_endpoint, registry, memory, session_a, caller, FOLLOW_UP)
        _log_run("A2", FOLLOW_UP, second)

        logger.info("== 3. Session A read back through a new helper")
        loaded = SessionMemory(cfg, w).load(session_a)
        expected = len(first.new_messages) + len(second.new_messages)
        with connect(cfg, w) as pg:
            seqs = [
                r[0]
                for r in pg.execute(
                    f'SELECT seq FROM "{cfg.db_schema}".{MESSAGES_TABLE} WHERE session_id = %s ORDER BY seq',
                    (session_a,),
                ).fetchall()
            ]
        logger.info(f"Loaded {len(loaded)} messages (expected {expected}); seq {seqs}")
        logger.info(f"Roles: {[m['role'] for m in loaded]}")
        if len(loaded) != expected or seqs != list(range(expected)):
            raise RuntimeError("Session A did not come back whole and in order")
        if loaded[0]["content"] != QUESTION or loaded[-1]["content"] != second.answer:
            raise RuntimeError("Session A does not start with the question and end with the last answer")
        _check_tool_pairs(loaded)

        logger.info("== 4. Saving the follow-up turn again")
        written = memory.save(session_a, len(first.new_messages), second.new_messages)
        logger.info(f"Written on the retry: {written}; messages now {len(memory.load(session_a))}")
        if written or len(memory.load(session_a)) != expected:
            raise RuntimeError("A repeated save added rows")

        logger.info(f"== 5. Session B ({session_b}): the same follow-up, with no history")
        if memory.load(session_b):
            raise RuntimeError("Session B has messages before its first turn")
        isolated = run_turn(client, cfg.llm_endpoint, registry, memory, session_b, caller, FOLLOW_UP)
        _log_run("B1", FOLLOW_UP, isolated)
        loaded_b = memory.load(session_b)
        if len(loaded_b) != len(isolated.new_messages) or loaded_b[0]["content"] != FOLLOW_UP:
            raise RuntimeError("Session B holds messages that are not its own")
        logger.info(f"Session B holds {len(loaded_b)} messages, none from A: {json.dumps(loaded_b[0])}")

        logger.info("== 6. Another caller opens session A")
        try:
            run_turn(client, cfg.llm_endpoint, registry, memory, session_a, "someone-else", FOLLOW_UP)
        except PermissionError as error:
            logger.info(f"Rejected before the model was called: {error}")
        else:
            raise RuntimeError("Another caller could continue session A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dev", choices=["dev", "acc", "prd"])
    parser.add_argument("--config", default="project_config.yml", help="Path to project_config.yml.")
    args = parser.parse_args()
    main(args.env, args.config)
