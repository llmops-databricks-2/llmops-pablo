"""Test the agent's MCP tools, the Genie space and `get_transcript_segment`, directly and then through the agent.

The week 3 check for the managed MCP tools. It lists each server's tools, calls the UC function and Genie
directly, including a follow-up in the same Genie conversation, then sends two questions through the agent with
every tool registered. It writes nothing. Run the `lecture_insights_assistant` job and deploy the bundle
first, then run it from a laptop with a CLI profile for the workspace.

Usage:
    DATABRICKS_CONFIG_PROFILE=<profile> python scripts/agent_mcp_check.py --env dev
"""

import argparse
import json
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import load_config
from arxiv_curator.utils.agent_functions import TRANSCRIPT_FUNCTION
from arxiv_curator.utils.agent_loop import run_agent
from arxiv_curator.utils.agent_tools import build_registry
from arxiv_curator.utils.lakebase import connect
from arxiv_curator.utils.llm import gateway_client
from arxiv_curator.utils.mcp_tools import functions_mcp_url, genie_mcp_url, mcp_tools

GENIE_QUESTION = "How many chat messages were sent per lecture?"
GENIE_FOLLOW_UP = "Which of those lectures was the longest, in minutes?"
AGENT_QUESTIONS = [
    ("How many chat messages were sent in each lecture?", "query_space_"),
    ("What exactly was said in the first two minutes of lecture-1?", TRANSCRIPT_FUNCTION),
]


def main(env: str, config_path: str) -> None:
    """Run the MCP checks for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    Raises:
        RuntimeError: If the config has no Genie space, a direct call fails, the follow-up starts a new
            conversation, or the agent does not choose Genie for the data question.

    """
    cfg = load_config(config_path, env)
    if not cfg.genie_space_id:
        raise RuntimeError(f"No genie_space_id for {env} in {config_path}; deploy the bundle and add it")
    w = WorkspaceClient()
    function_name = f"{cfg.catalog}__{cfg.db_schema}__{TRANSCRIPT_FUNCTION}"

    logger.info("== 1. MCP tools on each server (calls run as the CLI profile's user)")
    genie_tools = mcp_tools(w, genie_mcp_url(w, cfg.genie_space_id))
    function_tools = mcp_tools(w, functions_mcp_url(w, cfg), names={function_name})
    for tool in [*genie_tools, *function_tools]:
        params = list(tool.spec["function"]["parameters"].get("properties", {}))
        logger.info(f"{tool.name}({', '.join(params)})")

    with connect(cfg, w) as conn:
        registry = build_registry(conn, cfg, w)
        for tool in [*genie_tools, *function_tools]:
            registry.register(tool)

        logger.info(f"== 2. {TRANSCRIPT_FUNCTION} directly")
        args = json.dumps({"lecture_id": "lecture-1", "start_seconds": 0, "end_seconds": 120})
        result = json.loads(registry.execute(function_name, args))
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"{TRANSCRIPT_FUNCTION}: {result['error']}")
        logger.info(f"{function_name}({args}) -> {json.dumps(result)[:400]}")
        bad = json.dumps({"lecture_id": "lecture-1", "start_seconds": "soon"})
        logger.info(f"Controlled failure: {function_name}({bad}) -> {registry.execute(function_name, bad)[:200]}")

        logger.info("== 3. Genie directly through the SDK, with a follow-up in the same conversation")
        bound = timedelta(seconds=60)
        first = w.genie.start_conversation_and_wait(cfg.genie_space_id, GENIE_QUESTION, timeout=bound)
        follow = w.genie.create_message_and_wait(
            cfg.genie_space_id, first.conversation_id, GENIE_FOLLOW_UP, timeout=bound
        )
        for label, message in [("question", first), ("follow-up", follow)]:
            sql = next((a.query.query for a in message.attachments or [] if a.query), None)
            logger.info(
                f"{label}: conversation {message.conversation_id}, message {message.id}, "
                f"status {message.status}, SQL: {sql}"
            )
        if follow.conversation_id != first.conversation_id:
            raise RuntimeError("The follow-up started a new Genie conversation")

        logger.info("== 4. Genie directly through its MCP tool, with a follow-up")
        query_tool = genie_tools[0].name
        answer = json.loads(registry.execute(query_tool, json.dumps({"query": GENIE_QUESTION})))
        logger.info(f"question -> {json.dumps(answer)[:500]}")
        follow_args = {"query": GENIE_FOLLOW_UP, "conversation_id": answer["conversation_id"]}
        follow_answer = json.loads(registry.execute(query_tool, json.dumps(follow_args)))
        logger.info(f"follow-up -> {json.dumps(follow_answer)[:500]}")
        if follow_answer.get("conversation_id") != answer["conversation_id"]:
            raise RuntimeError("The MCP follow-up started a new Genie conversation")

        logger.info("== 5. Through the agent, with every tool registered")
        client = gateway_client(w)
        for question, expected in AGENT_QUESTIONS:
            run = run_agent(client, cfg.llm_endpoint, registry, question)
            tools = [call["name"] for call in run.tool_calls] or ["none"]
            logger.info(f"Q: {question}")
            logger.info(f"   tools: {', '.join(tools)}; rounds: {run.rounds}; hit limit: {run.hit_limit}")
            logger.info(f"   A: {run.answer}")
            chose = any(name.startswith(expected) or name.endswith(expected) for name in tools)
            if expected == "query_space_" and not chose:
                raise RuntimeError(f"The agent did not choose Genie for: {question}")
            if not chose:
                logger.warning(f"The agent did not choose {expected} for: {question}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dev", choices=["dev", "acc", "prd"])
    parser.add_argument("--config", default="project_config.yml", help="Path to project_config.yml.")
    args = parser.parse_args()
    main(args.env, args.config)
