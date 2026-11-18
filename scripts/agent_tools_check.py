"""Test the custom function tools directly, then through the agent, and log every call and result.

The week 3 check for `search_course_content`: each case runs the callable through the registry first, the
way the agent will, then two questions go through the agent loop. It needs `lecture_chunks_synced` in
Lakebase, so run the `lecture_insights_assistant` job first. It writes nothing. Run it from a laptop with a
CLI profile for the workspace.

Usage:
    DATABRICKS_CONFIG_PROFILE=<profile> python scripts/agent_tools_check.py --env dev
"""

import argparse
import json

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import load_config
from arxiv_curator.utils.agent_loop import run_agent
from arxiv_curator.utils.agent_tools import build_registry
from arxiv_curator.utils.lakebase import connect
from arxiv_curator.utils.llm import gateway_client

# Direct calls: the arguments the model would send, as JSON. The last one asks about something the lectures
# never cover, to record how low its best score is.
DIRECT_CASES = [
    ("normal query", {"query": "Why can adding more context make answers worse?"}),
    ("lecture filter", {"query": "provisioned throughput", "lecture_id": "lecture-1", "k": 3}),
    ("source filter", {"query": "How should chunks be checked before indexing?", "source": "slides", "k": 3}),
    ("not in the course", {"query": "How do I deploy the model to Kubernetes with Helm?", "k": 3}),
]

# Controlled failures: each must come back as {"error": ...} without raising.
FAILURE_CASES = [
    ("empty query", "search_course_content", json.dumps({"query": "  "})),
    ("unknown source", "search_course_content", json.dumps({"query": "guardrails", "source": "emails"})),
    ("k out of range", "search_course_content", json.dumps({"query": "guardrails", "k": 0})),
    ("unexpected argument", "search_course_content", json.dumps({"query": "guardrails", "year": 2024})),
    ("arguments not JSON", "search_course_content", "{query: guardrails"),
    ("unknown tool", "search_papers", json.dumps({"query": "guardrails"})),
]

AGENT_QUESTIONS = [
    "When did the lectures explain why adding more context can make answers worse, and what was said?",
    "How do I deploy the model to Kubernetes with Helm?",
]


def main(env: str, config_path: str) -> None:
    """Run the direct, failure and agent checks for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    Raises:
        RuntimeError: If a direct case fails, a failure case does not return an error result, or the agent
            returns an empty answer.

    """
    cfg = load_config(config_path, env)
    w = WorkspaceClient()
    with connect(cfg, w) as conn:
        registry = build_registry(conn, cfg, w)

        logger.info("== 1. Direct calls to search_course_content")
        for label, args in DIRECT_CASES:
            result = json.loads(registry.execute("search_course_content", json.dumps(args)))
            if isinstance(result, dict):
                raise RuntimeError(f"{label}: expected chunks, got {result}")
            logger.info(f"{label}: {json.dumps(args)} -> {len(result)} chunks, top score {result[0]['score']}")
            for chunk in result:
                where = f"page {chunk['page']}" if chunk["page"] else f"{chunk['start_offset_seconds']}s"
                logger.info(
                    f"   {chunk['score']:.4f} {chunk['lecture_id']} {chunk['source']} {where}: {chunk['excerpt'][:90]!r}"
                )

        logger.info("== 2. Controlled failures")
        for label, name, arguments in FAILURE_CASES:
            result = json.loads(registry.execute(name, arguments))
            if not isinstance(result, dict) or "error" not in result:
                raise RuntimeError(f"{label}: expected an error result, got {result}")
            logger.info(f"{label}: {name}({arguments}) -> {result['error']}")

        logger.info("== 3. Through the agent")
        client = gateway_client(w)
        for question in AGENT_QUESTIONS:
            run = run_agent(client, cfg.llm_endpoint, registry, question)
            tools = [call["name"] for call in run.tool_calls] or ["none"]
            logger.info(f"Q: {question}")
            logger.info(f"   tools: {', '.join(tools)}; rounds: {run.rounds}; hit limit: {run.hit_limit}")
            logger.info(f"   A: {run.answer}")
            if not run.answer:
                raise RuntimeError(f"Empty answer to: {question}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dev", choices=["dev", "acc", "prd"])
    parser.add_argument("--config", default="project_config.yml", help="Path to project_config.yml.")
    args = parser.parse_args()
    main(args.env, args.config)
