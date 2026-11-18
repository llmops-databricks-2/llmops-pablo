"""Create the agent's Unity Catalog functions, such as `get_transcript_segment`.

The job's entry point for the `agent_functions` task, which runs after `lecture_sync`. The work happens in
`arxiv_curator.utils.agent_functions`. Like the other scripts, the job passes the deployed
`project_config.yml` with `--config`.

Usage:
    python scripts/agent_functions.py --env dev
"""

import argparse

from databricks.connect import DatabricksSession

from arxiv_curator.config import load_config
from arxiv_curator.utils.agent_functions import create_agent_functions


def main(env: str, config_path: str) -> None:
    """Create the functions for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    """
    spark = DatabricksSession.builder.getOrCreate()
    create_agent_functions(spark, load_config(config_path, env))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dev", choices=["dev", "acc", "prd"])
    parser.add_argument(
        "--config",
        default="project_config.yml",
        help="Path to project_config.yml; the job passes the deployed copy.",
    )
    args = parser.parse_args()
    main(args.env, args.config)
