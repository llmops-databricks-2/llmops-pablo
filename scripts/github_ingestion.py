"""Merge the course organization's new GitHub events into the github_events table.

The entry point of the `github_events` job. The work happens in
`arxiv_curator.utils.github_ingestion`. The GitHub token is read from the Databricks secret scope
named in `project_config.yml`, never from code. As in `lecture_ingestion.py`, the job passes the
deployed `project_config.yml` with `--config`.

Usage:
    python scripts/github_ingestion.py --env dev
"""

import argparse

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from arxiv_curator.config import load_config
from arxiv_curator.utils.github_ingestion import ingest_github_events


def main(env: str, config_path: str) -> None:
    """Run the GitHub ingestion for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    """
    cfg = load_config(config_path, env)
    token = WorkspaceClient().dbutils.secrets.get(cfg.github_secret_scope, cfg.github_secret_key)
    spark = DatabricksSession.builder.getOrCreate()
    ingest_github_events(spark, cfg, token)


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
