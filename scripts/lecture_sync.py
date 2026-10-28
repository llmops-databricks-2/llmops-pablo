"""Sync the lecture chunks into the Lakebase synced table `lecture_chunks_synced`.

The job's entry point for the `lecture_sync` task, which runs after `lecture_chunking`. The work happens in
`arxiv_curator.utils.lakebase`, adapted from the course's notebook 2.5: it enables pgvector, creates or
syncs the synced table, indexes it, and logs and checks its vector type, row count and indexes. Like the
other scripts, the job passes the deployed `project_config.yml` with `--config`.

Usage:
    python scripts/lecture_sync.py --env dev
"""

import argparse

from databricks.connect import DatabricksSession

from arxiv_curator.config import load_config
from arxiv_curator.utils.lakebase import sync_lecture_chunks


def main(env: str, config_path: str) -> None:
    """Run the sync for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    """
    spark = DatabricksSession.builder.getOrCreate()
    sync_lecture_chunks(spark, load_config(config_path, env))


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
