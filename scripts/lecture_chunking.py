"""Chunk the parsed lecture tables into the lecture assistant's `lecture_chunks` table.

The job's entry point for the `lecture_chunking` task, which runs after `lecture_ingestion`. The
work happens in `arxiv_curator.utils.lecture_chunking`. Like `lecture_ingestion.py`, the job
passes the deployed `project_config.yml` with `--config`.

Usage:
    python scripts/lecture_chunking.py --env dev
"""

import argparse

from databricks.connect import DatabricksSession

from arxiv_curator.config import load_config
from arxiv_curator.utils.lecture_chunking import chunk_lectures


def main(env: str, config_path: str) -> None:
    """Run the chunking for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    """
    spark = DatabricksSession.builder.getOrCreate()
    chunk_lectures(spark, load_config(config_path, env))


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
