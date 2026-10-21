"""Parse the raw lecture files in the volume into the lecture assistant's Delta tables.

The job's entry point for the `lecture_ingestion` task. The work happens in
`arxiv_curator.utils.lecture_ingestion`, which `notebooks/1.3_lecture_data_ingestion.py` also uses.
The job passes the deployed `project_config.yml` with `--config`: serverless runs this file without
`__file__`, so the script cannot find the config relative to itself.

Usage:
    python scripts/lecture_ingestion.py --env dev
"""

import argparse

from databricks.connect import DatabricksSession

from arxiv_curator.config import load_config
from arxiv_curator.utils.lecture_ingestion import ingest_lectures


def main(env: str, config_path: str) -> None:
    """Run the ingestion for one environment.

    Args:
        env: The target environment (dev, acc or prd).
        config_path: Path to `project_config.yml`.

    """
    spark = DatabricksSession.builder.getOrCreate()
    ingest_lectures(spark, load_config(config_path, env))


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
