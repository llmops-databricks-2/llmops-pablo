"""Parse the raw lecture files in the volume into the lecture assistant's Delta tables.

The job's entry point for the `lecture_ingestion` task. The work happens in
`arxiv_curator.utils.lecture_ingestion`, which `notebooks/1.3_lecture_data_ingestion.py` also uses.

Usage:
    python scripts/lecture_ingestion.py --env dev
"""

import argparse
from pathlib import Path

from databricks.connect import DatabricksSession

from arxiv_curator.config import load_config
from arxiv_curator.utils.lecture_ingestion import ingest_lectures

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def main(env: str) -> None:
    """Run the ingestion for one environment.

    Args:
        env: The target environment (dev, acc or prd).

    """
    spark = DatabricksSession.builder.getOrCreate()
    ingest_lectures(spark, load_config(str(PROJECT_CONFIG), env))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dev", choices=["dev", "acc", "prd"])
    main(parser.parse_args().env)
