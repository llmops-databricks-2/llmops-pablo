"""Spark schema of the GitHub events table written by the GitHub ingestion.

The table also gets an `ingested_at` timestamp when rows are written.
"""

from pyspark.sql.types import StringType, StructField, StructType, TimestampType

GITHUB_EVENTS = StructType(
    [
        StructField("id", StringType(), False),
        StructField("type", StringType(), False),
        StructField("repo", StringType(), False),
        StructField("actor", StringType(), True),
        StructField("created_at", TimestampType(), False),
        StructField("payload", StringType(), True),
    ]
)
