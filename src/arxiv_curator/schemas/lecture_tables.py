"""Spark schemas of the lecture tables written by the ingestion.

`lecture_slides` has no schema here: its columns come from the `ai_parse_document` query.
Every table also gets an `ingested_at` timestamp when it is written.
"""

from pyspark.sql.types import (
    BooleanType,
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

LECTURES = StructType(
    [
        StructField("lecture_id", StringType(), False),
        StructField("lecture_date", DateType(), False),
        StructField("recording_start_utc", TimestampType(), False),
        StructField("duration_seconds", IntegerType(), False),
        StructField("source_path", StringType(), False),
    ]
)

LECTURE_TRANSCRIPT = StructType(
    [
        StructField("lecture_id", StringType(), False),
        StructField("turn", IntegerType(), False),
        StructField("speaker", StringType(), False),
        StructField("offset_seconds", IntegerType(), False),
        StructField("ts_utc", TimestampType(), False),
        StructField("text", StringType(), True),
        StructField("source_path", StringType(), False),
    ]
)

LECTURE_CHAT = StructType(
    [
        StructField("lecture_id", StringType(), False),
        StructField("message", IntegerType(), False),
        StructField("sender", StringType(), True),
        StructField("ts_utc", TimestampType(), False),
        StructField("offset_seconds", IntegerType(), False),
        StructField("text_truncated", BooleanType(), False),
        StructField("text", StringType(), True),
        StructField("source_path", StringType(), False),
    ]
)

# Table name → schema, for the tables built from Python rows.
TABLE_SCHEMAS = {
    "lectures": LECTURES,
    "lecture_transcript": LECTURE_TRANSCRIPT,
    "lecture_chat": LECTURE_CHAT,
}
