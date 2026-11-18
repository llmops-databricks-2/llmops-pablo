"""Unity Catalog functions the agent calls through the managed UC functions MCP server.

`get_transcript_segment` is the v1 tool set's UC function (proposal/AGENT_TOOLS.md): a SQL table function
over `lecture_transcript` that returns a lecture's turns between two offsets, so the agent can read around a
search hit before citing it. It leaves out `speaker`, so it never names a student. UC grants govern it: the
caller needs EXECUTE on the function and SELECT on the table. The MCP server for the schema is
`{host}/api/2.0/mcp/functions/<catalog>/<schema>`.
"""

from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig

TRANSCRIPT_FUNCTION = "get_transcript_segment"
MAX_TURNS = 100


def transcript_function_sql(cfg: ProjectConfig) -> str:
    """Return the statement that creates or replaces `get_transcript_segment`.

    Parameters share names with the table's columns, so the body qualifies them with the function name.

    Args:
        cfg: The project configuration for the target environment.

    Returns:
        The `CREATE OR REPLACE FUNCTION` statement.

    """
    schema = cfg.full_schema_name
    return f"""
CREATE OR REPLACE FUNCTION {schema}.{TRANSCRIPT_FUNCTION}(
  lecture_id STRING COMMENT 'The lecture, such as lecture-2.',
  start_seconds INT COMMENT 'Start of the segment, in seconds from the start of the recording.',
  end_seconds INT COMMENT 'End of the segment, in seconds from the start of the recording.'
)
RETURNS TABLE (turn INT, offset_seconds INT, text STRING)
COMMENT 'Return what was said in a lecture between two offsets in seconds, turn by turn, at most {MAX_TURNS} turns. Use it to read around a transcript search hit before quoting or citing it.'
RETURN
  SELECT t.turn, t.offset_seconds, t.text
  FROM {schema}.lecture_transcript t
  WHERE t.lecture_id = {TRANSCRIPT_FUNCTION}.lecture_id
    AND t.offset_seconds BETWEEN {TRANSCRIPT_FUNCTION}.start_seconds AND {TRANSCRIPT_FUNCTION}.end_seconds
  ORDER BY t.turn
  LIMIT {MAX_TURNS}
""".strip()


def create_agent_functions(spark: SparkSession, cfg: ProjectConfig) -> None:
    """Create or replace the agent's UC functions and check one call.

    Args:
        spark: An active Spark session.
        cfg: The project configuration for the target environment.

    """
    spark.sql(transcript_function_sql(cfg))
    name = f"{cfg.full_schema_name}.{TRANSCRIPT_FUNCTION}"
    rows = spark.sql(f"SELECT * FROM {name}('lecture-1', 0, 120)").collect()
    logger.info(f"Created {name}; lecture-1 from 0 to 120 s returns {len(rows)} turns")
