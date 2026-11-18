"""Read the course organization's GitHub events and merge them into the `github_events` table.

Shared by the job's `scripts/github_ingestion.py`. The repository events API returns at most 300
events per repository, from the last 90 days, so each run reads everything still available and the
`MERGE` on the event `id` keeps only the events not stored yet.
"""

import json
from datetime import UTC, datetime

import requests
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from arxiv_curator.config import ProjectConfig
from arxiv_curator.schemas.github_tables import GITHUB_EVENTS

API = "https://api.github.com"
PAGE_SIZE = 100
EVENT_PAGES = 3


def github_session(token: str) -> requests.Session:
    """Build an HTTP session that authenticates to the GitHub API.

    Args:
        token: A GitHub token with read access to the repositories.

    Returns:
        The session, with the token and API version headers set.

    """
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def get_pages(session: requests.Session, url: str, max_pages: int) -> list[dict]:
    """Read a paginated GitHub list endpoint until a short page or the page limit.

    Args:
        session: The authenticated session.
        url: The endpoint URL, without paging parameters.
        max_pages: The most pages to read.

    Returns:
        The items of all pages read, in order.

    Raises:
        requests.HTTPError: When GitHub answers with an error status, such as 401 for an expired
            token.

    """
    items: list[dict] = []
    for page in range(1, max_pages + 1):
        response = session.get(url, params={"per_page": PAGE_SIZE, "page": page}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        items += batch
        if len(batch) < PAGE_SIZE:
            break
    return items


def list_repos(session: requests.Session, org: str) -> list[str]:
    """List the organization's public repositories.

    Args:
        session: The authenticated session.
        org: The GitHub organization.

    Returns:
        The repositories' full names, such as `llmops-databricks-2/llmops-pablo`.

    """
    repos = get_pages(session, f"{API}/orgs/{org}/repos?type=public", max_pages=10)
    return sorted(r["full_name"] for r in repos)


def event_row(event: dict) -> dict:
    """Flatten one GitHub event into a `github_events` row.

    Args:
        event: The event as the events API returns it.

    Returns:
        The row, with the payload kept as a JSON string so nothing is lost.

    """
    return {
        "id": event["id"],
        "type": event["type"],
        "repo": event["repo"]["name"],
        "actor": (event.get("actor") or {}).get("login"),
        "created_at": datetime.fromisoformat(event["created_at"].replace("Z", "+00:00")),
        "payload": json.dumps(event.get("payload"), sort_keys=True),
    }


def fetch_events(session: requests.Session, org: str) -> list[dict]:
    """Read the events still available for every public repository of the organization.

    Args:
        session: The authenticated session.
        org: The GitHub organization.

    Returns:
        One row per event, without duplicates: an event can repeat across pages when new events
        arrive during the read.

    """
    rows: dict[str, dict] = {}
    repos = list_repos(session, org)
    for repo in repos:
        for event in get_pages(session, f"{API}/repos/{repo}/events", max_pages=EVENT_PAGES):
            rows[event["id"]] = event_row(event)
    logger.info(f"Read {len(rows)} events from {len(repos)} repositories of {org}")
    return list(rows.values())


def ingest_github_events(spark: SparkSession, cfg: ProjectConfig, token: str) -> int:
    """Merge the organization's new GitHub events into the `github_events` table.

    Args:
        spark: The Spark session.
        cfg: The project configuration for the target environment.
        token: A GitHub token with read access to the public repositories.

    Returns:
        The number of events added.

    """
    table = f"{cfg.full_schema_name}.github_events"
    rows = fetch_events(github_session(token), cfg.github_org)
    df = spark.createDataFrame(rows, GITHUB_EVENTS).withColumn("ingested_at", F.lit(datetime.now(UTC)))

    df.limit(0).write.format("delta").mode("ignore").saveAsTable(table)
    df.createOrReplaceTempView("new_github_events")
    result = spark.sql(f"""
        MERGE INTO {table} AS t
        USING new_github_events AS s
        ON t.id = s.id
        WHEN NOT MATCHED THEN INSERT *
    """).collect()[0]
    added = int(result["num_inserted_rows"])
    logger.info(f"Added {added} new events to {table}")
    return added
