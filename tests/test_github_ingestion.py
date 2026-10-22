"""Tests for the GitHub event reading in arxiv_curator.utils.github_ingestion, on synthetic events."""

import json
from datetime import UTC, datetime

from arxiv_curator.utils.github_ingestion import PAGE_SIZE, event_row, fetch_events, get_pages


class FakeResponse:
    """A canned GitHub API response."""

    def __init__(self, body: list[dict]) -> None:
        """Store the JSON body.

        Args:
            body: The list the endpoint returns.

        """
        self.body = body

    def raise_for_status(self) -> None:
        """Accept every response as successful."""

    def json(self) -> list[dict]:
        """Return the JSON body.

        Returns:
            The list the endpoint returns.

        """
        return self.body


class FakeSession:
    """A session that serves pages from a map of URL to pages, and records each call."""

    def __init__(self, pages: dict[str, list[list[dict]]]) -> None:
        """Store the pages each URL returns.

        Args:
            pages: For each URL, its pages in order.

        """
        self.pages = pages
        self.calls: list[tuple[str, int]] = []

    def get(self, url: str, params: dict, timeout: int) -> FakeResponse:
        """Return the requested page, or an empty page beyond the last.

        Args:
            url: The endpoint URL.
            params: The paging parameters.
            timeout: The request timeout, unused.

        Returns:
            The page.

        """
        self.calls.append((url, params["page"]))
        url_pages = self.pages.get(url, [])
        return FakeResponse(url_pages[params["page"] - 1] if params["page"] <= len(url_pages) else [])


def make_event(event_id: str, event_type: str = "PushEvent", repo: str = "org/repo-a") -> dict:
    """Build a minimal synthetic event in the events API's shape.

    Args:
        event_id: The event id.
        event_type: The event type.
        repo: The repository's full name.

    Returns:
        The event.

    """
    return {
        "id": event_id,
        "type": event_type,
        "repo": {"name": repo},
        "actor": {"login": "ada-example"},
        "created_at": "2026-10-21T20:32:40Z",
        "payload": {"ref": "refs/heads/main", "head": "abc123"},
    }


def test_event_row_flattens_an_event_and_keeps_the_payload() -> None:
    """Keep the id, type, repository and actor, parse the time as UTC, and keep the payload as JSON."""
    row = event_row(make_event("1", "PullRequestEvent"))

    assert row["id"] == "1"
    assert row["type"] == "PullRequestEvent"
    assert row["repo"] == "org/repo-a"
    assert row["actor"] == "ada-example"
    assert row["created_at"] == datetime(2026, 10, 21, 20, 32, 40, tzinfo=UTC)
    assert json.loads(row["payload"]) == {"head": "abc123", "ref": "refs/heads/main"}


def test_get_pages_stops_at_the_first_short_page() -> None:
    """Read the next page only while pages are full."""
    full = [make_event(str(i)) for i in range(PAGE_SIZE)]
    session = FakeSession({"u": [full, [make_event("x")], [make_event("never read")]]})

    items = get_pages(session, "u", max_pages=3)

    assert len(items) == PAGE_SIZE + 1
    assert session.calls == [("u", 1), ("u", 2)]


def test_fetch_events_reads_every_repo_and_drops_repeated_events() -> None:
    """Read events of each public repository, and keep an event that repeats across pages once."""
    api = "https://api.github.com"
    session = FakeSession(
        {
            f"{api}/orgs/org/repos?type=public": [[{"full_name": "org/repo-b"}, {"full_name": "org/repo-a"}]],
            f"{api}/repos/org/repo-a/events": [[make_event("1"), make_event("2"), make_event("1")]],
            f"{api}/repos/org/repo-b/events": [[make_event("3", repo="org/repo-b")]],
        }
    )

    rows = fetch_events(session, "org")

    assert sorted(r["id"] for r in rows) == ["1", "2", "3"]
