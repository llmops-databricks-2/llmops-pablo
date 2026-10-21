"""Tests for the lecture parsers, on synthetic Teams exports shaped like the real ones.

No real lecture data appears here: every name and line of text is made up.
"""

import zipfile
from datetime import UTC, date, datetime
from io import BytesIO

from arxiv_curator.lecture_parsing import parse_chat, parse_offset, parse_transcript, resolve_chat_date


def make_docx(paragraphs: list[list[str]]) -> bytes:
    """Build a minimal .docx whose paragraphs are runs separated by line breaks.

    Args:
        paragraphs: One list of lines per paragraph; lines are joined with `<w:br/>`.

    Returns:
        The .docx file content.

    """
    body = "".join(
        "<w:p>" + "<w:r><w:br/></w:r>".join(f"<w:r><w:t>{line}</w:t></w:r>" for line in lines) + "</w:p>"
        for lines in paragraphs
    )
    xml = f'<w:document xmlns:w="w"><w:body>{body}</w:body></w:document>'
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as docx:
        docx.writestr("word/document.xml", xml)
    return buffer.getvalue()


TRANSCRIPT = make_docx(
    [
        ["Sample Course Meeting-20261012_160034-Meeting Recording"],
        ["October 12, 2026, 2:00PM"],
        ["1h 2m 3s"],
        ["Ada Example started transcription"],
        ["Ada Example   0:19", "Welcome everyone."],
        ["Today we look at Delta &amp; Unity Catalog."],
        ["Bo Sample   12:05", "Can you repeat the last part?"],
        ["Ada Example   1:01:30", "Sure."],
    ]
)

CHAT = """Meeting chat preamble line

Can you share the… by Bo Sample
Bo Sample
Monday 16:09

Can you share the slides?

Thanks by Cy Test
Cy Test
Monday 16:15

Thanks
👍
2 Like reactions.

Here is the link… by Ada Example
Monday 16:20
Ada Example

Here is the link to the deck

Short preview of a lost message… by Example, A. (Org)
12/10/2026 16:30
"""


def test_parse_offset_handles_minutes_and_hours() -> None:
    """Convert both timestamp forms Teams uses to seconds."""
    assert parse_offset("0:19") == 19
    assert parse_offset("12:05") == 725
    assert parse_offset("1:01:30") == 3690


def test_parse_transcript_splits_turns_and_merges_continuations() -> None:
    """Split turns on speaker headers and attach header-less paragraphs to the turn above."""
    header, turns = parse_transcript(TRANSCRIPT, "kick-off")

    assert header.recording_start_utc == datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    assert header.duration_seconds == 3723
    assert [t["speaker"] for t in turns] == ["Ada Example", "Bo Sample", "Ada Example"]
    assert turns[0]["text"] == "Welcome everyone.\nToday we look at Delta & Unity Catalog."
    assert turns[2]["offset_seconds"] == 3690
    assert turns[2]["ts_utc"] == datetime(2026, 10, 12, 15, 1, 30, tzinfo=UTC)


def test_resolve_chat_date_uses_dates_and_weekdays_after_the_lecture() -> None:
    """Read full dates as given and place weekdays on or after the lecture date."""
    lecture = date(2026, 10, 14)  # a Wednesday
    assert resolve_chat_date("12/10/2026", lecture) == date(2026, 10, 12)
    assert resolve_chat_date("Wednesday", lecture) == lecture
    assert resolve_chat_date("Friday", lecture) == date(2026, 10, 16)


def test_parse_chat_groups_messages_and_drops_reactions() -> None:
    """Group lines into messages, skip summaries and reactions, and handle both header orders."""
    messages = parse_chat(CHAT, "kick-off", date(2026, 10, 12), datetime(2026, 10, 12, 14, 0, tzinfo=UTC))

    assert [m["sender"] for m in messages] == ["Bo Sample", "Cy Test", "Ada Example", "Example, A. (Org)"]
    assert messages[0]["text"] == "Can you share the slides?"
    assert messages[1]["text"] == "Thanks"
    assert messages[2]["text"] == "Here is the link to the deck"
    assert not any(m["text_truncated"] for m in messages[:3])


def test_parse_chat_falls_back_to_the_summary_preview() -> None:
    """Use a cut-off summary as the text when the paste kept no message body."""
    last = parse_chat(CHAT, "kick-off", date(2026, 10, 12), datetime(2026, 10, 12, 14, 0, tzinfo=UTC))[-1]

    assert last["text"] == "Short preview of a lost message"
    assert last["text_truncated"]


def test_parse_chat_converts_amsterdam_time_across_the_dst_change() -> None:
    """Convert local chat times to UTC with the offset valid on each date."""
    start = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    summer = parse_chat("Bo Sample\n12/10/2026 16:09\n\nHi", "x", date(2026, 10, 12), start)[0]
    winter = parse_chat("Bo Sample\n26/10/2026 16:09\n\nHi", "x", date(2026, 10, 26), start)[0]

    assert summer["ts_utc"] == datetime(2026, 10, 12, 14, 9, tzinfo=UTC)
    assert summer["offset_seconds"] == 540
    assert winter["ts_utc"] == datetime(2026, 10, 26, 15, 9, tzinfo=UTC)
