"""Parse raw lecture files (Teams transcripts and copy-pasted Teams chats) into table rows.

The parsers are pure Python with no Spark dependency, so they can be unit-tested locally and
reused by the data preparation notebook and job. They only handle the formats observed in the
course's Teams exports; lines they cannot place are counted and logged, never silently kept.
"""

import html
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

from loguru import logger

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_TURN_HEADER = re.compile(r"^(?P<speaker>.+?)\s+(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)$")
_TRANSCRIPTION_EVENT = re.compile(r"(started|stopped) transcription$")
_DURATION = re.compile(r"^(?:(?P<h>\d+)h)?\s*(?:(?P<m>\d+)m)?\s*(?:(?P<s>\d+)s)?$")
_CHAT_TIME = re.compile(rf"^(?P<day>{'|'.join(WEEKDAYS)}|\d{{2}}/\d{{2}}/\d{{4}})\s+(?P<hm>\d{{1,2}}:\d{{2}})$")
_ELLIPSIS = ("...", "…")
_REACTION = re.compile(r"^(\d+\s+\w+\s+reactions?\.?|[^\w\s]+)$")
_PASTE_ARTIFACT = re.compile(r"<\d+></\d+>")


@dataclass
class TranscriptHeader:
    """Metadata from the first lines of a Teams transcript.

    Attributes:
        recording_start_utc: When the recording started, in UTC.
        duration_seconds: Length of the recording.

    """

    recording_start_utc: datetime
    duration_seconds: int


def parse_offset(ts: str) -> int:
    """Convert a transcript timestamp to seconds since the recording started.

    Args:
        ts: A timestamp in `M:SS` or `H:MM:SS` form.

    Returns:
        The offset in seconds.

    """
    seconds = 0
    for part in ts.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def docx_paragraphs(docx_bytes: bytes) -> list[str]:
    """Extract paragraph texts from a .docx file, keeping line breaks inside a paragraph.

    Args:
        docx_bytes: The raw .docx file content.

    Returns:
        The non-empty paragraphs, stripped, with `<w:br/>` turned into newlines.

    """
    xml = zipfile.ZipFile(BytesIO(docx_bytes)).read("word/document.xml").decode("utf-8")
    paragraphs = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        runs = re.finditer(r"<w:br\s*/>|<w:t(?:\s[^>]*)?>(.*?)</w:t>", para, flags=re.S)
        text = "".join("\n" if m[1] is None else html.unescape(m[1]) for m in runs).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def parse_transcript_header(paragraphs: list[str]) -> TranscriptHeader:
    """Read the recording start and duration from a transcript's header paragraphs.

    Args:
        paragraphs: The transcript paragraphs, as returned by `docx_paragraphs`.

    Returns:
        The parsed header.

    Raises:
        ValueError: If the start time or duration is not where Teams puts it.

    """
    start = datetime.strptime(paragraphs[1].replace(" ", ""), "%B%d,%Y,%I:%M%p").replace(tzinfo=UTC)
    match = _DURATION.match(paragraphs[2].strip())
    if not match or not any(match.groupdict().values()):
        raise ValueError(f"Unrecognised transcript duration line: {paragraphs[2]!r}")
    duration = sum(int(match[k] or 0) * f for k, f in (("h", 3600), ("m", 60), ("s", 1)))
    return TranscriptHeader(recording_start_utc=start, duration_seconds=duration)


def parse_transcript(docx_bytes: bytes, lecture_id: str) -> tuple[TranscriptHeader, list[dict]]:
    """Split a Teams transcript into speaker turns with timestamps.

    A turn starts with a paragraph whose first line is `Speaker   M:SS`; paragraphs without
    such a header continue the previous turn.

    Args:
        docx_bytes: The raw .docx transcript.
        lecture_id: The lecture this transcript belongs to.

    Returns:
        The transcript header and one dict per turn with `lecture_id`, `turn`, `speaker`,
        `offset_seconds`, `ts_utc` and `text`.

    """
    paragraphs = docx_paragraphs(docx_bytes)
    header = parse_transcript_header(paragraphs)
    turns: list[dict] = []
    skipped = 0
    for para in paragraphs[3:]:
        first, _, rest = para.partition("\n")
        match = _TURN_HEADER.match(first.strip())
        if match:
            offset = parse_offset(match["ts"])
            turns.append(
                {
                    "lecture_id": lecture_id,
                    "turn": len(turns) + 1,
                    "speaker": match["speaker"].strip(),
                    "offset_seconds": offset,
                    "ts_utc": header.recording_start_utc + timedelta(seconds=offset),
                    "text": rest.strip(),
                }
            )
        elif _TRANSCRIPTION_EVENT.search(para) or not turns:
            skipped += 1
        else:
            turns[-1]["text"] = f"{turns[-1]['text']}\n{para}".strip()
    logger.info(f"{lecture_id}: {len(turns)} transcript turns, {skipped} non-turn paragraphs skipped")
    return header, turns


def resolve_chat_date(day: str, lecture_date: date) -> date:
    """Turn a chat time-line's day part into a calendar date.

    Teams shows a full date for older messages and a weekday for recent ones. A weekday is
    taken to be the lecture date or one of the six days after it.

    Args:
        day: `DD/MM/YYYY` or a weekday name.
        lecture_date: The local date of the lecture.

    Returns:
        The message's local date.

    """
    if day[0].isdigit():
        return datetime.strptime(day, "%d/%m/%Y").date()
    return lecture_date + timedelta(days=(WEEKDAYS.index(day) - lecture_date.weekday()) % 7)


def _is_noise(line: str) -> bool:
    """Tell whether a chat line is a reaction or other non-message noise.

    Args:
        line: A stripped chat line.

    Returns:
        True for reaction counts and emoji- or punctuation-only lines.

    """
    return bool(_REACTION.match(line))


@dataclass
class ChatSummary:
    """A Teams copy summary line: the first words of a message, then `by <Sender>`.

    Attributes:
        preview: The message's first words, without the ellipsis.
        cut: Whether the preview was cut off with an ellipsis.

    """

    preview: str
    cut: bool


def _summary_for(line: str, sender: str) -> ChatSummary | None:
    """Read a line as a given sender's summary line.

    Args:
        line: A stripped chat line.
        sender: The sender name the summary must end with.

    Returns:
        The summary, or None if the line is not that sender's summary.

    """
    suffix = f"by {sender}"
    if not sender or not line.endswith(suffix) or line == sender:
        return None
    preview = line[: -len(suffix)].rstrip()
    cut = preview.endswith(_ELLIPSIS)
    for mark in _ELLIPSIS:
        preview = preview.removesuffix(mark)
    return ChatSummary(preview=preview.strip(), cut=cut)


def _summary_sender(line: str) -> str:
    """Guess the sender of a summary line whose sender line is missing.

    Args:
        line: A stripped chat line.

    Returns:
        The text after the last ` by `, or an empty string if there is none.

    """
    head, sep, sender = line.rpartition(" by ")
    return sender.strip() if sep and head else ""


def parse_chat(
    text: str,
    lecture_id: str,
    lecture_date: date,
    recording_start_utc: datetime,
    tz: str = "Europe/Amsterdam",
) -> list[dict]:
    """Split a copy-pasted Teams meeting chat into messages.

    Each message has a header: a summary line (`<first words>… by <Sender>`), a sender line and
    a `<Weekday> HH:MM` or `DD/MM/YYYY HH:MM` time line; occasionally the sender comes after the
    time line. The message text follows until the next header. When the paste kept no text,
    the summary's preview is used and the message is flagged as truncated.

    Args:
        text: The pasted chat.
        lecture_id: The lecture this chat belongs to.
        lecture_date: The local date of the lecture, used to resolve weekday names.
        recording_start_utc: When the recording started, to compute offsets.
        tz: The time zone the chat times were shown in.

    Returns:
        One dict per message with `lecture_id`, `message`, `sender`, `ts_utc`,
        `offset_seconds`, `text_truncated` and `text`.

    """
    lines = [_PASTE_ARTIFACT.sub("", line).strip() for line in text.splitlines()]
    zone = ZoneInfo(tz)
    headers = []
    for i, line in enumerate(lines):
        match = _CHAT_TIME.match(line)
        if not match:
            continue
        before = lines[i - 1] if i > 0 else ""
        after = lines[i + 1] if i + 1 < len(lines) else ""
        if after and _summary_for(before, after):
            # Time line before the sender line.
            sender, start, body = after, i, i + 2
        elif not after and _summary_sender(before):
            # No sender line: only the summary names the sender.
            sender, start, body = _summary_sender(before), i, i + 1
        else:
            sender, start, body = before, i - 1, i + 1
        summary = _summary_for(lines[start - 1], sender) if start > 0 else None
        if summary:
            start -= 1
        headers.append((match, sender, start, body, summary))

    messages = []
    for k, (match, sender, _, body, summary) in enumerate(headers):
        end = headers[k + 1][2] if k + 1 < len(headers) else len(lines)
        body_text = "\n".join(line for line in lines[body:end] if line and not _is_noise(line))
        truncated = not body_text and bool(summary and summary.cut)
        if not body_text and summary:
            body_text = summary.preview
        hour, minute = map(int, match["hm"].split(":"))
        local = datetime.combine(resolve_chat_date(match["day"], lecture_date), datetime.min.time())
        ts_utc = local.replace(hour=hour, minute=minute, tzinfo=zone).astimezone(UTC)
        messages.append(
            {
                "lecture_id": lecture_id,
                "message": len(messages) + 1,
                "sender": sender,
                "ts_utc": ts_utc,
                "offset_seconds": int((ts_utc - recording_start_utc).total_seconds()),
                "text_truncated": truncated,
                "text": body_text,
            }
        )
    preamble = headers[0][2] if headers else len(lines)
    logger.info(f"{lecture_id}: {len(messages)} chat messages, {preamble} preamble lines skipped")
    return messages
