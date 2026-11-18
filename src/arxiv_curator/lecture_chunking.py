"""Split parsed lecture content into chunks for embedding and search.

Pure Python with no Spark dependency, like `lecture_parsing`, so the rules can be unit-tested
locally. Transcript turns are packed into windows of up to `MAX_WORDS` words, split into word pieces or,
as an alternative, into sentences; slide and materials pages become one chunk each. Speaker names stay out of the chunk text: search returns what was
said, not who said it (see Privacy and compliance in `proposal/README.md`).
"""

import re
from collections.abc import Callable
from itertools import pairwise

MAX_WORDS = 300
# A turn is repeated at the start of the next window only when it is at most this long, so a long
# turn is not stored twice.
MAX_OVERLAP_WORDS = MAX_WORDS // 2


def clean_text(text: str | None) -> str:
    """Normalize whitespace and rejoin words hyphenated across line breaks.

    The same cleaning as the course reference's `DataProcessor._clean_chunk`.

    Args:
        text: Raw text from a transcript turn or a parsed PDF page.

    Returns:
        The cleaned text; an empty string for missing text.

    """
    if not text:
        return ""
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_words(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split text into consecutive pieces of at most `max_words` words.

    Args:
        text: Cleaned text.
        max_words: The largest number of words in a piece.

    Returns:
        The pieces, in order; an empty list for empty text.

    """
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def split_sentences(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split text into sentences, cutting only a sentence longer than `max_words` words.

    A sentence ends at `.`, `!` or `?` followed by whitespace. The rule is the course reference's
    sentence-based chunking; it also breaks after abbreviations such as "e.g.", which is accepted.

    Args:
        text: Cleaned text.
        max_words: The largest number of words in a piece.

    Returns:
        The pieces, in order; an empty list for empty text.

    """
    return [piece for sentence in re.split(r"(?<=[.!?])\s+", text) for piece in split_words(sentence, max_words)]


def chunk_transcript(
    lecture_id: str,
    turns: list[dict],
    max_words: int = MAX_WORDS,
    split: Callable[[str, int], list[str]] = split_words,
) -> list[dict]:
    """Pack one lecture's transcript turns into windows of up to `max_words` words.

    Turns are taken in order and cut into pieces by `split`: by default a turn stays whole unless it is
    longer than `max_words`; with `split_sentences` every sentence is a piece. When a window is full, the
    next one starts with the window's last piece if that piece has at most `MAX_OVERLAP_WORDS` words.
    Pieces of one turn are joined with a space, and turns with a newline. Each chunk spans from its first
    piece's turn offset to the offset of the turn after its last piece.

    Args:
        lecture_id: The lecture the turns belong to.
        turns: Rows of `lecture_transcript` with `turn`, `offset_seconds` and `text`.
        max_words: The largest number of words in a chunk.
        split: Cuts a turn's cleaned text into pieces of at most `max_words` words.

    Returns:
        Chunk rows without `source_path` and `ingested_at`.

    """
    ordered = sorted(turns, key=lambda t: t["turn"])
    pieces = []
    for i, turn in enumerate(ordered):
        end = ordered[i + 1]["offset_seconds"] if i + 1 < len(ordered) else turn["offset_seconds"]
        for text in split(clean_text(turn["text"]), max_words):
            pieces.append(
                {"turn": i, "text": text, "words": len(text.split()), "start": turn["offset_seconds"], "end": end}
            )

    windows: list[list[dict]] = []
    window: list[dict] = []
    for piece in pieces:
        if window and sum(p["words"] for p in window) + piece["words"] > max_words:
            windows.append(window)
            last = window[-1]
            window = (
                [last] if last["words"] + piece["words"] <= max_words and last["words"] <= MAX_OVERLAP_WORDS else []
            )
        window.append(piece)
    if window:
        windows.append(window)

    chunks = []
    for seq, window in enumerate(windows):
        text = window[0]["text"]
        for prev, piece in pairwise(window):
            text += (" " if piece["turn"] == prev["turn"] else "\n") + piece["text"]
        chunks.append(
            {
                "chunk_id": f"{lecture_id}:transcript:{seq:04d}",
                "lecture_id": lecture_id,
                "source": "transcript",
                "text": text,
                "word_count": len(text.split()),
                "page": None,
                "start_offset_seconds": window[0]["start"],
                "end_offset_seconds": window[-1]["end"],
            }
        )
    return chunks


def chunk_page(lecture_id: str, kind: str, page: int, text: str | None) -> dict | None:
    """Turn one parsed PDF page into a chunk.

    Args:
        lecture_id: The lecture the page belongs to.
        kind: The PDF's name, such as `slides` or `materials`.
        page: The 1-based page number.
        text: The page text from `lecture_slides`.

    Returns:
        The chunk row without `source_path` and `ingested_at`, or None for a page with no text.

    """
    cleaned = clean_text(text)
    if not cleaned:
        return None
    return {
        "chunk_id": f"{lecture_id}:{kind}:p{page:03d}",
        "lecture_id": lecture_id,
        "source": kind,
        "text": cleaned,
        "word_count": len(cleaned.split()),
        "page": page,
        "start_offset_seconds": None,
        "end_offset_seconds": None,
    }
