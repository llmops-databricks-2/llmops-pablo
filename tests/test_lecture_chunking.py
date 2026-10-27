"""Tests for the lecture chunking rules, on made-up turns and pages."""

from arxiv_curator.lecture_chunking import (
    MAX_OVERLAP_WORDS,
    chunk_page,
    chunk_transcript,
    clean_text,
    split_sentences,
    split_words,
)


def turn(n: int, offset: int, words: int, word: str = "w") -> dict:
    """Build a transcript row of `words` copies of `word`.

    Args:
        n: The turn number.
        offset: The turn's offset in seconds.
        words: The number of words in the turn.
        word: The word to repeat.

    Returns:
        A row shaped like `lecture_transcript`.

    """
    return {"turn": n, "offset_seconds": offset, "speaker": "Ada Example", "text": " ".join([word] * words)}


def test_clean_text_rejoins_hyphenation_and_collapses_whitespace() -> None:
    """Hyphenated line breaks are rejoined and whitespace runs become one space."""
    assert clean_text("docu-\nments  are\n\n  here ") == "documents are here"
    assert clean_text(None) == ""


def test_split_words_caps_each_piece() -> None:
    """A long text is cut into pieces of at most the given number of words."""
    assert split_words("a b c d e", 2) == ["a b", "c d", "e"]
    assert split_words("", 2) == []


def test_short_turns_pack_into_one_chunk() -> None:
    """Turns that fit together become one chunk spanning their offsets."""
    chunks = chunk_transcript("l1", [turn(2, 30, 10), turn(1, 0, 10), turn(3, 60, 10)], max_words=300)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "l1:transcript:0000"
    assert chunks[0]["word_count"] == 30
    assert (chunks[0]["start_offset_seconds"], chunks[0]["end_offset_seconds"]) == (0, 60)


def test_full_window_carries_a_short_last_turn() -> None:
    """The next window starts with the previous window's last turn when that turn is short."""
    turns = [turn(1, 0, 200, "a"), turn(2, 10, 50, "b"), turn(3, 20, 100, "c")]
    chunks = chunk_transcript("l1", turns, max_words=300)
    assert [c["word_count"] for c in chunks] == [250, 150]
    assert chunks[1]["text"].startswith("b")
    assert chunks[1]["start_offset_seconds"] == 10


def test_long_turn_is_split_and_not_repeated() -> None:
    """A turn longer than the window is split, and its long pieces are not carried over."""
    chunks = chunk_transcript("l1", [turn(1, 0, 700), turn(2, 90, 5)], max_words=300)
    assert [c["word_count"] for c in chunks] == [300, 300, 105]
    assert all(c["word_count"] <= 300 for c in chunks)
    assert chunks[-1]["end_offset_seconds"] == 90
    assert MAX_OVERLAP_WORDS == 150


def test_empty_turns_are_dropped() -> None:
    """Turns with no text add no words and no chunks."""
    assert chunk_transcript("l1", [{"turn": 1, "offset_seconds": 0, "text": None}]) == []


def test_chunk_page_keeps_the_page_and_skips_empty_pages() -> None:
    """A page becomes one cleaned chunk; a page with no text becomes none."""
    chunk = chunk_page("l1", "slides", 3, "Delta\nLake")
    assert chunk["chunk_id"] == "l1:slides:p003"
    assert (chunk["source"], chunk["page"], chunk["text"]) == ("slides", 3, "Delta Lake")
    assert chunk_page("l1", "slides", 4, "  \n ") is None


def test_split_sentences_cuts_at_sentence_ends_only() -> None:
    """Sentences end at a full stop, question or exclamation mark, and only an overlong one is cut."""
    assert split_sentences("One two. Three? Four five!") == ["One two.", "Three?", "Four five!"]
    assert split_sentences("a b c d e. f", 2) == ["a b", "c d", "e.", "f"]
    assert split_sentences("") == []


def test_sentence_windows_end_on_sentence_boundaries() -> None:
    """With sentence pieces, a window closes before a sentence that would not fit, not inside it."""
    sentence = " ".join(["w"] * 9) + "."
    turns = [{"turn": 1, "offset_seconds": 0, "text": " ".join([sentence] * 7)}]
    chunks = chunk_transcript("l1", turns, max_words=25, split=split_sentences)
    assert [c["word_count"] for c in chunks] == [18, 18, 18, 18, 18, 18]
    assert all(c["text"].endswith(".") for c in chunks)
    assert "\n" not in chunks[0]["text"]


def test_sentence_windows_carry_the_last_sentence() -> None:
    """The next window starts with the previous window's last sentence, and turns stay on their own lines."""
    turns = [
        {"turn": 1, "offset_seconds": 0, "text": "Alpha one two. Beta three four."},
        {"turn": 2, "offset_seconds": 30, "text": "Gamma five six."},
    ]
    chunks = chunk_transcript("l1", turns, max_words=7, split=split_sentences)
    assert [c["text"] for c in chunks] == ["Alpha one two. Beta three four.", "Beta three four.\nGamma five six."]
    assert (chunks[1]["start_offset_seconds"], chunks[1]["end_offset_seconds"]) == (0, 30)


def test_chunk_ids_are_stable_across_reruns() -> None:
    """The same turns give the same chunk ids, whatever order they arrive in, so the sync key holds."""
    turns = [turn(n, n * 10, 40 + n * 7) for n in range(1, 30)]
    first = [c["chunk_id"] for c in chunk_transcript("l1", turns, split=split_sentences)]
    again = [c["chunk_id"] for c in chunk_transcript("l1", list(reversed(turns)), split=split_sentences)]
    assert first == again
    assert len(set(first)) == len(first)
