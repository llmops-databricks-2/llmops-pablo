"""Tests for the lecture search's filters and vector formatting, without a database."""

import pytest

from arxiv_curator.utils.lecture_search import filter_clause, filtered_search, vector_literal


def test_filter_clause_is_true_without_filters() -> None:
    """No filter matches every row and adds no parameters."""
    assert filter_clause() == ("TRUE", [])


def test_filter_clause_combines_filters_with_parameters() -> None:
    """Each filter becomes a placeholder condition, joined with AND, and its value a parameter."""
    assert filter_clause(lecture_id="lecture-2") == ("lecture_id = %s", ["lecture-2"])
    assert filter_clause(source="slides") == ("source = %s", ["slides"])
    assert filter_clause("lecture-1", "transcript") == ("lecture_id = %s AND source = %s", ["lecture-1", "transcript"])


def test_vector_literal_uses_pgvector_syntax() -> None:
    """A vector is written as bracketed comma-separated floats."""
    assert vector_literal([1, 0.5, -2]) == "[1.0,0.5,-2.0]"


def test_filtered_search_requires_a_filter() -> None:
    """Filtered search without a filter is refused before any query runs."""
    with pytest.raises(ValueError, match="needs lecture_id, source or both"):
        filtered_search(None, None, [0.0])  # type: ignore[arg-type]
