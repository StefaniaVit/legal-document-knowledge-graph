"""
Unit tests for src/rag/graph_rag.py's pure-function logic. Deliberately
excludes anything that calls Gemini or Neo4j directly.
"""
from src.rag.graph_rag import _filter_known_names, _normalize_for_match, _truncate_at_sentence


class TestNormalizeForMatch:
    def test_lowercases_and_collapses_whitespace(self):
        assert _normalize_for_match("  ENISA  ") == "enisa"


class TestFilterKnownNames:
    def test_keeps_exact_match(self):
        known = ["ENISA", "Commission"]
        assert _filter_known_names(["ENISA"], known) == ["ENISA"]

    def test_drops_unknown_candidate(self):
        # The hard backstop this function exists for: even if the model
        # invents a name not in the real list, it must not pass through.
        known = ["ENISA"]
        assert _filter_known_names(["Some Made Up Entity"], known) == []

    def test_keeps_candidate_despite_case_mismatch(self):
        # The model echoes back a name it was shown, but isn't guaranteed
        # to preserve exact casing -- a naive exact-match filter would
        # silently drop this valid selection.
        known = ["ENISA", "Commission"]
        assert _filter_known_names(["enisa", "COMMISSION"], known) == ["ENISA", "Commission"]

    def test_returns_the_known_names_own_casing_not_the_candidates(self):
        # Downstream Cypher lookups match by exact stored name, so the
        # *known* casing must be returned, not whatever the candidate used.
        known = ["Supervisory Authority"]
        result = _filter_known_names(["supervisory authority"], known)
        assert result == ["Supervisory Authority"]

    def test_deduplicates_candidates_matching_the_same_known_name(self):
        known = ["ENISA"]
        result = _filter_known_names(["ENISA", "enisa", "  Enisa  "], known)
        assert result == ["ENISA"]

    def test_empty_candidates_returns_empty(self):
        assert _filter_known_names([], ["ENISA"]) == []


class TestTruncateAtSentence:
    def test_returns_unchanged_text_under_limit(self):
        assert _truncate_at_sentence("Short text.", 600) == "Short text."

    def test_cuts_at_the_last_sentence_boundary_before_the_limit(self):
        # Regression test: a blind text[:limit] slice used to cut mid-word/
        # mid-clause on longer chunks (e.g. "...in accordance with Regulation
        # (EU"), silently dropping whatever came after. This must back up to
        # the nearest preceding ". " or "; " instead.
        text = "First sentence is here. Second sentence runs long past the cutoff point."
        result = _truncate_at_sentence(text, 30)
        assert result == "First sentence is here."
        assert text.startswith(result)

    def test_falls_back_to_ellipsis_when_no_sentence_boundary_found(self):
        text = "a" * 700
        result = _truncate_at_sentence(text, 600)
        assert result == "a" * 600 + "..."

    def test_does_not_truncate_when_text_is_exactly_the_limit(self):
        text = "a" * 600
        assert _truncate_at_sentence(text, 600) == text
