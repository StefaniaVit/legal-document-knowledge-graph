"""
Unit tests for src/graph/neo4j_loader.py's pure-function logic (entity
normalization, self-reference detection, canonical type resolution).
Deliberately excludes anything that touches an actual Neo4j connection.
"""
from src.graph.neo4j_loader import (
    _compute_canonical_types,
    _entity_key,
    _is_self_reference,
    _normalize,
)


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("ENISA") == "enisa"

    def test_collapses_internal_whitespace(self):
        assert _normalize("supervisory   authority\n") == "supervisory authority"

    def test_strips_leading_and_trailing_whitespace(self):
        assert _normalize("  Member States  ") == "member states"


class TestEntityKey:
    def test_matches_normalize(self):
        assert _entity_key("ENISA") == _normalize("ENISA")

    def test_same_entity_different_casing_produces_same_key(self):
        # this is the actual deduplication mechanism -- two mentions with
        # different surface casing must collapse to the same graph node
        assert _entity_key("Supervisory Authority") == _entity_key("supervisory authority")


class TestIsSelfReference:
    def test_matches_this_regulation(self):
        assert _is_self_reference("this Regulation")

    def test_matches_the_directive_case_insensitive(self):
        assert _is_self_reference("The Directive")

    def test_matches_this_act_and_this_decision(self):
        assert _is_self_reference("this Act")
        assert _is_self_reference("this Decision")

    def test_does_not_match_named_entity(self):
        assert not _is_self_reference("ENISA")
        assert not _is_self_reference("Regulation (EU) 2016/679")

    def test_known_gap_bare_word_is_not_caught(self):
        # Documented known limitation (see neo4j_loader.py module docstring
        # and CLAUDE.md): a bare "Regulation"/"Directive" with no leading
        # "this"/"the" is NOT caught by this filter, and still collides
        # across unrelated documents when loaded into the graph. This test
        # exists to make that gap explicit and fail loudly if the regex is
        # ever tightened in a way that changes this without updating the
        # docs -- not to assert this is desired behavior.
        assert not _is_self_reference("Regulation")
        assert not _is_self_reference("Directive")


class TestComputeCanonicalTypes:
    def test_majority_vote_picks_most_common_type(self):
        # Mirrors the real "ENISA" case: mostly ORGANIZATION, a couple of
        # mismatched mentions elsewhere -- majority should win.
        results = [
            {"entities": [{"text": "ENISA", "type": "ORGANIZATION"}]},
            {"entities": [{"text": "ENISA", "type": "ORGANIZATION"}]},
            {"entities": [{"text": "ENISA", "type": "ORGANIZATION"}]},
            {"entities": [{"text": "ENISA", "type": "LEGAL_ACT"}]},
            {"entities": [{"text": "enisa", "type": "LEGAL_CONCEPT"}]},  # different casing, same key
        ]
        canonical = _compute_canonical_types(results)
        assert canonical["enisa"] == "ORGANIZATION"

    def test_excludes_self_referential_entities(self):
        results = [{"entities": [{"text": "this Regulation", "type": "LEGAL_ACT"}]}]
        canonical = _compute_canonical_types(results)
        assert "this regulation" not in canonical

    def test_excludes_entities_with_invalid_type(self):
        # A type outside the known whitelist (e.g. malformed extraction
        # output) must not silently get counted as if valid.
        results = [{"entities": [{"text": "Some Entity", "type": "NOT_A_REAL_TYPE"}]}]
        canonical = _compute_canonical_types(results)
        assert "some entity" not in canonical

    def test_empty_input_produces_empty_mapping(self):
        assert _compute_canonical_types([]) == {}

    def test_relations_key_is_ignored_if_absent(self):
        # compute_canonical_types only reads "entities" -- must not require
        # a "relations" key to be present
        results = [{"entities": [{"text": "Commission", "type": "ORGANIZATION"}]}]
        assert _compute_canonical_types(results) == {"commission": "ORGANIZATION"}
