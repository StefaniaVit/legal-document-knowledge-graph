"""
Unit tests for src/extraction/entity_relation_extractor.py's pure-function
logic. Deliberately excludes extract_entities()/extract_relations()
themselves, since those call the local LLM.
"""
from src.extraction.entity_relation_extractor import _filter_valid_relations, _normalize_for_match


class TestNormalizeForMatch:
    def test_lowercases_and_collapses_whitespace(self):
        assert _normalize_for_match("  ENISA  ") == "enisa"
        assert _normalize_for_match("supervisory   authority") == "supervisory authority"


class TestFilterValidRelations:
    def test_keeps_relation_with_exact_match(self):
        entities = [{"text": "ENISA", "type": "ORGANIZATION"}, {"text": "Commission", "type": "ORGANIZATION"}]
        relations = [{"subject": "ENISA", "relation": "APPLIES_TO", "object": "Commission"}]
        assert _filter_valid_relations(relations, entities) == relations

    def test_drops_relation_referencing_unlisted_entity(self):
        # This is the hard backstop this function exists for: even if the
        # model violates its own prompt instruction and names something
        # outside the confirmed entity list, it must not pass through.
        entities = [{"text": "ENISA", "type": "ORGANIZATION"}]
        relations = [{"subject": "ENISA", "relation": "APPLIES_TO", "object": "Some Made Up Thing"}]
        assert _filter_valid_relations(relations, entities) == []

    def test_keeps_relation_despite_case_mismatch(self):
        # Entities and relations come from two separate LLM calls -- the
        # model isn't guaranteed to echo a name back with identical casing.
        # A naive exact-match filter would silently drop this valid relation.
        entities = [{"text": "ENISA", "type": "ORGANIZATION"}, {"text": "Commission", "type": "ORGANIZATION"}]
        relations = [{"subject": "enisa", "relation": "APPLIES_TO", "object": "COMMISSION"}]
        assert _filter_valid_relations(relations, entities) == relations

    def test_keeps_relation_despite_whitespace_mismatch(self):
        entities = [{"text": "supervisory authority", "type": "ORGANIZATION"},
                    {"text": "administrative fines", "type": "PENALTY"}]
        relations = [{"subject": "supervisory  authority", "relation": "IMPOSES_OBLIGATION_ON",
                      "object": " administrative fines "}]
        assert _filter_valid_relations(relations, entities) == relations

    def test_empty_relations_returns_empty(self):
        entities = [{"text": "ENISA", "type": "ORGANIZATION"}]
        assert _filter_valid_relations([], entities) == []
