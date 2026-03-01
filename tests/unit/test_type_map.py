"""Tests for WIKIDATA_TYPE_MAP and grounding models.

Validates that:
- Every type map value exists in SEED_TYPES
- normalize_type() produces consistent lowercase
- PublicEntity and GroundingDecision dataclasses serialize correctly
"""

from smartmemory.grounding.type_map import WIKIDATA_TYPE_MAP, normalize_type, type_to_p31_qids
from smartmemory.graph.ontology_graph import SEED_TYPES
from smartmemory.grounding.models import PublicEntity, GroundingDecision


class TestWikidataTypeMap:
    """WIKIDATA_TYPE_MAP must bridge Wikidata P31 QIDs to valid SEED_TYPES."""

    def test_all_values_in_seed_types(self):
        """Every mapped type must exist in the canonical SEED_TYPES list."""
        for qid, entity_type in WIKIDATA_TYPE_MAP.items():
            assert entity_type in SEED_TYPES, (
                f"WIKIDATA_TYPE_MAP['{qid}'] = '{entity_type}' not in SEED_TYPES"
            )

    def test_map_is_not_empty(self):
        assert len(WIKIDATA_TYPE_MAP) > 0

    def test_all_keys_are_qid_format(self):
        """QIDs start with 'Q' followed by digits."""
        for qid in WIKIDATA_TYPE_MAP:
            assert qid.startswith("Q"), f"'{qid}' does not start with Q"
            assert qid[1:].isdigit(), f"'{qid}' is not a valid QID format"

    def test_values_are_title_case(self):
        """SEED_TYPES uses title case — map values must match."""
        for qid, entity_type in WIKIDATA_TYPE_MAP.items():
            assert entity_type == entity_type.title(), (
                f"WIKIDATA_TYPE_MAP['{qid}'] = '{entity_type}' is not title case"
            )


class TestNormalizeType:
    """normalize_type() is the canonical case normalizer for type comparisons."""

    def test_lowercase_output(self):
        assert normalize_type("Technology") == "technology"

    def test_already_lowercase(self):
        assert normalize_type("person") == "person"

    def test_uppercase_input(self):
        assert normalize_type("ORGANIZATION") == "organization"

    def test_mixed_case(self):
        assert normalize_type("ProGramMinG") == "programming"

    def test_empty_string(self):
        assert normalize_type("") == ""

    def test_consistent_across_calls(self):
        """Same input always produces same output."""
        for _ in range(100):
            assert normalize_type("Technology") == "technology"


class TestTypeToP31Qids:
    """type_to_p31_qids() reverse-maps SmartMemory types to Wikidata P31 QIDs."""

    def test_technology_returns_multiple_qids(self):
        qids = type_to_p31_qids("Technology")
        assert len(qids) >= 4  # programming language, framework, library, OS, ...
        assert "Q9143" in qids  # programming language
        assert "Q271680" in qids  # software framework

    def test_case_insensitive(self):
        """EntityRuler emits lowercase types — reverse lookup must be case-insensitive."""
        assert type_to_p31_qids("technology") == type_to_p31_qids("Technology")

    def test_person_returns_single_qid(self):
        qids = type_to_p31_qids("Person")
        assert qids == ["Q5"]

    def test_unknown_type_returns_empty(self):
        assert type_to_p31_qids("NonexistentType") == []

    def test_empty_string_returns_empty(self):
        assert type_to_p31_qids("") == []


class TestPublicEntity:
    """PublicEntity dataclass for canonical Wikidata entities."""

    def test_required_fields(self):
        entity = PublicEntity(qid="Q28865", label="Python")
        assert entity.qid == "Q28865"
        assert entity.label == "Python"

    def test_default_values(self):
        entity = PublicEntity(qid="Q28865", label="Python")
        assert entity.aliases == []
        assert entity.description == ""
        assert entity.entity_type == ""
        assert entity.instance_of == []
        assert entity.domain == ""
        assert entity.confidence == 1.0

    def test_full_construction(self):
        entity = PublicEntity(
            qid="Q28865",
            label="Python",
            aliases=["Python programming language", "Python 3"],
            description="General-purpose programming language",
            entity_type="Technology",
            instance_of=["Q9143"],
            domain="software",
            confidence=0.95,
        )
        assert entity.entity_type == "Technology"
        assert len(entity.aliases) == 2
        assert entity.confidence == 0.95


class TestGroundingDecision:
    """GroundingDecision records the outcome of each grounding attempt."""

    def test_grounded_decision(self):
        decision = GroundingDecision(
            surface_form="Python",
            candidates_found=1,
            selected_qid="Q28865",
            selected_label="Python",
            confidence=0.95,
            source="alias_lookup",
            disambiguation_reason="single_candidate",
            timestamp="2026-03-01T00:00:00Z",
        )
        assert decision.selected_qid == "Q28865"
        assert decision.source == "alias_lookup"

    def test_ungrounded_decision(self):
        decision = GroundingDecision(
            surface_form="FooBarBaz",
            candidates_found=0,
            selected_qid=None,
            selected_label=None,
            confidence=0.0,
            source="ungrounded",
            disambiguation_reason="ambiguous",
            timestamp="2026-03-01T00:00:00Z",
        )
        assert decision.selected_qid is None
        assert decision.source == "ungrounded"
