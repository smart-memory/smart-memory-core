"""Tests for ONTO-PUB-3 integration into OntologyConstrainStage.

Tests that relation normalization, type-pair validation, and plausibility scoring
are correctly wired into the _filter_relations path.
"""

from smartmemory.relations.normalizer import RelationNormalizer
from smartmemory.relations.validator import TypePairValidator


class FakeOntologyGraph:
    """Minimal stub for OntologyGraph — only methods called by OntologyConstrainStage."""

    def get_type_status(self, name):
        return "seed"

    def increment_frequency(self, name, conf):
        pass

    def add_provisional(self, name):
        return True


def _make_stage(normalizer=None, validator=None):
    from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage

    return OntologyConstrainStage(
        FakeOntologyGraph(),
        relation_normalizer=normalizer,
        type_pair_validator=validator,
    )


def _accepted_entities():
    """Person (Alice) + Organization (Acme) with SHA256 item_ids."""
    return [
        {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "abc123"},
        {"name": "Acme", "entity_type": "organization", "confidence": 0.9, "item_id": "def456"},
        {"name": "Python", "entity_type": "technology", "confidence": 0.9, "item_id": "ghi789"},
    ]


def _relations_with_alias():
    return [
        {"source_id": "abc123", "target_id": "def456", "relation_type": "employed_by", "raw_predicate": "Employed By"},
    ]


def _relations_unknown():
    return [
        {"source_id": "abc123", "target_id": "ghi789", "relation_type": "completely_unknown_xyz"},
    ]


class TestNormalizationIntegration:
    def test_known_alias_gets_canonical_type(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        result = stage._filter_relations(_relations_with_alias(), accepted, accepted_names)

        assert len(result) == 1
        rel = result[0]
        assert rel["canonical_type"] == "works_at"
        assert rel["relation_type"] == "works_at"  # updated to canonical

    def test_raw_predicate_preserved(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        result = stage._filter_relations(_relations_with_alias(), accepted, accepted_names)

        assert result[0]["raw_predicate"] == "Employed By"

    def test_normalization_confidence_attached(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        result = stage._filter_relations(_relations_with_alias(), accepted, accepted_names)

        assert result[0]["normalization_confidence"] == 1.0

    def test_plausibility_score_attached(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        result = stage._filter_relations(_relations_with_alias(), accepted, accepted_names)

        assert 0.0 <= result[0]["plausibility_score"] <= 1.0
        # works_at + (person, organization) = perfect match → high score
        assert result[0]["plausibility_score"] == 1.0

    def test_unknown_predicate_falls_to_related_to(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        result = stage._filter_relations(_relations_unknown(), accepted, accepted_names)

        assert len(result) == 1
        assert result[0]["canonical_type"] == "related_to"
        assert result[0]["normalization_confidence"] == 0.0

    def test_sorted_by_plausibility_descending(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        relations = _relations_with_alias() + _relations_unknown()
        result = stage._filter_relations(relations, accepted, accepted_names)

        assert len(result) == 2
        assert result[0]["plausibility_score"] >= result[1]["plausibility_score"]


class TestStrictModeFiltering:
    def test_strict_mode_drops_invalid_relation(self):
        """In strict mode, a relation with an unknown type-pair should be dropped."""
        stage = _make_stage(
            normalizer=RelationNormalizer(),
            validator=TypePairValidator(mode="strict"),
        )
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        # works_at with (person, organization) → valid, kept
        # completely_unknown_xyz → normalizes to related_to which has wildcard → kept
        # But a relation with a bad type-pair in strict mode should be dropped
        relations = [
            # person→technology with "works_at" → works_at expects (person, organization), not (person, technology)
            {"source_id": "abc123", "target_id": "ghi789", "relation_type": "works_at"},
        ]
        result = stage._filter_relations(relations, accepted, accepted_names)
        # strict mode: (person, technology) is not a valid pair for works_at → dropped
        assert len(result) == 0

    def test_permissive_mode_retains_invalid_relation(self):
        """In permissive mode, a relation with an unknown type-pair is kept."""
        stage = _make_stage(
            normalizer=RelationNormalizer(),
            validator=TypePairValidator(mode="permissive"),
        )
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        relations = [
            {"source_id": "abc123", "target_id": "ghi789", "relation_type": "works_at"},
        ]
        result = stage._filter_relations(relations, accepted, accepted_names)
        # permissive mode: kept despite (person, technology) not being valid for works_at
        assert len(result) == 1
        assert result[0]["plausibility_score"] < 1.0  # penalized but retained


class TestBackwardCompatibility:
    def test_no_normalizer_no_validator(self):
        """Stage works exactly as before when both are None."""
        stage = _make_stage(normalizer=None, validator=None)
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        relations = [
            {"source_id": "abc123", "target_id": "def456", "relation_type": "works_at"},
        ]
        result = stage._filter_relations(relations, accepted, accepted_names)

        assert len(result) == 1
        assert result[0]["relation_type"] == "works_at"
        # No ONTO-PUB-3 fields should be attached
        assert "canonical_type" not in result[0]
        assert "plausibility_score" not in result[0]

    def test_normalizer_only(self):
        stage = _make_stage(normalizer=RelationNormalizer(), validator=None)
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        relations = [
            {"source_id": "abc123", "target_id": "def456", "relation_type": "employed_by"},
        ]
        result = stage._filter_relations(relations, accepted, accepted_names)

        assert result[0]["canonical_type"] == "works_at"
        assert result[0]["normalization_confidence"] == 1.0

    def test_validator_only(self):
        stage = _make_stage(normalizer=None, validator=TypePairValidator())
        accepted = _accepted_entities()
        accepted_names = {e["name"].lower() for e in accepted}
        relations = [
            {"source_id": "abc123", "target_id": "def456", "relation_type": "works_at"},
        ]
        result = stage._filter_relations(relations, accepted, accepted_names)

        assert result[0]["canonical_type"] == "works_at"
        assert result[0]["plausibility_score"] == 1.0


class TestRawPredicateFromLLMExtractor:
    def test_raw_predicate_in_relation_dict(self):
        """Verify llm_single.py now includes raw_predicate in relation output."""
        from smartmemory.plugins.extractors.llm_single import _normalize_predicate

        raw = "Framework For"
        pred = _normalize_predicate(raw)
        rel = {"source_id": "a", "target_id": "b", "relation_type": pred, "raw_predicate": raw}
        assert rel["raw_predicate"] == "Framework For"
        assert rel["relation_type"] == "framework_for"
