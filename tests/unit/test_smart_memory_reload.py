"""Tests for SmartMemory.reload_relation_overlays() (CORE-EXT-1c).

Verifies that reload correctly rebuilds the normalizer and validator with
fresh workspace overlays and patches the live OntologyConstrainStage.
"""

import json
from unittest.mock import MagicMock, patch

from smartmemory.relations.normalizer import RelationNormalizer
from smartmemory.relations.validator import TypePairValidator


def _mock_ontology_graph(promoted_types=None):
    """Build a mock OntologyGraph that returns promoted relation types."""
    og = MagicMock()
    types = []
    for rt in (promoted_types or []):
        types.append({
            "name": rt["name"],
            "status": "confirmed",
            "category": rt.get("category", "discovered"),
            "aliases": rt.get("aliases", []),
            "type_pairs": rt.get("type_pairs", []),
            "frequency": rt.get("frequency", 0),
            "promoted_at": None,
        })
    og.get_relation_types.return_value = types
    return og


def _make_smart_memory_with_reload(ontology_graph):
    """Create a minimal SmartMemory-like object with the reload path set up.

    Rather than constructing a full SmartMemory (which needs FalkorDB), we
    simulate the state that _create_pipeline_runner() produces: an instance
    with _ontology_graph, _relation_normalizer, _type_pair_validator, and
    _ontology_constrain_stage attributes.
    """
    from smartmemory.smart_memory import SmartMemory
    from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage

    # Build a minimal stage with initial normalizer/validator
    initial_normalizer = RelationNormalizer()
    initial_validator = TypePairValidator(mode="permissive")
    stage = OntologyConstrainStage(
        MagicMock(),  # ontology graph for entity type checks
        relation_normalizer=initial_normalizer,
        type_pair_validator=initial_validator,
    )

    # Simulate the instance attributes that _create_pipeline_runner() sets
    obj = object.__new__(SmartMemory)
    obj._ontology_graph = ontology_graph
    obj._relation_normalizer = initial_normalizer
    obj._type_pair_validator = initial_validator
    obj._ontology_constrain_stage = stage

    return obj, initial_normalizer, initial_validator, stage


class TestReloadRelationOverlays:

    def test_reload_updates_normalizer(self):
        """After promoting a type, reload makes the normalizer resolve its aliases."""
        og = _mock_ontology_graph([
            {"name": "supervises", "aliases": ["mentors_closely", "coaches_directly"]},
        ])
        obj, old_normalizer, _, _ = _make_smart_memory_with_reload(og)

        # Before reload: "mentors_closely" is unknown (not in seed ALIAS_INDEX)
        canonical, conf = old_normalizer.normalize("mentors_closely")
        assert conf == 0.0

        obj.reload_relation_overlays()

        # After reload: "mentors_closely" resolves to "supervises"
        canonical, conf = obj._relation_normalizer.normalize("mentors_closely")
        assert canonical == "supervises"
        assert conf == 1.0

    def test_reload_updates_validator(self):
        """After promoting a type with type_pairs, reload makes the validator accept them."""
        og = _mock_ontology_graph([
            {"name": "supervises", "type_pairs": [["robot", "task"]]},
        ])
        obj, _, old_validator, _ = _make_smart_memory_with_reload(og)

        # Before reload: ("robot", "task") is unknown for "supervises"
        is_valid, score = old_validator.validate("robot", "supervises", "task")
        assert score == 0.0

        obj.reload_relation_overlays()

        # After reload: ("robot", "task") is valid for "supervises"
        is_valid, score = obj._type_pair_validator.validate("robot", "supervises", "task")
        assert is_valid is True
        assert score == 1.0

    def test_reload_updates_stage_references(self):
        """After reload, the OntologyConstrainStage holds the new normalizer/validator."""
        og = _mock_ontology_graph([
            {"name": "supervises", "aliases": ["mentors_closely"]},
        ])
        obj, old_normalizer, old_validator, stage = _make_smart_memory_with_reload(og)

        assert stage._relation_normalizer is old_normalizer
        assert stage._type_pair_validator is old_validator

        obj.reload_relation_overlays()

        # Stage references updated to new instances
        assert stage._relation_normalizer is not old_normalizer
        assert stage._type_pair_validator is not old_validator
        assert stage._relation_normalizer is obj._relation_normalizer
        assert stage._type_pair_validator is obj._type_pair_validator

    def test_reload_noop_without_ontology_graph(self):
        """When _ontology_graph is not set, reload is a no-op."""
        from smartmemory.smart_memory import SmartMemory

        obj = object.__new__(SmartMemory)
        # No _ontology_graph attribute — should not raise
        obj.reload_relation_overlays()

    def test_reload_noop_without_stage(self):
        """When _ontology_constrain_stage is not set, normalizer/validator still update."""
        og = _mock_ontology_graph([
            {"name": "supervises", "aliases": ["mentors_closely"]},
        ])
        from smartmemory.smart_memory import SmartMemory

        obj = object.__new__(SmartMemory)
        obj._ontology_graph = og
        obj._relation_normalizer = RelationNormalizer()
        obj._type_pair_validator = TypePairValidator(mode="permissive")
        # No _ontology_constrain_stage — should not raise

        obj.reload_relation_overlays()

        canonical, conf = obj._relation_normalizer.normalize("mentors_closely")
        assert canonical == "supervises"
        assert conf == 1.0
