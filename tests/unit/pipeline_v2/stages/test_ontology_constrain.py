"""Unit tests for OntologyConstrainStage."""
from datetime import UTC

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig, ConstrainConfig, PromotionConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage


def _mock_ontology(type_statuses=None):
    """Build a mock OntologyGraph.

    Args:
        type_statuses: dict mapping type name (title-case) to status string.
            Returns None for types not in the dict.
    """
    og = MagicMock()
    statuses = type_statuses or {}

    def get_status(name):
        return statuses.get(name)

    og.get_type_status.side_effect = get_status
    og.add_provisional.return_value = True
    og.promote.return_value = True
    return og


class TestOntologyConstrainStage:
    """Tests for the ontology constrain pipeline stage."""

    def test_merge_ruler_and_llm_entities(self):
        """Ruler + LLM entities are merged by name, ruler type preferred."""
        og = _mock_ontology({"Person": "seed", "Organization": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            ruler_entities=[{"name": "John", "entity_type": "person", "confidence": 0.9}],
            llm_entities=[
                {"name": "John", "entity_type": "person", "confidence": 0.95},
                {"name": "Google", "entity_type": "organization", "confidence": 0.85},
            ],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        names = [e["name"] for e in result.entities]
        assert "John" in names
        assert "Google" in names
        # John should have higher confidence from LLM
        john = next(e for e in result.entities if e["name"] == "John")
        assert john["confidence"] == 0.95

    def test_seed_types_accepted(self):
        """Entities with seed type status are accepted."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            ruler_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Alice"

    def test_confirmed_types_accepted(self):
        """Entities with confirmed type status are accepted."""
        og = _mock_ontology({"Technology": "confirmed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Python", "entity_type": "technology", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1

    def test_provisional_types_accepted(self):
        """Entities with provisional type status are accepted."""
        og = _mock_ontology({"Framework": "provisional"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Django", "entity_type": "framework", "confidence": 0.8}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1

    def test_unknown_type_high_confidence_becomes_provisional(self):
        """Unknown type with confidence >= threshold creates provisional type."""
        og = _mock_ontology({})  # No known types
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "React", "entity_type": "library", "confidence": 0.8}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        og.add_provisional.assert_called_once_with("Library")
        assert len(result.entities) == 1
        assert len(result.promotion_candidates) == 1

    def test_unknown_type_low_confidence_rejected(self):
        """Unknown type with confidence < threshold is rejected."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Unknown", "entity_type": "mystery", "confidence": 0.3}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 0
        assert len(result.rejected) == 1

    def test_relation_filtering(self):
        """Only relations with both endpoints in accepted entities are kept."""
        og = _mock_ontology({"Person": "seed", "Organization": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Google", "entity_type": "organization", "confidence": 0.9, "item_id": "g1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "g1", "relation_type": "works_at"},
                {"source_id": "a1", "target_id": "unknown_id", "relation_type": "knows"},
            ],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.relations) == 1
        assert result.relations[0]["relation_type"] == "works_at"

    def test_max_entities_limit(self):
        """Entities are truncated to max_entities."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        entities = [{"name": f"Person{i}", "entity_type": "person", "confidence": 0.9} for i in range(30)]
        state = PipelineState(text="Test.", llm_entities=entities)
        config = PipelineConfig()
        config.extraction.constrain = ConstrainConfig(max_entities=5)

        result = stage.execute(state, config)

        assert len(result.entities) == 5

    def test_max_relations_limit(self):
        """Relations are truncated to max_relations."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        entities = [
            {"name": "A", "entity_type": "person", "confidence": 0.9, "item_id": "a"},
            {"name": "B", "entity_type": "person", "confidence": 0.9, "item_id": "b"},
        ]
        relations = [{"source_id": "a", "target_id": "b", "relation_type": f"r{i}"} for i in range(50)]
        state = PipelineState(text="Test.", llm_entities=entities, llm_relations=relations)
        config = PipelineConfig()
        config.extraction.constrain = ConstrainConfig(max_relations=3)

        result = stage.execute(state, config)

        assert len(result.relations) == 3

    def test_auto_promote_without_approval(self):
        """When require_approval=False, provisional types are auto-promoted."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Django", "entity_type": "framework", "confidence": 0.8}],
        )
        config = PipelineConfig()
        config.extraction.promotion = PromotionConfig(require_approval=False)

        stage.execute(state, config)

        og.promote.assert_called_once_with("Framework")

    def test_require_approval_skips_promote(self):
        """When require_approval=True, promote() is not called."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Django", "entity_type": "framework", "confidence": 0.8}],
        )
        config = PipelineConfig()
        config.extraction.promotion = PromotionConfig(require_approval=True)

        stage.execute(state, config)

        og.promote.assert_not_called()

    def test_undo_clears_all_outputs(self):
        """Undo resets entities, relations, rejected, promotion_candidates."""
        og = _mock_ontology()
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            entities=[{"name": "A"}],
            relations=[{"source_id": "a"}],
            rejected=[{"name": "B"}],
            promotion_candidates=[{"name": "C"}],
        )

        result = stage.undo(state)

        assert result.entities == []
        assert result.relations == []
        assert result.rejected == []
        assert result.promotion_candidates == []


class TestOntologyConstrainVersionCapture:
    """Tests for OL-2: version capture from registry."""

    def test_version_captured_from_registry(self):
        """When registry is provided, version and registry_id are captured."""
        og = _mock_ontology({"Person": "seed"})
        mock_registry = MagicMock()
        mock_registry.get_registry.return_value = {
            "id": "my-registry",
            "current_version": "v2.3.1",
        }
        stage = OntologyConstrainStage(og, ontology_registry=mock_registry)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.ontology_registry_id == "my-registry"
        assert result.ontology_version == "v2.3.1"

    def test_version_empty_without_registry(self):
        """When no registry is provided, version fields are empty strings."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.ontology_registry_id == ""
        assert result.ontology_version == ""

    def test_version_graceful_on_registry_error(self):
        """When registry.get_registry() raises, version fields default to empty."""
        og = _mock_ontology({"Person": "seed"})
        mock_registry = MagicMock()
        mock_registry.get_registry.side_effect = RuntimeError("DB down")
        stage = OntologyConstrainStage(og, ontology_registry=mock_registry)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.ontology_registry_id == ""
        assert result.ontology_version == ""
        # Should still process entities normally
        assert len(result.entities) == 1


class TestOntologyConstrainUnresolved:
    """Tests for OL-4: structured unresolved entity reporting."""

    def test_unresolved_entity_structure(self):
        """Rejected entities produce structured unresolved entries."""
        og = _mock_ontology({})  # No known types
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="The quick brown fox jumps.",
            llm_entities=[{"name": "Fox", "entity_type": "animal", "confidence": 0.3}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.unresolved_entities) == 1
        unresolved = result.unresolved_entities[0]
        assert unresolved["entity_name"] == "Fox"
        assert unresolved["attempted_type"] == "animal"
        assert unresolved["reason"] in ("unknown_type", "low_confidence")
        assert unresolved["confidence"] == 0.3
        assert "quick brown fox" in unresolved["source_content"]

    def test_no_unresolved_when_all_accepted(self):
        """No unresolved entries when all entities match known types."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.unresolved_entities == []

    def test_source_content_truncated(self):
        """source_content is truncated to 200 chars max."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        long_text = "A" * 500
        state = PipelineState(
            text=long_text,
            llm_entities=[{"name": "Thing", "entity_type": "mystery", "confidence": 0.1}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.unresolved_entities[0]["source_content"]) == 200

    def test_undo_clears_unresolved(self):
        """Undo resets unresolved_entities and constraint_violations."""
        og = _mock_ontology()
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            unresolved_entities=[{"entity_name": "X"}],
            constraint_violations=[{"entity_name": "Y"}],
        )

        result = stage.undo(state)

        assert result.unresolved_entities == []
        assert result.constraint_violations == []


class TestOntologyConstrainPropertyValidation:
    """Tests for OL-3: property constraint validation."""

    def _make_ontology_model(self, type_name, constraints):
        """Build a mock ontology model with property constraints."""
        from smartmemory.ontology.models import Ontology, EntityTypeDefinition, PropertyConstraint
        from datetime import datetime

        ontology = Ontology("test", "1.0.0")
        et = EntityTypeDefinition(
            name=type_name,
            description="Test type",
            properties={},
            required_properties=set(),
            parent_types=set(),
            aliases=set(),
            examples=[],
            created_by="human",
            created_at=datetime.now(UTC),
            property_constraints={name: PropertyConstraint(**params) for name, params in constraints.items()},
        )
        ontology.add_entity_type(et)
        return ontology

    def test_soft_violation_keeps_entity(self):
        """Soft constraint violations keep the entity and report violations."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"required": True, "type": "string", "kind": "soft"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # Entity kept despite missing required field (soft constraint)
        assert len(result.entities) == 1
        assert len(result.constraint_violations) == 1
        violation = result.constraint_violations[0]
        assert violation["entity_name"] == "Alice"
        assert violation["property_name"] == "email"
        assert violation["constraint_type"] == "required"
        assert violation["kind"] == "soft"

    def test_hard_violation_rejects_entity(self):
        """Hard constraint violations reject the entity."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"required": True, "type": "string", "kind": "hard"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # Entity rejected due to hard constraint
        assert len(result.entities) == 0
        assert len(result.rejected) == 1
        assert len(result.unresolved_entities) == 1
        assert result.unresolved_entities[0]["reason"] == "constraint_violation"

    def test_number_type_validation(self):
        """Number type constraint catches non-numeric values."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "age": {"type": "number", "kind": "soft"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9, "age": "not-a-number"}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.constraint_violations) == 1
        assert result.constraint_violations[0]["constraint_type"] == "type"
        assert result.constraint_violations[0]["expected"] == "number"

    def test_enum_validation(self):
        """Enum constraint catches values not in allowed list."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "role": {"type": "enum", "enum_values": ["admin", "user"], "kind": "soft"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9, "role": "superadmin"}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.constraint_violations) == 1
        assert result.constraint_violations[0]["constraint_type"] == "enum"

    def test_cardinality_validation(self):
        """Cardinality=one constraint catches list values."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"type": "string", "cardinality": "one", "kind": "soft"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "email": ["a@b.com", "c@d.com"]}
            ],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.constraint_violations) == 1
        assert result.constraint_violations[0]["constraint_type"] == "cardinality"

    def test_no_violations_when_no_model(self):
        """When no ontology_model is provided, no constraint validation occurs."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)  # No ontology_model
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.constraint_violations == []
        assert len(result.entities) == 1

    def test_valid_entity_passes_constraints(self):
        """Entity that satisfies all constraints passes without violations."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"required": True, "type": "string", "kind": "hard"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9, "email": "alice@example.com"}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1
        assert result.constraint_violations == []

    def test_required_check_allows_falsy_values(self):
        """Falsy values like 0 and False satisfy 'required' (only None triggers)."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "count": {"required": True, "type": "number", "kind": "hard"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9, "count": 0}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # count=0 is present — should NOT be flagged as missing
        assert len(result.entities) == 1
        assert result.constraint_violations == []

    def test_mixed_hard_and_soft_violations_on_same_entity(self):
        """Hard violation rejects entity even when soft violations are also present."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"required": True, "type": "string", "kind": "hard"},
                "phone": {"required": True, "type": "string", "kind": "soft"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # Hard violation on email → entity rejected
        assert len(result.entities) == 0
        assert len(result.rejected) == 1
        # Entity also appears as unresolved with constraint_violation reason
        assert len(result.unresolved_entities) == 1
        assert result.unresolved_entities[0]["reason"] == "constraint_violation"

    def test_multiple_soft_violations_all_recorded(self):
        """Multiple soft constraint failures on one entity are all recorded."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"required": True, "type": "string", "kind": "soft"},
                "age": {"type": "number", "kind": "soft"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9, "age": "not-a-number"}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # Entity kept (soft only) but both violations recorded
        assert len(result.entities) == 1
        assert len(result.constraint_violations) == 2
        types = {v["constraint_type"] for v in result.constraint_violations}
        assert types == {"required", "type"}

    def test_hard_rejection_removes_from_relation_filtering(self):
        """Entity rejected by hard constraint is excluded from relation endpoints."""
        og = _mock_ontology({"Person": "seed"})
        model = self._make_ontology_model(
            "person",
            {
                "email": {"required": True, "type": "string", "kind": "hard"},
            },
        )
        stage = OntologyConstrainStage(og, ontology_model=model)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Bob", "entity_type": "person", "confidence": 0.9, "email": "bob@test.com", "item_id": "b1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "b1", "relation_type": "knows"},
            ],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        # Alice rejected (missing email), Bob accepted
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Bob"
        # Relation Alice→Bob filtered out because Alice was rejected
        assert len(result.relations) == 0


class TestOntologyConstrainRelationTracking:
    """Tests for CORE-EXT-1c: inline relation frequency and novel label tracking."""

    def test_successful_normalization_increments_frequency(self):
        """When norm_conf > 0.0, increment_relation_frequency is called."""
        from smartmemory.relations.normalizer import RelationNormalizer

        og = _mock_ontology({"Person": "seed"})
        normalizer = RelationNormalizer()
        stage = OntologyConstrainStage(og, relation_normalizer=normalizer)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Google", "entity_type": "organization", "confidence": 0.9, "item_id": "g1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "g1", "relation_type": "works_at"},
            ],
        )
        config = PipelineConfig()

        stage.execute(state, config)

        og.increment_relation_frequency.assert_called()
        # Find the call for "works_at"
        calls = [c for c in og.increment_relation_frequency.call_args_list if c[0][0] == "works_at"]
        assert len(calls) >= 1

    def test_unknown_label_tracked_as_provisional(self):
        """When norm_conf == 0.0, add_provisional_relation_type is called."""
        from smartmemory.relations.normalizer import RelationNormalizer

        og = _mock_ontology({"Person": "seed"})
        normalizer = RelationNormalizer()
        stage = OntologyConstrainStage(og, relation_normalizer=normalizer)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Bob", "entity_type": "person", "confidence": 0.9, "item_id": "b1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "b1", "relation_type": "supervises_directly"},
            ],
        )
        config = PipelineConfig()

        stage.execute(state, config)

        # "supervises_directly" is not in ALIAS_INDEX, so norm_conf == 0.0
        og.add_provisional_relation_type.assert_called()

    def test_stopwords_not_tracked(self):
        """Relation labels that are stopwords (e.g. 'is', 'has') are not tracked."""
        from smartmemory.relations.normalizer import RelationNormalizer

        og = _mock_ontology({"Person": "seed"})
        normalizer = RelationNormalizer()
        stage = OntologyConstrainStage(og, relation_normalizer=normalizer)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Bob", "entity_type": "person", "confidence": 0.9, "item_id": "b1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "b1", "relation_type": "is"},
            ],
        )
        config = PipelineConfig()

        stage.execute(state, config)

        # "is" is in _RELATION_STOPWORDS — should NOT be tracked
        # add_provisional_relation_type should not be called for "is"
        for c in og.add_provisional_relation_type.call_args_list:
            assert c[0][0] != "is", "Stopword 'is' should not be tracked as provisional"

    def test_short_labels_not_tracked(self):
        """Relation labels shorter than 3 chars are not tracked."""
        from smartmemory.relations.normalizer import RelationNormalizer

        og = _mock_ontology({"Person": "seed"})
        normalizer = RelationNormalizer()
        stage = OntologyConstrainStage(og, relation_normalizer=normalizer)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Bob", "entity_type": "person", "confidence": 0.9, "item_id": "b1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "b1", "relation_type": "at"},
            ],
        )
        config = PipelineConfig()

        stage.execute(state, config)

        # "at" is only 2 chars — should NOT be tracked
        for c in og.add_provisional_relation_type.call_args_list:
            assert c[0][0] != "at", "Short label 'at' should not be tracked"


class TestOntologyConstrainRegistryEdgeCases:
    """Edge cases for OL-2 registry version capture."""

    def test_registry_partial_response_missing_version(self):
        """Registry returns id but no current_version."""
        og = _mock_ontology({"Person": "seed"})
        mock_registry = MagicMock()
        mock_registry.get_registry.return_value = {"id": "reg-1"}
        stage = OntologyConstrainStage(og, ontology_registry=mock_registry)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.ontology_registry_id == "reg-1"
        assert result.ontology_version == ""

    def test_registry_partial_response_missing_id(self):
        """Registry returns current_version but no id — falls back to 'default'."""
        og = _mock_ontology({"Person": "seed"})
        mock_registry = MagicMock()
        mock_registry.get_registry.return_value = {"current_version": "v1.0"}
        stage = OntologyConstrainStage(og, ontology_registry=mock_registry)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.ontology_registry_id == "default"
        assert result.ontology_version == "v1.0"

    def test_registry_empty_response(self):
        """Registry returns empty dict."""
        og = _mock_ontology({"Person": "seed"})
        mock_registry = MagicMock()
        mock_registry.get_registry.return_value = {}
        stage = OntologyConstrainStage(og, ontology_registry=mock_registry)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.ontology_registry_id == "default"
        assert result.ontology_version == ""
        assert len(result.entities) == 1
