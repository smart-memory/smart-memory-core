"""Unit tests for PropertyConstraint model (OL-3)."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.ontology.models import PropertyConstraint, EntityTypeDefinition, Ontology
from datetime import datetime, UTC


class TestPropertyConstraint:
    """Tests for the PropertyConstraint dataclass."""

    def test_default_construction(self):
        """All fields have sane defaults."""
        pc = PropertyConstraint()
        assert pc.required is False
        assert pc.type == "string"
        assert pc.cardinality == "one"
        assert pc.kind == "soft"
        assert pc.enum_values == []

    def test_custom_construction(self):
        """Custom fields are set correctly."""
        pc = PropertyConstraint(
            required=True,
            type="enum",
            cardinality="many",
            kind="hard",
            enum_values=["a", "b", "c"],
        )
        assert pc.required is True
        assert pc.type == "enum"
        assert pc.cardinality == "many"
        assert pc.kind == "hard"
        assert pc.enum_values == ["a", "b", "c"]

    def test_to_dict_basic(self):
        """to_dict() produces a plain dict with expected keys."""
        pc = PropertyConstraint(required=True, type="number", kind="hard")
        d = pc.to_dict()
        assert d == {"required": True, "type": "number", "cardinality": "one", "kind": "hard"}

    def test_to_dict_includes_enum_values_when_present(self):
        """enum_values appear in dict only when non-empty."""
        pc = PropertyConstraint(type="enum", enum_values=["x", "y"])
        d = pc.to_dict()
        assert d["enum_values"] == ["x", "y"]

    def test_to_dict_excludes_enum_values_when_empty(self):
        """enum_values are excluded from dict when empty."""
        pc = PropertyConstraint(type="string")
        d = pc.to_dict()
        assert "enum_values" not in d

    def test_from_dict_basic(self):
        """from_dict() creates a PropertyConstraint from a plain dict."""
        d = {"required": True, "type": "number", "cardinality": "many", "kind": "hard"}
        pc = PropertyConstraint.from_dict(d)
        assert pc.required is True
        assert pc.type == "number"
        assert pc.cardinality == "many"
        assert pc.kind == "hard"

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys in the dict are silently ignored."""
        d = {"required": True, "type": "string", "unknown_field": "ignored"}
        pc = PropertyConstraint.from_dict(d)
        assert pc.required is True
        assert pc.type == "string"

    def test_round_trip(self):
        """to_dict() -> from_dict() preserves all values."""
        original = PropertyConstraint(
            required=True,
            type="enum",
            cardinality="many",
            kind="hard",
            enum_values=["a", "b"],
        )
        restored = PropertyConstraint.from_dict(original.to_dict())
        assert restored.required == original.required
        assert restored.type == original.type
        assert restored.cardinality == original.cardinality
        assert restored.kind == original.kind
        assert restored.enum_values == original.enum_values

    def test_list_defaults_are_independent(self):
        """Each instance gets its own list for enum_values."""
        a = PropertyConstraint()
        b = PropertyConstraint()
        a.enum_values.append("x")
        assert b.enum_values == []


class TestEntityTypeDefinitionConstraints:
    """Tests for property_constraints on EntityTypeDefinition."""

    def _make_entity_type(self, constraints=None):
        return EntityTypeDefinition(
            name="TestType",
            description="A test type",
            properties={"name": "string"},
            required_properties=set(),
            parent_types=set(),
            aliases=set(),
            examples=[],
            created_by="human",
            created_at=datetime.now(UTC),
            property_constraints=constraints or {},
        )

    def test_default_empty_constraints(self):
        """By default, property_constraints is empty dict."""
        et = self._make_entity_type()
        assert et.property_constraints == {}

    def test_constraints_set_on_construction(self):
        """Constraints dict is set properly on construction."""
        constraints = {
            "email": PropertyConstraint(required=True, type="string"),
            "age": PropertyConstraint(type="number", kind="hard"),
        }
        et = self._make_entity_type(constraints)
        assert len(et.property_constraints) == 2
        assert et.property_constraints["email"].required is True
        assert et.property_constraints["age"].kind == "hard"


class TestOntologyConstraintSerialization:
    """Tests for Ontology.to_dict() / from_dict() with property constraints."""

    def _make_ontology(self):
        ontology = Ontology("test-ontology", "1.0.0")
        et = EntityTypeDefinition(
            name="Person",
            description="A person",
            properties={"name": "string", "age": "number"},
            required_properties={"name"},
            parent_types=set(),
            aliases=set(),
            examples=["Alice"],
            created_by="human",
            created_at=datetime.now(UTC),
            property_constraints={
                "name": PropertyConstraint(required=True, type="string", kind="hard"),
                "age": PropertyConstraint(type="number", kind="soft"),
                "role": PropertyConstraint(type="enum", enum_values=["admin", "user"], kind="hard"),
            },
        )
        ontology.add_entity_type(et)
        return ontology

    def test_to_dict_includes_property_constraints(self):
        """to_dict() serializes property constraints on entity types."""
        ontology = self._make_ontology()
        d = ontology.to_dict()
        person = d["entity_types"]["person"]
        assert "property_constraints" in person
        assert person["property_constraints"]["name"]["required"] is True
        assert person["property_constraints"]["age"]["type"] == "number"
        assert person["property_constraints"]["role"]["enum_values"] == ["admin", "user"]

    def test_from_dict_parses_property_constraints(self):
        """from_dict() correctly parses property constraints back."""
        ontology = self._make_ontology()
        d = ontology.to_dict()
        restored = Ontology.from_dict(d)

        person = restored.get_entity_type("person")
        assert person is not None
        assert len(person.property_constraints) == 3
        assert isinstance(person.property_constraints["name"], PropertyConstraint)
        assert person.property_constraints["name"].required is True
        assert person.property_constraints["role"].enum_values == ["admin", "user"]

    def test_from_dict_without_constraints(self):
        """from_dict() works when property_constraints is absent (backward compat)."""
        ontology = Ontology("old-ontology", "0.9.0")
        et = EntityTypeDefinition(
            name="Thing",
            description="A thing",
            properties={},
            required_properties=set(),
            parent_types=set(),
            aliases=set(),
            examples=[],
            created_by="human",
            created_at=datetime.now(UTC),
        )
        ontology.add_entity_type(et)
        d = ontology.to_dict()
        # Remove property_constraints to simulate old data
        del d["entity_types"]["thing"]["property_constraints"]

        restored = Ontology.from_dict(d)
        thing = restored.get_entity_type("thing")
        assert thing.property_constraints == {}

    def test_round_trip_preserves_constraints(self):
        """Full round trip preserves all constraint details."""
        ontology = self._make_ontology()
        restored = Ontology.from_dict(ontology.to_dict())

        person = restored.get_entity_type("person")
        assert person is not None
        assert len(person.property_constraints) == 3

        name_c = person.property_constraints["name"]
        assert name_c.required is True
        assert name_c.type == "string"
        assert name_c.kind == "hard"

        age_c = person.property_constraints["age"]
        assert age_c.type == "number"
        assert age_c.kind == "soft"

        role_c = person.property_constraints["role"]
        assert role_c.type == "enum"
        assert role_c.enum_values == ["admin", "user"]
        assert role_c.kind == "hard"
