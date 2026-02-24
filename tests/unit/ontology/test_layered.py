"""Tests for LayeredOntology merge class."""

import pytest


pytestmark = pytest.mark.unit
from datetime import datetime

from smartmemory.ontology.layered import LayeredOntology
from smartmemory.ontology.models import (
    EntityTypeDefinition,
    Ontology,
    OntologyRule,
    OntologySubscription,
    RelationshipTypeDefinition,
)


def _make_entity(name: str, desc: str = "test") -> EntityTypeDefinition:
    return EntityTypeDefinition(
        name=name,
        description=desc,
        properties={},
        required_properties=set(),
        parent_types=set(),
        aliases=set(),
        examples=[],
        created_by="test",
        created_at=datetime.now(),
    )


def _make_rel(name: str) -> RelationshipTypeDefinition:
    return RelationshipTypeDefinition(
        name=name,
        description="test rel",
        source_types=set(),
        target_types=set(),
        properties={},
    )


def _make_rule(rule_id: str, name: str) -> OntologyRule:
    return OntologyRule(
        id=rule_id,
        name=name,
        description="test rule",
        rule_type="validation",
        conditions={},
        actions={},
    )


def _make_ontology(name: str, entity_names: list[str] | None = None, rel_names: list[str] | None = None) -> Ontology:
    o = Ontology(name=name)
    for ename in (entity_names or []):
        o.add_entity_type(_make_entity(ename))
    for rname in (rel_names or []):
        o.add_relationship_type(_make_rel(rname))
    return o


class TestNoBase:
    """When base is None, LayeredOntology should behave like the overlay alone."""

    def test_entity_types_from_overlay_only(self):
        overlay = _make_ontology("overlay", ["Person", "Company"])
        layered = LayeredOntology(overlay=overlay)
        assert set(layered.entity_types.keys()) == {"person", "company"}

    def test_get_entity_type(self):
        overlay = _make_ontology("overlay", ["Person"])
        layered = LayeredOntology(overlay=overlay)
        assert layered.get_entity_type("person") is not None
        assert layered.get_entity_type("Missing") is None

    def test_rules_from_overlay(self):
        overlay = _make_ontology("overlay")
        overlay.add_rule(_make_rule("r1", "Rule1"))
        layered = LayeredOntology(overlay=overlay)
        assert "r1" in layered.rules

    def test_provenance_all_local(self):
        overlay = _make_ontology("overlay", ["Person"])
        layered = LayeredOntology(overlay=overlay)
        assert layered.get_provenance("person") == "local"

    def test_diff_no_base(self):
        overlay = _make_ontology("overlay", ["Person", "Company"])
        layered = LayeredOntology(overlay=overlay)
        diff = layered.compute_diff()
        assert diff.base_only == []
        assert sorted(diff.overlay_only) == ["company", "person"]
        assert diff.overridden == []
        assert diff.hidden == []


class TestMerge:
    """When base is provided, overlay wins on conflict."""

    def test_union_of_types(self):
        base = _make_ontology("base", ["Animal", "Plant"])
        overlay = _make_ontology("overlay", ["Robot"])
        layered = LayeredOntology(overlay=overlay, base=base)
        assert set(layered.entity_types.keys()) == {"animal", "plant", "robot"}

    def test_overlay_wins_on_conflict(self):
        base = _make_ontology("base", ["Person"])
        overlay = _make_ontology("overlay")
        # Add a Person with different description to overlay
        overlay.add_entity_type(_make_entity("Person", desc="overlay-version"))
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.entity_types["person"].description == "overlay-version"

    def test_get_entity_type_overlay_first(self):
        base = _make_ontology("base", ["Person"])
        overlay = _make_ontology("overlay")
        overlay.add_entity_type(_make_entity("Person", desc="overlay"))
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.get_entity_type("Person").description == "overlay"

    def test_get_entity_type_falls_through_to_base(self):
        base = _make_ontology("base", ["Animal"])
        overlay = _make_ontology("overlay", ["Robot"])
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.get_entity_type("Animal") is not None

    def test_relationship_types_merge(self):
        base = _make_ontology("base", rel_names=["KNOWS"])
        overlay = _make_ontology("overlay", rel_names=["WORKS_AT"])
        layered = LayeredOntology(overlay=overlay, base=base)
        assert set(layered.relationship_types.keys()) == {"knows", "works_at"}

    def test_relationship_overlay_wins(self):
        base = _make_ontology("base")
        base.add_relationship_type(_make_rel("KNOWS"))
        overlay = _make_ontology("overlay")
        overlay_knows = _make_rel("KNOWS")
        overlay_knows.description = "overlay-rel"
        overlay.add_relationship_type(overlay_knows)
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.relationship_types["knows"].description == "overlay-rel"

    def test_get_relationship_type_overlay_first(self):
        base = _make_ontology("base", rel_names=["KNOWS"])
        overlay = _make_ontology("overlay")
        overlay_knows = _make_rel("KNOWS")
        overlay_knows.description = "overlay-rel"
        overlay.add_relationship_type(overlay_knows)
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.get_relationship_type("KNOWS").description == "overlay-rel"

    def test_get_relationship_type_falls_through(self):
        base = _make_ontology("base", rel_names=["KNOWS"])
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.get_relationship_type("KNOWS") is not None

    def test_rules_overlay_only(self):
        base = _make_ontology("base")
        base.add_rule(_make_rule("br1", "BaseRule"))
        overlay = _make_ontology("overlay")
        overlay.add_rule(_make_rule("or1", "OverlayRule"))
        layered = LayeredOntology(overlay=overlay, base=base)
        assert "or1" in layered.rules
        assert "br1" not in layered.rules


class TestHiddenTypes:
    """Hidden types should be excluded from merged entity_types."""

    def test_hidden_excluded_from_entity_types(self):
        base = _make_ontology("base", ["Animal", "Plant"])
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"animal"})
        assert "animal" not in layered.entity_types
        assert "plant" in layered.entity_types

    def test_hidden_case_insensitive(self):
        base = _make_ontology("base", ["Person"])
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"PERSON"})
        assert "person" not in layered.entity_types

    def test_get_entity_type_returns_none_for_hidden(self):
        base = _make_ontology("base", ["Animal"])
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"animal"})
        assert layered.get_entity_type("Animal") is None

    def test_hidden_type_also_in_overlay_overlay_wins(self):
        """If a type is hidden AND overridden in overlay, overlay wins."""
        base = _make_ontology("base", ["Person"])
        overlay = _make_ontology("overlay")
        overlay.add_entity_type(_make_entity("Person", desc="overlay-person"))
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"person"})
        # Overlay override should be visible despite hidden
        assert layered.get_entity_type("Person") is not None
        assert layered.get_entity_type("Person").description == "overlay-person"
        assert "person" in layered.entity_types
        assert layered.get_provenance("person") == "override"


class TestProvenance:
    def test_local(self):
        overlay = _make_ontology("overlay", ["Robot"])
        layered = LayeredOntology(overlay=overlay)
        assert layered.get_provenance("robot") == "local"

    def test_base(self):
        base = _make_ontology("base", ["Animal"])
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.get_provenance("animal") == "base"

    def test_override(self):
        base = _make_ontology("base", ["Person"])
        overlay = _make_ontology("overlay", ["Person"])
        layered = LayeredOntology(overlay=overlay, base=base)
        assert layered.get_provenance("person") == "override"

    def test_hidden(self):
        base = _make_ontology("base", ["Animal"])
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"animal"})
        assert layered.get_provenance("animal") == "hidden"

    def test_unknown(self):
        overlay = _make_ontology("overlay")
        layered = LayeredOntology(overlay=overlay)
        assert layered.get_provenance("nonexistent") == "unknown"

    def test_provenance_map(self):
        base = _make_ontology("base", ["Animal", "Plant"])
        overlay = _make_ontology("overlay", ["Plant", "Robot"])
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"animal"})
        prov = layered.get_provenance_map()
        assert prov["animal"] == "hidden"
        assert prov["plant"] == "override"
        assert prov["robot"] == "local"


class TestDiff:
    def test_full_diff(self):
        base = _make_ontology("base", ["Animal", "Plant", "Hidden"])
        overlay = _make_ontology("overlay", ["Plant", "Robot"])
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"hidden"})
        diff = layered.compute_diff()
        assert diff.base_only == ["animal"]
        assert diff.overlay_only == ["robot"]
        assert diff.overridden == ["plant"]
        assert diff.hidden == ["hidden"]


class TestDetach:
    def test_detach_produces_flat_ontology(self):
        base = _make_ontology("base", ["Animal", "Plant"])
        overlay = _make_ontology("overlay", ["Robot"])
        overlay.subscription = OntologySubscription(base_registry_id="base-id")
        layered = LayeredOntology(overlay=overlay, base=base)
        flat = layered.detach()
        assert flat.subscription is None
        assert flat.is_base_layer is False
        assert set(flat.entity_types.keys()) == {"animal", "plant", "robot"}

    def test_detach_respects_hidden(self):
        base = _make_ontology("base", ["Animal", "Plant"])
        overlay = _make_ontology("overlay", ["Robot"])
        overlay.subscription = OntologySubscription(base_registry_id="base-id", hidden_types={"animal"})
        layered = LayeredOntology(overlay=overlay, base=base, hidden_types={"animal"})
        flat = layered.detach()
        assert "animal" not in flat.entity_types
        assert "plant" in flat.entity_types
        assert "robot" in flat.entity_types

    def test_detach_copies_base_relationships(self):
        base = _make_ontology("base", rel_names=["KNOWS"])
        overlay = _make_ontology("overlay", rel_names=["WORKS_AT"])
        layered = LayeredOntology(overlay=overlay, base=base)
        flat = layered.detach()
        assert set(flat.relationship_types.keys()) == {"knows", "works_at"}


class TestToDict:
    def test_includes_provenance(self):
        base = _make_ontology("base", ["Animal"])
        overlay = _make_ontology("overlay", ["Robot"])
        layered = LayeredOntology(overlay=overlay, base=base)
        d = layered.to_dict()
        assert "provenance" in d
        assert d["provenance"]["animal"] == "base"
        assert d["provenance"]["robot"] == "local"

    def test_merged_entity_types_in_dict(self):
        base = _make_ontology("base", ["Animal"])
        overlay = _make_ontology("overlay", ["Robot"])
        layered = LayeredOntology(overlay=overlay, base=base)
        d = layered.to_dict()
        assert "animal" in d["entity_types"]
        assert "robot" in d["entity_types"]


class TestProxyProperties:
    def test_id_name_version_tenant(self):
        overlay = _make_ontology("overlay")
        overlay.tenant_id = "t1"
        layered = LayeredOntology(overlay=overlay)
        assert layered.id == overlay.id
        assert layered.name == "overlay"
        assert layered.version == overlay.version
        assert layered.tenant_id == "t1"
