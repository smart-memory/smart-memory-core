"""Tests for OntologyGraph RelationType support (ONTO-PUB-3).

Uses a mock backend to avoid FalkorDB dependency.
"""

from unittest.mock import MagicMock

from smartmemory.graph.ontology_graph import OntologyGraph, SEED_RELATION_TYPES


class FakeBackend:
    """In-memory backend stub for OntologyGraph tests."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}  # keyed by "label:name"

    def query(self, cypher: str, params=None, graph_name=None):
        params = params or {}
        # Simple pattern matching for the queries used by OntologyGraph
        if "CREATE (t:RelationType" in cypher:
            key = f"RelationType:{params['name']}"
            self._nodes[key] = {"name": params["name"], "status": params.get("status", "seed"),
                                "category": params.get("category", "")}
            return []
        if "MATCH (t:RelationType {name: $name})" in cypher and "RETURN t.name" in cypher:
            key = f"RelationType:{params['name']}"
            if key in self._nodes:
                return [[self._nodes[key]["name"]]]
            return []
        if "MATCH (t:RelationType {status: $status})" in cypher:
            results = []
            for k, v in self._nodes.items():
                if k.startswith("RelationType:") and v["status"] == params["status"]:
                    results.append([v["name"], v["status"], v.get("category", "")])
            return results
        if "MATCH (t:RelationType)" in cypher and "RETURN" in cypher:
            results = []
            for k, v in self._nodes.items():
                if k.startswith("RelationType:"):
                    results.append([v["name"], v["status"], v.get("category", "")])
            return results
        if "MERGE (s:EntityType" in cypher and "VALID_FOR" in cypher:
            return []
        # EntityType queries from seed_types
        if "CREATE (t:EntityType" in cypher:
            key = f"EntityType:{params['name']}"
            self._nodes[key] = {"name": params["name"], "status": params.get("status", "seed")}
            return []
        if "MATCH (t:EntityType {name: $name})" in cypher:
            key = f"EntityType:{params['name']}"
            if key in self._nodes:
                return [[self._nodes[key]["name"], self._nodes[key]["status"]]]
            return []
        return []


class TestSeedRelationTypes:
    def test_seed_creates_nodes(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        count = og.seed_relation_types()
        assert count == len(SEED_RELATION_TYPES)
        assert count > 0

    def test_seed_is_idempotent(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        count1 = og.seed_relation_types()
        count2 = og.seed_relation_types()
        assert count1 > 0
        assert count2 == 0  # all already exist

    def test_seed_relation_types_constant_populated(self):
        assert len(SEED_RELATION_TYPES) == 39


class TestGetRelationTypes:
    def test_returns_all(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.seed_relation_types()
        types = og.get_relation_types()
        assert len(types) == 39
        assert all("name" in t and "status" in t and "category" in t for t in types)

    def test_filter_by_status(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.seed_relation_types()
        seed_types = og.get_relation_types(status="seed")
        assert len(seed_types) == 39
        provisional = og.get_relation_types(status="provisional")
        assert len(provisional) == 0


class TestAddProvisionalRelationType:
    def test_add_new(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        result = og.add_provisional_relation_type("custom_relation", category="custom")
        assert result is True

    def test_add_existing_noop(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_provisional_relation_type("custom_relation")
        result = og.add_provisional_relation_type("custom_relation")
        assert result is False


class TestSeedTypePairEdges:
    def test_ensures_edges(self):
        backend = FakeBackend()
        og = OntologyGraph(workspace_id="test", backend=backend)
        og.seed_types()  # need EntityType nodes first
        ensured = og.seed_type_pair_edges()
        # Should ensure edges for concrete type-pairs (excludes wildcards)
        assert ensured > 0

    def test_idempotent_count(self):
        """Calling seed_type_pair_edges twice returns the same count (MERGE is idempotent)."""
        backend = FakeBackend()
        og = OntologyGraph(workspace_id="test", backend=backend)
        og.seed_types()
        first = og.seed_type_pair_edges()
        second = og.seed_type_pair_edges()
        assert first == second
