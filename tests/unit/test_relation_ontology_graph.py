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
        import json as _json

        params = params or {}
        # Simple pattern matching for the queries used by OntologyGraph

        # --- RelationType ---
        if "CREATE (t:RelationType" in cypher:
            key = f"RelationType:{params['name']}"
            self._nodes[key] = {"name": params["name"], "status": params.get("status", "seed"),
                                "category": params.get("category", ""),
                                "aliases": None, "type_pairs": None,
                                "frequency": 0, "promoted_at": None}
            return []
        if "MERGE (t:RelationType {name: $name})" in cypher and "SET t.frequency" in cypher:
            # increment_relation_frequency
            key = f"RelationType:{params['name']}"
            if key not in self._nodes:
                self._nodes[key] = {"name": params["name"], "status": "provisional",
                                    "category": "", "aliases": None, "type_pairs": None,
                                    "frequency": 0, "promoted_at": None}
            self._nodes[key]["frequency"] = self._nodes[key].get("frequency", 0) + 1
            self._nodes[key]["last_seen"] = params.get("now")
            return []
        if "MERGE (t:RelationType {name: $name})" in cypher and "status = 'confirmed'" in cypher:
            # promote_relation_type
            key = f"RelationType:{params['name']}"
            if key not in self._nodes:
                self._nodes[key] = {"name": params["name"], "status": "confirmed",
                                    "category": params.get("category", ""),
                                    "aliases": params.get("aliases"), "type_pairs": params.get("type_pairs"),
                                    "frequency": 0, "promoted_at": params.get("now")}
            else:
                self._nodes[key]["status"] = "confirmed"
                self._nodes[key]["category"] = params.get("category", "")
                self._nodes[key]["aliases"] = params.get("aliases")
                self._nodes[key]["type_pairs"] = params.get("type_pairs")
                self._nodes[key]["promoted_at"] = params.get("now")
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
                    results.append([v["name"], v["status"], v.get("category", ""),
                                    v.get("aliases"), v.get("type_pairs"),
                                    v.get("frequency", 0), v.get("promoted_at")])
            return results
        if "MATCH (t:RelationType)" in cypher and "RETURN" in cypher:
            results = []
            for k, v in self._nodes.items():
                if k.startswith("RelationType:"):
                    results.append([v["name"], v["status"], v.get("category", ""),
                                    v.get("aliases"), v.get("type_pairs"),
                                    v.get("frequency", 0), v.get("promoted_at")])
            return results

        # --- NovelRelationLabel ---
        if "MATCH (n:NovelRelationLabel {name: $name})" in cypher and "RETURN n.name, n.frequency" in cypher:
            key = f"NovelRelationLabel:{params['name']}"
            if key in self._nodes:
                return [[self._nodes[key]["name"], self._nodes[key]["frequency"]]]
            return []
        if "MATCH (n:NovelRelationLabel {name: $name})" in cypher and "n.frequency + 1" in cypher:
            key = f"NovelRelationLabel:{params['name']}"
            if key in self._nodes:
                self._nodes[key]["frequency"] += 1
                self._nodes[key]["last_seen"] = params.get("now")
            return []
        if "CREATE (n:NovelRelationLabel" in cypher:
            key = f"NovelRelationLabel:{params['name']}"
            self._nodes[key] = {
                "name": params["name"], "frequency": 1, "status": "tracking",
                "raw_examples": params.get("raw_examples", "[]"),
                "source_types": params.get("source_types", "[]"),
                "target_types": params.get("target_types", "[]"),
                "workspace_id": params.get("ws", ""),
                "first_seen": params.get("now"), "last_seen": params.get("now"),
            }
            return []
        if "MATCH (n:NovelRelationLabel {name: $name}) SET n.status = 'promoted'" in cypher:
            key = f"NovelRelationLabel:{params['name']}"
            if key in self._nodes:
                self._nodes[key]["status"] = "promoted"
            return []
        if "MATCH (n:NovelRelationLabel)" in cypher and "WHERE" in cypher:
            # get_novel_relation_labels
            min_freq = params.get("min_freq", 1)
            status_filter = params.get("status")
            results = []
            for k, v in self._nodes.items():
                if not k.startswith("NovelRelationLabel:"):
                    continue
                if v["frequency"] < min_freq:
                    continue
                if status_filter and v["status"] != status_filter:
                    continue
                results.append([
                    v["name"], v["frequency"], v["status"],
                    v.get("raw_examples", "[]"), v.get("source_types", "[]"),
                    v.get("target_types", "[]"), v.get("workspace_id", ""),
                    v.get("first_seen"), v.get("last_seen"),
                ])
            # Sort by frequency DESC
            results.sort(key=lambda r: r[1], reverse=True)
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


class TestIncrementRelationFrequency:
    """Tests for CORE-EXT-1c: increment_relation_frequency."""

    def test_increment_creates_node(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.increment_relation_frequency("custom_rel", confidence=0.5)
        # Should create a RelationType node with frequency=1
        types = og.get_relation_types()
        custom = [t for t in types if t["name"] == "custom_rel"]
        assert len(custom) == 1
        assert custom[0]["frequency"] == 1

    def test_increment_twice(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.increment_relation_frequency("custom_rel", confidence=0.5)
        og.increment_relation_frequency("custom_rel", confidence=0.8)
        types = og.get_relation_types()
        custom = [t for t in types if t["name"] == "custom_rel"]
        assert custom[0]["frequency"] == 2


class TestAddNovelRelationLabel:
    """Tests for CORE-EXT-1c: add_novel_relation_label."""

    def test_add_new_returns_true(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        result = og.add_novel_relation_label("supervises", raw_examples=["supervises"])
        assert result is True

    def test_add_existing_returns_false(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_novel_relation_label("supervises")
        result = og.add_novel_relation_label("supervises")
        assert result is False

    def test_frequency_increments_on_repeat(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_novel_relation_label("supervises")
        og.add_novel_relation_label("supervises")
        labels = og.get_novel_relation_labels(min_frequency=1)
        assert len(labels) == 1
        assert labels[0]["frequency"] == 2


class TestGetNovelRelationLabels:
    """Tests for CORE-EXT-1c: get_novel_relation_labels."""

    def test_min_frequency_filter(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_novel_relation_label("rare_label")
        og.add_novel_relation_label("common_label")
        og.add_novel_relation_label("common_label")  # freq=2
        og.add_novel_relation_label("common_label")  # freq=3
        labels = og.get_novel_relation_labels(min_frequency=3)
        assert len(labels) == 1
        assert labels[0]["name"] == "common_label"

    def test_status_filter(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_novel_relation_label("label_a")
        og.add_novel_relation_label("label_b")
        # Promote label_b by simulating promote_relation_type (which marks NovelRelationLabel as promoted)
        og.promote_relation_type("label_b")
        tracking = og.get_novel_relation_labels(status="tracking")
        assert len(tracking) == 1
        assert tracking[0]["name"] == "label_a"

    def test_empty_when_none(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        assert og.get_novel_relation_labels() == []

    def test_json_decoded_fields(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_novel_relation_label(
            "mentors",
            raw_examples=["mentors", "mentoring"],
            source_types=["person"],
            target_types=["person"],
        )
        labels = og.get_novel_relation_labels()
        assert labels[0]["raw_examples"] == ["mentors", "mentoring"]
        assert labels[0]["source_types"] == ["person"]
        assert labels[0]["target_types"] == ["person"]


class TestPromoteRelationType:
    """Tests for CORE-EXT-1c: promote_relation_type."""

    def test_promote_creates_confirmed_node(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        result = og.promote_relation_type(
            "supervises", category="discovered",
            aliases=["oversees", "manages_directly"],
            type_pairs=[("person", "person")],
        )
        assert result is True
        confirmed = og.get_relation_types(status="confirmed")
        assert len(confirmed) == 1
        assert confirmed[0]["name"] == "supervises"
        assert confirmed[0]["status"] == "confirmed"

    def test_promote_marks_novel_label_promoted(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.add_novel_relation_label("supervises")
        og.promote_relation_type("supervises")
        promoted = og.get_novel_relation_labels(status="promoted")
        assert len(promoted) == 1
        assert promoted[0]["name"] == "supervises"


class TestGetRelationTypesExtendedFields:
    """Tests for CORE-EXT-1c: extended fields in get_relation_types."""

    def test_returns_aliases_and_type_pairs(self):
        import json
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.promote_relation_type(
            "supervises", aliases=["oversees"],
            type_pairs=[("person", "person")],
        )
        types = og.get_relation_types(status="confirmed")
        assert len(types) == 1
        assert types[0]["aliases"] == ["oversees"]
        assert types[0]["type_pairs"] == [["person", "person"]]  # JSON round-trip: tuples → lists

    def test_seed_types_have_empty_aliases(self):
        og = OntologyGraph(workspace_id="test", backend=FakeBackend())
        og.seed_relation_types()
        types = og.get_relation_types(status="seed")
        for t in types:
            assert t["aliases"] == []
            assert t["type_pairs"] == []
            assert t["frequency"] == 0
