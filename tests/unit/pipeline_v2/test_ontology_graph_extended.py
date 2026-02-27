"""Unit tests for OntologyGraph Phase 4 extensions — frequency tracking, entity patterns, pattern layers."""

import pytest


pytestmark = pytest.mark.unit

from smartmemory.graph.ontology_graph import OntologyGraph


class ExtendedMockBackend:
    """In-memory mock that supports EntityType + EntityPattern nodes and edges."""

    def __init__(self):
        self._types: dict[str, dict] = {}  # name -> {status, frequency, avg_confidence}
        self._patterns: dict[tuple[str, str], dict] = {}  # (name, label) -> props
        self._edges: list[tuple[str, str, str]] = []  # (pattern_name, label, edge_type)

    def query(self, cypher: str, params=None, graph_name=None):
        params = params or {}

        # --- EntityType MERGE with frequency ---
        if "MERGE (t:EntityType {name: $name})" in cypher and "frequency" in cypher and "conf" in (
            cypher + str(params)
        ):
            name = params["name"]
            conf = params.get("conf", 0.0)
            if name not in self._types:
                self._types[name] = {"status": "provisional", "frequency": 0, "avg_confidence": 0.0}
            t = self._types[name]
            old_freq = t["frequency"]
            old_avg = t["avg_confidence"]
            t["frequency"] = old_freq + 1
            t["avg_confidence"] = (old_avg * old_freq + conf) / (old_freq + 1)
            return []

        # --- EntityPattern MERGE with IS_INSTANCE_OF ---
        if "MERGE (p:EntityPattern" in cypher and "IS_INSTANCE_OF" in cypher:
            name = params["name"]
            label = params["label"]
            conf = params.get("conf", 0.5)
            ws = params.get("ws", "default")
            glob = params.get("glob", False)
            source = params.get("source", "llm_discovery")
            initial_count = params.get("initial_count", 1)
            key = (name, label)
            if key not in self._patterns:
                self._patterns[key] = {
                    "name": name, "label": label, "confidence": conf,
                    "workspace_id": ws, "is_global": glob, "source": source, "count": initial_count,
                }
            else:
                p = self._patterns[key]
                p["count"] = p.get("count", 1) + 1
                p["confidence"] = (p["confidence"] + conf) / 2
            # Ensure the EntityType exists
            if label not in self._types:
                self._types[label] = {"status": "provisional", "frequency": 0, "avg_confidence": 0.0}
            self._edges.append((name, label, "IS_INSTANCE_OF"))
            return []

        # --- COALESCE frequency query ---
        if "COALESCE(t.frequency, 0)" in cypher and "EntityType" in cypher:
            name = params.get("name", "")
            if name in self._types:
                return [[self._types[name]["frequency"]]]
            return [[0]]

        # --- EntityPattern type assignments ---
        if "EntityPattern" in cypher and "IS_INSTANCE_OF" in cypher and "RETURN t.name" in cypher:
            name = params.get("name", "")
            results = []
            for (pname, label), props in self._patterns.items():
                if pname == name:
                    results.append([label, props.get("count", 1)])
            return results

        # --- EntityPattern listing for workspace ---
        if "EntityPattern" in cypher and "is_global" in cypher and "workspace_id" in cypher:
            ws = params.get("ws", "default")
            # Match production filter: COALESCE(p.count, 1) >= 2
            has_count_filter = "count" in cypher and ">= 2" in cypher
            results = []
            for (_name, _label), props in self._patterns.items():
                if has_count_filter and props.get("count", 1) < 2:
                    continue
                if props.get("is_global") or props.get("workspace_id") == ws:
                    results.append([props["name"], props["label"], props["confidence"], props["source"]])
            return results

        # --- Pattern stats ---
        if "EntityPattern" in cypher and "p.source, count(p)" in cypher:
            counts: dict[str, int] = {}
            for (_name, _label), props in self._patterns.items():
                src = props.get("source", "llm_discovery")
                counts[src] = counts.get(src, 0) + 1
            return [[src, cnt] for src, cnt in sorted(counts.items())]

        # --- Standard EntityType CREATE ---
        if "CREATE" in cypher and "EntityType" in cypher:
            name = params["name"]
            status = params["status"]
            self._types[name] = {"status": status, "frequency": 0, "avg_confidence": 0.0}
            return []

        # --- Standard EntityType SET (promote) ---
        if "SET" in cypher and "confirmed" in cypher:
            name = params.get("name", "")
            if name in self._types:
                self._types[name]["status"] = "confirmed"
                return [[name, "confirmed"]]
            return []

        # --- Standard EntityType MATCH ---
        if "MATCH" in cypher and "EntityType" in cypher and "RETURN" in cypher:
            if params and "name" in params:
                name = params["name"]
                if name in self._types:
                    return [[name, self._types[name]["status"]]]
                return []
            return sorted([[n, d["status"]] for n, d in self._types.items()])

        return []


@pytest.fixture
def backend():
    return ExtendedMockBackend()


@pytest.fixture
def graph(backend):
    return OntologyGraph(workspace_id="test", backend=backend)


# ------------------------------------------------------------------ #
# Frequency tracking
# ------------------------------------------------------------------ #


def test_increment_frequency_creates_type_if_missing(graph, backend):
    graph.increment_frequency("NewType", 0.9)
    assert backend._types["NewType"]["frequency"] == 1
    assert backend._types["NewType"]["avg_confidence"] == pytest.approx(0.9)


def test_increment_frequency_accumulates(graph):
    graph.increment_frequency("Tech", 0.8)
    graph.increment_frequency("Tech", 1.0)
    assert graph.get_frequency("Tech") == 2


def test_get_frequency_returns_zero_for_unknown(graph):
    assert graph.get_frequency("NonExistent") == 0


def test_increment_frequency_updates_avg_confidence(graph, backend):
    graph.increment_frequency("A", 0.6)
    graph.increment_frequency("A", 1.0)
    assert backend._types["A"]["avg_confidence"] == pytest.approx(0.8)


# ------------------------------------------------------------------ #
# Entity patterns
# ------------------------------------------------------------------ #


def test_add_entity_pattern_creates_pattern(graph, backend):
    result = graph.add_entity_pattern("python", "Technology", 0.95, is_global=True, source="seed")
    assert result is True
    assert ("python", "Technology") in backend._patterns


def test_add_entity_pattern_increments_count(graph, backend):
    graph.add_entity_pattern("react", "Technology", 0.9)
    graph.add_entity_pattern("react", "Technology", 0.8)
    assert backend._patterns[("react", "Technology")]["count"] == 2


def test_get_entity_patterns_filters_by_workspace(graph):
    graph.add_entity_pattern("python", "Technology", 0.9, workspace_id="test", is_global=False, initial_count=2)
    graph.add_entity_pattern("java", "Technology", 0.8, workspace_id="other", is_global=False, initial_count=2)
    graph.add_entity_pattern("docker", "Technology", 0.95, is_global=True, initial_count=2)

    patterns = graph.get_entity_patterns("test")
    names = {p["name"] for p in patterns}
    assert "python" in names
    assert "docker" in names  # global patterns included
    assert "java" not in names  # different workspace excluded


def test_get_type_assignments_returns_types(graph):
    graph.add_entity_pattern("alice", "Person", 0.9)
    assignments = graph.get_type_assignments("alice")
    assert len(assignments) == 1
    assert assignments[0]["type"] == "Person"


# ------------------------------------------------------------------ #
# initial_count threshold (CODE-DEV-4)
# ------------------------------------------------------------------ #


def test_initial_count_2_visible_immediately(graph):
    """Pattern created with initial_count=2 passes the count >= 2 threshold."""
    graph.add_entity_pattern("fastapi", "Technology", 0.9, workspace_id="test", initial_count=2)
    patterns = graph.get_entity_patterns("test")
    names = {p["name"] for p in patterns}
    assert "fastapi" in names


def test_initial_count_1_invisible_until_second_call(graph):
    """Default initial_count=1 means the pattern is invisible until seen twice."""
    graph.add_entity_pattern("flask", "Technology", 0.8, workspace_id="test")
    patterns = graph.get_entity_patterns("test")
    names = {p["name"] for p in patterns}
    assert "flask" not in names  # count=1, below threshold

    # Second call triggers ON MATCH SET count = count + 1 → count=2
    graph.add_entity_pattern("flask", "Technology", 0.85, workspace_id="test")
    patterns = graph.get_entity_patterns("test")
    names = {p["name"] for p in patterns}
    assert "flask" in names  # count=2, passes threshold


def test_seed_patterns_visible_via_get_entity_patterns(graph):
    """Seed patterns use initial_count=2 and are immediately visible."""
    graph.seed_entity_patterns({"numpy": "Technology"})
    patterns = graph.get_entity_patterns("test")
    names = {p["name"] for p in patterns}
    assert "numpy" in names


# ------------------------------------------------------------------ #
# Pattern layers (Step 6)
# ------------------------------------------------------------------ #


def test_seed_entity_patterns_creates_defaults(graph):
    created = graph.seed_entity_patterns()
    assert created > 0


def test_seed_entity_patterns_custom(graph):
    created = graph.seed_entity_patterns({"rust": "Technology", "meta": "Organization"})
    assert created == 2


def test_get_pattern_stats_returns_layer_counts(graph, backend):
    graph.add_entity_pattern("python", "Technology", 0.95, is_global=True, source="seed")
    graph.add_entity_pattern("react", "Technology", 0.9, source="promoted")
    graph.add_entity_pattern("fastapi", "Technology", 0.85, source="llm_discovery")

    stats = graph.get_pattern_stats()
    assert stats["seed"] == 1
    assert stats["promoted"] == 1
    assert stats["llm_discovery"] == 1


def test_pattern_layers_have_correct_source_values(graph, backend):
    """Seed, promoted, and tenant patterns use distinct source values."""
    graph.add_entity_pattern("docker", "Technology", 0.95, is_global=True, source="seed")
    graph.add_entity_pattern("k8s", "Technology", 0.9, is_global=True, source="promoted")
    graph.add_entity_pattern("mylib", "Technology", 0.7, workspace_id="acme", is_global=False, source="llm_discovery")

    seed_p = backend._patterns[("docker", "Technology")]
    promoted_p = backend._patterns[("k8s", "Technology")]
    tenant_p = backend._patterns[("mylib", "Technology")]

    assert seed_p["is_global"] is True and seed_p["source"] == "seed"
    assert promoted_p["is_global"] is True and promoted_p["source"] == "promoted"
    assert tenant_p["is_global"] is False and tenant_p["source"] == "llm_discovery"
    assert tenant_p["workspace_id"] == "acme"
