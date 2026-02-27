"""Unit tests for the three-layer entity pattern system.

| Layer | is_global | workspace_id | source |
|-------|-----------|--------------|--------|
| Seed  | true      | null         | seed   |
| Learned global | true | null   | promoted |
| Learned tenant | false | {ws}   | llm_discovery |
"""

import pytest


pytestmark = pytest.mark.unit

from smartmemory.graph.ontology_graph import OntologyGraph
from smartmemory.ontology.pattern_manager import PatternManager

from tests.unit.pipeline_v2.test_ontology_graph_extended import ExtendedMockBackend


@pytest.fixture
def backend():
    return ExtendedMockBackend()


@pytest.fixture
def graph(backend):
    return OntologyGraph(workspace_id="default", backend=backend)


# ------------------------------------------------------------------ #
# Layer isolation
# ------------------------------------------------------------------ #


def test_seed_layer_is_global(graph, backend):
    graph.seed_entity_patterns({"python": "Technology"})
    pattern = backend._patterns[("python", "Technology")]
    assert pattern["is_global"] is True
    assert pattern["source"] == "seed"


def test_promoted_layer_is_global(graph, backend):
    graph.add_entity_pattern("react", "Technology", 0.9, is_global=True, source="promoted", initial_count=2)
    pattern = backend._patterns[("react", "Technology")]
    assert pattern["is_global"] is True
    assert pattern["source"] == "promoted"


def test_tenant_layer_is_workspace_scoped(graph, backend):
    graph.add_entity_pattern("mylib", "Technology", 0.7, workspace_id="acme", is_global=False, source="llm_discovery", initial_count=2)
    pattern = backend._patterns[("mylib", "Technology")]
    assert pattern["is_global"] is False
    assert pattern["workspace_id"] == "acme"
    assert pattern["source"] == "llm_discovery"


# ------------------------------------------------------------------ #
# Pattern visibility
# ------------------------------------------------------------------ #


def test_tenant_patterns_only_visible_to_own_workspace(graph):
    graph.add_entity_pattern("internal-tool", "Tool", 0.8, workspace_id="acme", is_global=False, initial_count=2)
    graph.add_entity_pattern("shared-lib", "Technology", 0.9, is_global=True, source="seed", initial_count=2)

    acme_patterns = graph.get_entity_patterns("acme")
    other_patterns = graph.get_entity_patterns("other")

    acme_names = {p["name"] for p in acme_patterns}
    other_names = {p["name"] for p in other_patterns}

    assert "internal-tool" in acme_names
    assert "internal-tool" not in other_names
    assert "shared-lib" in acme_names  # global visible to acme
    assert "shared-lib" in other_names  # global visible to other


def test_pattern_manager_merges_all_visible_layers(graph):
    graph.seed_entity_patterns({"python": "Technology"})
    graph.add_entity_pattern("react", "Technology", 0.9, is_global=True, source="promoted", initial_count=2)
    graph.add_entity_pattern("mylib", "Technology", 0.7, workspace_id="default", is_global=False, source="llm_discovery", initial_count=2)

    pm = PatternManager(graph, workspace_id="default")
    patterns = pm.get_patterns()

    assert "python" in patterns
    assert "react" in patterns
    assert "mylib" in patterns


# ------------------------------------------------------------------ #
# Stats
# ------------------------------------------------------------------ #


def test_pattern_stats_counts_by_source(graph):
    graph.seed_entity_patterns({"python": "Technology", "react": "Technology"})
    graph.add_entity_pattern("fastapi", "Technology", 0.9, is_global=True, source="promoted", initial_count=2)
    graph.add_entity_pattern("mylib", "Technology", 0.7, source="llm_discovery", initial_count=2)

    stats = graph.get_pattern_stats()
    assert stats["seed"] == 2
    assert stats["promoted"] == 1
    assert stats["llm_discovery"] == 1
