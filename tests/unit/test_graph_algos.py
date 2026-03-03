"""Tests for Phase 2 of DIST-LITE-PARITY-1: GraphAlgos protocol layer.

Covers:
- GraphComputeLayer sync hooks (add/remove node/edge, clear, reload)
- NetworkXAlgos correctness on a small graph
- SQLiteBackend.algos end-to-end wiring
- GraphAlgos protocol compliance
"""

import pytest

from smartmemory.graph.algos import GraphAlgos
from smartmemory.graph.backends.sqlite import SQLiteBackend
from smartmemory.graph.compute import GraphComputeLayer
from smartmemory.graph.networkx_algos import NetworkXAlgos


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def backend(tmp_path):
    """File-based SQLiteBackend with GraphComputeLayer already wired."""
    return SQLiteBackend(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def populated(backend):
    """Backend with a small diamond-shaped graph:

        A ──RELATES_TO──> B
        A ──RELATES_TO──> C
        B ──CAUSES──────> D
        C ──CAUSES──────> D
        E  (orphan, no edges)
    """
    backend.add_node("A", {"content": "Node A", "memory_type": "semantic"})
    backend.add_node("B", {"content": "Node B", "memory_type": "semantic"})
    backend.add_node("C", {"content": "Node C", "memory_type": "episodic"})
    backend.add_node("D", {"content": "Node D", "memory_type": "semantic"})
    backend.add_node("E", {"content": "Orphan E", "memory_type": "working"})
    backend.add_edge("A", "B", "RELATES_TO", {})
    backend.add_edge("A", "C", "RELATES_TO", {})
    backend.add_edge("B", "D", "CAUSES", {})
    backend.add_edge("C", "D", "CAUSES", {})
    return backend


# ── GraphComputeLayer sync tests ─────────────────────────────────────────────


class TestGraphComputeLayerSync:
    """Verify the in-memory DiGraph stays in sync with SQL mutations."""

    def test_initial_load_empty(self, backend):
        g = backend._compute.nx_graph
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_sync_add_node(self, backend):
        backend.add_node("n1", {"content": "hello"})
        g = backend._compute.nx_graph
        assert "n1" in g.nodes

    def test_sync_add_edge(self, backend):
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "LINKS_TO", {})
        g = backend._compute.nx_graph
        assert g.has_edge("a", "b", key="LINKS_TO")
        # MultiDiGraph: g[u][v] is a dict of {key: edge_data}
        assert g["a"]["b"]["LINKS_TO"]["edge_type"] == "LINKS_TO"

    def test_sync_remove_node(self, backend):
        backend.add_node("x", {})
        backend.add_node("y", {})
        backend.add_edge("x", "y", "REL", {})
        assert "x" in backend._compute.nx_graph.nodes
        backend.remove_node("x")
        assert "x" not in backend._compute.nx_graph.nodes
        # Edge should also be gone (CASCADE in SQL, remove_node in nx removes edges too)
        assert not backend._compute.nx_graph.has_edge("x", "y")

    def test_sync_remove_edge(self, backend):
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "REL", {})
        backend.remove_edge("a", "b", "REL")
        assert not backend._compute.nx_graph.has_edge("a", "b")

    def test_sync_clear(self, backend):
        backend.add_node("n1", {})
        backend.add_node("n2", {})
        backend.add_edge("n1", "n2", "REL", {})
        backend.clear()
        g = backend._compute.nx_graph
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_sync_deserialize_reload(self, backend):
        backend.add_node("a", {"content": "alpha", "memory_type": "semantic"})
        backend.add_node("b", {"content": "beta", "memory_type": "semantic"})
        backend.add_edge("a", "b", "REL", {})
        data = backend.serialize()
        backend.clear()
        assert backend._compute.nx_graph.number_of_nodes() == 0
        backend.deserialize(data)
        g = backend._compute.nx_graph
        assert g.number_of_nodes() == 2
        assert g.has_edge("a", "b")

    def test_populated_graph_node_count(self, populated):
        assert populated._compute.nx_graph.number_of_nodes() == 5

    def test_populated_graph_edge_count(self, populated):
        assert populated._compute.nx_graph.number_of_edges() == 4


# ── NetworkXAlgos tests ──────────────────────────────────────────────────────


class TestNetworkXAlgos:
    """Verify each GraphAlgos method on the diamond graph."""

    def test_orphan_nodes(self, populated):
        orphans = populated.algos.orphan_nodes()
        assert orphans == ["E"]

    def test_edge_type_counts(self, populated):
        counts = populated.algos.edge_type_counts()
        assert counts == {"RELATES_TO": 2, "CAUSES": 2}

    def test_degree_map(self, populated):
        degrees = populated.algos.degree_map()
        # A: 2 out, B: 1 in + 1 out = 2, C: 1 in + 1 out = 2, D: 2 in, E: 0
        assert degrees["A"] == 2
        assert degrees["B"] == 2
        assert degrees["C"] == 2
        assert degrees["D"] == 2
        assert degrees["E"] == 0

    def test_find_paths_A_to_D(self, populated):
        paths = populated.algos.find_paths("A", "D", max_depth=3)
        # Two paths: A→B→D and A→C→D
        assert len(paths) == 2
        path_sets = {tuple(p) for p in paths}
        assert ("A", "B", "D") in path_sets
        assert ("A", "C", "D") in path_sets

    def test_find_paths_no_connection(self, populated):
        paths = populated.algos.find_paths("E", "A")
        assert paths == []

    def test_find_paths_nonexistent_node(self, populated):
        paths = populated.algos.find_paths("Z", "A")
        assert paths == []

    def test_shortest_path(self, populated):
        path = populated.algos.shortest_path("A", "D")
        # Either A→B→D or A→C→D (both length 2)
        assert path is not None
        assert len(path) == 3
        assert path[0] == "A"
        assert path[-1] == "D"

    def test_shortest_path_no_connection(self, populated):
        assert populated.algos.shortest_path("E", "A") is None

    def test_shortest_path_nonexistent_node(self, populated):
        assert populated.algos.shortest_path("Z", "A") is None

    def test_transitive_closure(self, populated):
        # A→B and A→C via RELATES_TO.  No length-2+ chain of RELATES_TO exists,
        # so transitive closure should be empty.
        closure = populated.algos.transitive_closure("RELATES_TO")
        assert closure == []

    def test_transitive_closure_with_chain(self, backend):
        """Chain: X──T──>Y──T──>Z. Closure should find (X, Z)."""
        backend.add_node("X", {})
        backend.add_node("Y", {})
        backend.add_node("Z", {})
        backend.add_edge("X", "Y", "T", {})
        backend.add_edge("Y", "Z", "T", {})
        closure = backend.algos.transitive_closure("T")
        assert ("X", "Z") in closure

    def test_pattern_match_2hop(self, populated):
        # A──RELATES_TO──>B──CAUSES──>D  and  A──RELATES_TO──>C──CAUSES──>D
        matches = populated.algos.pattern_match_2hop("RELATES_TO", "CAUSES")
        assert len(matches) == 2
        match_set = {(a, b, c) for a, b, c in matches}
        assert ("A", "B", "D") in match_set
        assert ("A", "C", "D") in match_set

    def test_betweenness_centrality(self, populated):
        bc = populated.algos.betweenness_centrality()
        # All 5 nodes should be present
        assert len(bc) == 5
        # B and C are bridges between A and D
        assert bc["B"] > 0 or bc["C"] > 0

    def test_betweenness_centrality_empty(self, backend):
        assert backend.algos.betweenness_centrality() == {}

    def test_connected_components(self, populated):
        components = populated.algos.connected_components()
        # Two components: {A, B, C, D} and {E}
        assert len(components) == 2
        sizes = sorted(len(c) for c in components)
        assert sizes == [1, 4]

    def test_connected_components_empty(self, backend):
        assert backend.algos.connected_components() == []


# ── Protocol compliance ──────────────────────────────────────────────────────


class TestProtocolCompliance:
    """Verify that concrete implementations satisfy the GraphAlgos protocol."""

    def test_networkx_algos_is_graph_algos(self, populated):
        assert isinstance(populated.algos, GraphAlgos)

    def test_backend_algos_property_type(self, populated):
        # .algos should be a NetworkXAlgos for SQLiteBackend
        assert isinstance(populated.algos, NetworkXAlgos)

    def test_backend_abc_algos_raises_by_default(self):
        """SmartGraphBackend.algos default raises NotImplementedError."""
        from smartmemory.graph.backends.backend import SmartGraphBackend

        # Can't instantiate ABC directly, but we can test the property
        # descriptor on a minimal subclass
        class _Stub(SmartGraphBackend):
            def add_node(self, *a, **kw): ...
            def clear(self): ...
            def add_edge(self, *a, **kw): ...
            def get_node(self, *a, **kw): ...
            def get_neighbors(self, *a, **kw): ...
            def remove_node(self, *a, **kw): ...
            def remove_edge(self, *a, **kw): ...
            def search_nodes(self, *a, **kw): ...
            def get_edges_for_node(self, *a, **kw): ...
            def get_all_edges(self, *a, **kw): ...
            def serialize(self, *a, **kw): ...
            def deserialize(self, *a, **kw): ...

        stub = _Stub()
        with pytest.raises(NotImplementedError, match="does not provide graph algorithms"):
            _ = stub.algos


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestAlgosEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_node_no_crash(self, backend):
        backend.add_node("solo", {})
        assert backend.algos.orphan_nodes() == ["solo"]
        assert backend.algos.edge_type_counts() == {}
        assert backend.algos.degree_map() == {"solo": 0}
        assert backend.algos.connected_components() == [{"solo"}]

    def test_self_loop_handling(self, backend):
        backend.add_node("a", {})
        backend.add_edge("a", "a", "SELF_REF", {})
        assert backend.algos.edge_type_counts() == {"SELF_REF": 1}
        # Degree should count the self-loop
        assert backend.algos.degree_map()["a"] >= 1

    def test_algos_after_mutations(self, backend):
        """Algos reflect state changes after add/remove."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "REL", {})
        assert backend.algos.orphan_nodes() == []

        backend.remove_edge("a", "b", "REL")
        orphans = set(backend.algos.orphan_nodes())
        assert orphans == {"a", "b"}

    def test_algos_after_clear_and_repopulate(self, populated):
        assert populated.algos.connected_components() != []
        populated.clear()
        assert populated.algos.connected_components() == []
        # Re-add a node
        populated.add_node("new", {})
        assert populated.algos.orphan_nodes() == ["new"]


# ── Regression tests ─────────────────────────────────────────────────────────


class TestParallelEdgeRegression:
    """Regression: parallel edges between the same node pair must coexist."""

    def test_parallel_edges_both_visible(self, backend):
        """Two edge types between same pair — both must appear in algos."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "T1", {})
        backend.add_edge("a", "b", "T2", {})
        counts = backend.algos.edge_type_counts()
        assert counts == {"T1": 1, "T2": 1}

    def test_remove_one_parallel_edge_keeps_other(self, backend):
        """Removing T1 must NOT remove T2 between same pair."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "T1", {})
        backend.add_edge("a", "b", "T2", {})
        backend.remove_edge("a", "b", "T1")
        counts = backend.algos.edge_type_counts()
        assert counts == {"T2": 1}
        # Nodes still connected via T2
        assert backend.algos.orphan_nodes() == []

    def test_remove_all_edges_without_type(self, backend):
        """remove_edge without edge_type removes ALL edges between pair."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "T1", {})
        backend.add_edge("a", "b", "T2", {})
        backend.remove_edge("a", "b")  # no edge_type
        assert backend.algos.edge_type_counts() == {}

    def test_pattern_match_2hop_with_parallel_edges(self, backend):
        """2-hop pattern works when node pair has multiple edge types."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_node("c", {})
        backend.add_edge("a", "b", "REL", {})
        backend.add_edge("a", "b", "CAUSES", {})
        backend.add_edge("b", "c", "CAUSES", {})
        # Only a──REL──>b──CAUSES──>c should match, not a──CAUSES──>b──CAUSES──>c
        # (wait — CAUSES→CAUSES is also a valid 2-hop)
        rel_then_causes = backend.algos.pattern_match_2hop("REL", "CAUSES")
        assert ("a", "b", "c") in rel_then_causes
        causes_then_causes = backend.algos.pattern_match_2hop("CAUSES", "CAUSES")
        assert ("a", "b", "c") in causes_then_causes

    def test_deserialize_preserves_parallel_edges(self, backend):
        """serialize → clear → deserialize preserves parallel edges."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        backend.add_edge("a", "b", "T1", {})
        backend.add_edge("a", "b", "T2", {})
        data = backend.serialize()
        backend.clear()
        backend.deserialize(data)
        counts = backend.algos.edge_type_counts()
        assert counts == {"T1": 1, "T2": 1}


class TestSyncAddEdgeKwargsRegression:
    """Regression: sync_add_edge must not crash on properties containing edge_type."""

    def test_properties_with_edge_type_key(self, backend):
        """Properties dict containing 'edge_type' must not cause TypeError."""
        backend.add_node("a", {})
        backend.add_node("b", {})
        # This previously threw TypeError: got multiple values for keyword argument 'edge_type'
        backend.add_edge("a", "b", "REL", {"edge_type": "REL", "weight": 0.5})
        assert backend.algos.edge_type_counts() == {"REL": 1}

    def test_properties_with_source_target_keys(self, backend):
        """Properties dict containing 'source_id'/'target_id' must not crash."""
        backend.add_node("x", {})
        backend.add_node("y", {})
        backend.add_edge("x", "y", "LINK", {"source_id": "x", "target_id": "y"})
        assert backend.algos.edge_type_counts() == {"LINK": 1}
