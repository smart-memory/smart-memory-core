"""Cross-backend parity tests — same assertions on SQLite and FalkorDB.

Verifies that NetworkXAlgos (SQLite) and CypherAlgos (FalkorDB) produce
identical results for the same graph inputs.  Also covers inference engine
matchers vs Cypher, merge_nodes, and health metrics.

Requires: FalkorDB running on localhost:9010 (Docker)
"""
import json
import pytest
from uuid import uuid4

from smartmemory.graph.backends.sqlite import SQLiteBackend


def _make_falkordb():
    """Try to create a FalkorDB backend with a unique graph name. Returns None on failure."""
    try:
        from smartmemory.graph.backends.falkordb import FalkorDBBackend

        graph_name = f"test_parity_{uuid4().hex[:10]}"
        backend = FalkorDBBackend(host="localhost", port=9010, graph_name=graph_name)
        # Smoke-test the connection
        backend.clear()
        return backend
    except Exception:
        return None


@pytest.fixture(params=["sqlite", "falkordb"])
def backend(request, tmp_path):
    """Yield a clean backend for each parameterized run."""
    if request.param == "sqlite":
        yield SQLiteBackend(db_path=str(tmp_path / "parity.db"))
    else:
        b = _make_falkordb()
        if b is None:
            pytest.skip("FalkorDB not available on localhost:9010")
        yield b
        try:
            b.clear()
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _seed_linear(backend, n=4):
    """Create A->B->C->D chain with RELATES_TO edges."""
    ids = [chr(ord("a") + i) for i in range(n)]
    for nid in ids:
        backend.add_node(nid, {"memory_type": "semantic", "content": f"node {nid}"})
    for i in range(len(ids) - 1):
        backend.add_edge(ids[i], ids[i + 1], "RELATES_TO", {})
    return ids


def _seed_causal_chain(backend):
    """A-[CAUSES]->B-[CAUSES]->C, no direct A->C."""
    for nid in ("x", "y", "z"):
        backend.add_node(nid, {"memory_type": "semantic"})
    backend.add_edge("x", "y", "CAUSES", {"confidence": 0.9})
    backend.add_edge("y", "z", "CAUSES", {"confidence": 0.8})


def _seed_contradiction(backend):
    """A-[CONTRADICTS]->B, but no reverse."""
    backend.add_node("p", {"memory_type": "semantic"})
    backend.add_node("q", {"memory_type": "semantic"})
    backend.add_edge("p", "q", "CONTRADICTS", {})


def _seed_topic_inheritance(backend):
    """decision-[DERIVED_FROM]->semantic, no INFLUENCES reverse."""
    backend.add_node("dec", {"memory_type": "decision"})
    backend.add_node("sem", {"memory_type": "semantic"})
    backend.add_edge("dec", "sem", "DERIVED_FROM", {})


# ── GraphAlgos protocol parity ────────────────────────────────────────────────


class TestAlgosParity:
    """Verify GraphAlgos methods produce identical results on both backends."""

    def test_orphan_nodes(self, backend):
        _seed_linear(backend)
        backend.add_node("orphan", {"memory_type": "semantic"})
        orphans = backend.algos.orphan_nodes()
        assert orphans == ["orphan"]

    def test_edge_type_counts(self, backend):
        _seed_linear(backend, n=4)
        backend.add_edge("a", "c", "SIMILAR_TO", {})
        counts = backend.algos.edge_type_counts()
        assert counts["RELATES_TO"] == 3
        assert counts["SIMILAR_TO"] == 1

    def test_degree_map(self, backend):
        _seed_linear(backend, n=3)  # a->b->c
        dm = backend.algos.degree_map()
        # a: 1 out, b: 1 in + 1 out, c: 1 in
        assert dm["a"] == 1
        assert dm["b"] == 2
        assert dm["c"] == 1

    def test_find_paths(self, backend):
        _seed_linear(backend, n=4)  # a->b->c->d
        paths = backend.algos.find_paths("a", "d", max_depth=5)
        assert len(paths) >= 1
        # At least one path should go a -> b -> c -> d
        assert any(set(p) == {"a", "b", "c", "d"} for p in paths)

    def test_find_paths_no_connection(self, backend):
        backend.add_node("lone1", {"memory_type": "semantic"})
        backend.add_node("lone2", {"memory_type": "semantic"})
        paths = backend.algos.find_paths("lone1", "lone2")
        assert paths == []

    def test_shortest_path(self, backend):
        _seed_linear(backend, n=4)  # a->b->c->d
        path = backend.algos.shortest_path("a", "d")
        assert path is not None
        assert path[0] == "a"
        assert path[-1] == "d"
        assert len(path) == 4

    def test_shortest_path_unreachable(self, backend):
        backend.add_node("x1", {"memory_type": "semantic"})
        backend.add_node("x2", {"memory_type": "semantic"})
        assert backend.algos.shortest_path("x1", "x2") is None

    def test_transitive_closure(self, backend):
        _seed_causal_chain(backend)  # x-[CAUSES]->y-[CAUSES]->z
        pairs = backend.algos.transitive_closure("CAUSES")
        assert ("x", "z") in pairs

    def test_transitive_closure_with_direct_edge(self, backend):
        """No pair returned when direct edge already exists."""
        _seed_causal_chain(backend)
        backend.add_edge("x", "z", "CAUSES", {})
        pairs = backend.algos.transitive_closure("CAUSES")
        assert ("x", "z") not in pairs

    def test_pattern_match_2hop(self, backend):
        _seed_causal_chain(backend)  # x-[CAUSES]->y-[CAUSES]->z
        triples = backend.algos.pattern_match_2hop("CAUSES", "CAUSES")
        assert ("x", "y", "z") in triples

    def test_betweenness_centrality(self, backend):
        _seed_linear(backend, n=4)
        bc = backend.algos.betweenness_centrality()
        # Middle nodes b and c should have higher centrality
        assert bc.get("b", 0) > bc.get("a", 0)

    def test_connected_components(self, backend):
        _seed_linear(backend, n=3)  # a-b-c connected
        backend.add_node("island", {"memory_type": "semantic"})
        components = backend.algos.connected_components()
        assert len(components) == 2
        sizes = sorted(len(c) for c in components)
        assert sizes == [1, 3]


# ── Inference engine parity ───────────────────────────────────────────────────


class TestInferenceParity:
    """Verify InferenceEngine produces identical results on both backends."""

    def _make_engine(self, backend):
        from smartmemory.inference.engine import InferenceEngine

        class _GraphShim:
            def __init__(self, b):
                self.backend = b

        return InferenceEngine(memory=None, graph=_GraphShim(backend))

    def test_causal_transitivity(self, backend):
        _seed_causal_chain(backend)
        result = self._make_engine(backend).run()
        assert "causal_transitivity" in result.rules_applied
        assert result.edges_created >= 1
        # Verify the inferred edge exists
        edges = backend.get_edges_for_node("x")
        inferred = [e for e in edges if e.get("target_id") == "z" and e.get("edge_type") == "CAUSES"]
        assert len(inferred) == 1

    def test_contradiction_symmetry(self, backend):
        _seed_contradiction(backend)
        result = self._make_engine(backend).run()
        assert "contradiction_symmetry" in result.rules_applied
        edges_q = backend.get_edges_for_node("q")
        reverse = [
            e
            for e in edges_q
            if e.get("source_id") == "q" and e.get("target_id") == "p" and e.get("edge_type") == "CONTRADICTS"
        ]
        assert len(reverse) == 1

    def test_topic_inheritance(self, backend):
        _seed_topic_inheritance(backend)
        result = self._make_engine(backend).run()
        assert "topic_inheritance" in result.rules_applied
        edges_sem = backend.get_edges_for_node("sem")
        influences = [
            e
            for e in edges_sem
            if e.get("source_id") == "sem" and e.get("target_id") == "dec" and e.get("edge_type") == "INFLUENCES"
        ]
        assert len(influences) == 1

    def test_all_three_rules(self, backend):
        _seed_causal_chain(backend)
        _seed_contradiction(backend)
        _seed_topic_inheritance(backend)
        result = self._make_engine(backend).run()
        assert set(result.rules_applied) == {"causal_transitivity", "contradiction_symmetry", "topic_inheritance"}
        assert result.edges_created == 3

    def test_idempotent(self, backend):
        """Running twice should not create duplicate edges."""
        _seed_contradiction(backend)
        engine = self._make_engine(backend)
        r1 = engine.run()
        assert r1.edges_created == 1
        r2 = engine.run()
        # Second run: symmetry already established, no new edges
        assert r2.edges_created == 0


# ── merge_nodes parity ────────────────────────────────────────────────────────


class TestMergeNodesParity:
    """Verify merge_nodes() behaves identically on both backends."""

    def test_basic_merge(self, backend):
        backend.add_node("target", {"memory_type": "semantic", "content": "target"})
        backend.add_node("source", {"memory_type": "semantic", "content": "source"})
        backend.add_node("other", {"memory_type": "semantic"})
        backend.add_edge("source", "other", "RELATES_TO", {})

        result = backend.merge_nodes("target", ["source"])
        assert result is True
        assert backend.get_node("source") is None
        # Edge rewired to target
        edges = backend.get_edges_for_node("target")
        outgoing = [e for e in edges if e.get("source_id") == "target" and e.get("edge_type") == "RELATES_TO"]
        assert len(outgoing) == 1

    def test_missing_target_returns_false(self, backend):
        backend.add_node("src", {"content": "data"})
        result = backend.merge_nodes("nonexistent", ["src"])
        assert result is False
        assert backend.get_node("src") is not None


# ── Health metrics parity ─────────────────────────────────────────────────────


class TestHealthMetricsParity:
    """Verify GraphHealthChecker produces consistent metrics on both backends."""

    def test_health_report(self, backend):
        from smartmemory.metrics.graph_health import GraphHealthChecker

        # Seed some data
        _seed_linear(backend, n=3)
        backend.add_node("orphan", {"memory_type": "semantic"})

        # GraphHealthChecker(memory, graph) — pass None for memory, backend as graph
        # _backend property does getattr(self.graph, "backend", self.graph)
        checker = GraphHealthChecker(None, graph=backend)
        report = checker.collect_health()

        assert report.total_nodes == 4
        assert report.total_edges == 2
        assert report.orphan_count == 1
        assert "semantic" in report.type_distribution
        assert report.type_distribution["semantic"] == 4
        assert "RELATES_TO" in report.edge_distribution
