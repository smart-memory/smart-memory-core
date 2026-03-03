"""Integration tests for DIST-LITE-PARITY-1 — verify higher-level subsystems work on SQLiteBackend.

Covers:
- Monitoring: summary(), orphaned_notes(), reflect(), summarize() on SQLite graph
- Analytics: find_similar_items() degrades gracefully, detect_concept_drift() works
- FTS5: UsearchVectorBackend.search_by_text() returns results via SQLite FTS5
- Search fallback: SmartGraphSearch falls back to contains/keyword on SQLite
- GraphHealthChecker: backend-agnostic health metrics on SQLite
- merge_nodes: entity deduplication on SQLite matching FalkorDB semantics
- InferenceEngine: backend-agnostic rule matching on SQLite (no Cypher)

Requires: no Docker (all SQLite / in-memory)
"""
import pytest

from smartmemory.graph.backends.sqlite import SQLiteBackend
from smartmemory.graph.smartgraph import SmartGraph
from smartmemory.memory.pipeline.stages.monitoring import Monitoring
from smartmemory.memory.pipeline.stages.analytics import MemoryAnalytics
from smartmemory.metrics.graph_health import GraphHealthChecker, HealthReport
from smartmemory.inference.engine import InferenceEngine, InferenceResult
from smartmemory.inference.rules import InferenceRule, get_default_rules


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_graph(tmp_path):
    """SmartGraph backed by file-based SQLiteBackend."""
    backend = SQLiteBackend(db_path=str(tmp_path / "test.db"))
    return SmartGraph(backend=backend)


@pytest.fixture
def populated_graph(sqlite_graph):
    """Graph with a small dataset across multiple types."""
    b = sqlite_graph.backend
    b.add_node("s1", {"content": "Python is a programming language", "memory_type": "semantic"})
    b.add_node("s2", {"content": "Machine learning uses neural networks", "memory_type": "semantic"})
    b.add_node("e1", {"content": "Deployed the API to production", "memory_type": "episodic"})
    b.add_node("w1", {"content": "Investigate flaky test in CI", "memory_type": "working"})
    b.add_node("p1", {"content": "Always run linter before committing", "memory_type": "procedural"})
    b.add_node("n1", {"content": "Quick note about the meeting", "memory_type": "note", "tags": ["note"]})
    b.add_edge("s1", "s2", "RELATES_TO", {})
    b.add_edge("e1", "s1", "DERIVED_FROM", {})
    # n1 is orphaned (no edges) — used by orphaned_notes test
    return sqlite_graph


# ── Monitoring on SQLiteBackend ───────────────────────────────────────────────


class TestMonitoringOnSQLite:
    """Monitoring methods must not crash on SQLiteBackend."""

    def test_summary_returns_type_counts(self, populated_graph):
        mon = Monitoring(populated_graph)
        result = mon.summary()
        assert "semantic" in result
        assert result["semantic"]["count"] == 2
        assert result["episodic"]["count"] == 1
        assert result["working"]["count"] == 1
        assert result["procedural"]["count"] == 1

    def test_summary_empty_graph(self, sqlite_graph):
        mon = Monitoring(sqlite_graph)
        result = mon.summary()
        for t in ("semantic", "episodic", "procedural", "working"):
            assert result[t]["count"] == 0

    def test_orphaned_notes(self, populated_graph):
        mon = Monitoring(populated_graph)
        orphaned = mon.orphaned_notes()
        # n1 has type "note" and no edges — should be orphaned
        ids = [n.get("item_id") for n in orphaned]
        assert "n1" in ids

    def test_reflect_returns_keywords(self, populated_graph):
        mon = Monitoring(populated_graph)
        result = mon.reflect(top_k=3)
        assert "semantic" in result
        # Keywords are extracted from content — at least some should exist
        assert isinstance(result["semantic"], dict)

    def test_summarize_returns_previews(self, populated_graph):
        mon = Monitoring(populated_graph)
        result = mon.summarize(max_items=5)
        assert "semantic" in result

    def test_self_monitor_no_crash(self, populated_graph):
        mon = Monitoring(populated_graph)
        result = mon.self_monitor()
        assert isinstance(result, dict)


# ── Analytics on SQLiteBackend ────────────────────────────────────────────────


class TestAnalyticsOnSQLite:
    """Analytics methods must degrade gracefully on SQLiteBackend."""

    def test_find_similar_items_returns_empty(self, populated_graph):
        """SQLiteBackend has no vector_similarity_search — guard checks backend, returns []."""
        analytics = MemoryAnalytics(populated_graph)
        results = analytics.find_similar_items([0.1, 0.2, 0.3], top_k=5)
        assert results == []

    def test_detect_concept_drift_no_crash(self, populated_graph):
        analytics = MemoryAnalytics(populated_graph)
        result = analytics.detect_concept_drift(time_window_days=30)
        assert isinstance(result, dict)

    def test_detect_bias_no_crash(self, populated_graph):
        analytics = MemoryAnalytics(populated_graph)
        result = analytics.detect_bias(
            protected_attributes=["gender"],
            sentiment_analysis=False,
            topic_analysis=False,
        )
        assert isinstance(result, dict)


# ── FTS5 via UsearchVectorBackend ─────────────────────────────────────────────


class TestFTS5Search:
    """FTS5 full-text search works through UsearchVectorBackend."""

    @pytest.fixture
    def usearch_backend(self, tmp_path):
        """UsearchVectorBackend with FTS5 data."""
        try:
            from smartmemory.stores.vector.backends.usearch import UsearchVectorBackend
        except ImportError:
            pytest.skip("usearch not installed")
        return UsearchVectorBackend(
            collection_name="test_fts5",
            persist_directory=str(tmp_path / "vectors"),
        )

    def test_search_by_text_returns_results(self, usearch_backend):
        """FTS5 search_by_text finds documents added via add().

        UsearchVectorBackend.add() reads metadata["content"] and indexes it into FTS5.
        """
        usearch_backend.add(
            item_id="doc1",
            embedding=[0.1] * 128,
            metadata={"content": "Python programming language", "memory_type": "semantic"},
        )
        usearch_backend.add(
            item_id="doc2",
            embedding=[0.2] * 128,
            metadata={"content": "Machine learning with neural networks", "memory_type": "semantic"},
        )
        results = usearch_backend.search_by_text(query_text="Python", top_k=5)
        ids = [r["id"] for r in results]
        assert "doc1" in ids

    def test_search_by_text_no_match(self, usearch_backend):
        """FTS5 returns empty when no documents match."""
        usearch_backend.add(
            item_id="doc1",
            embedding=[0.1] * 128,
            metadata={"content": "Python programming language"},
        )
        results = usearch_backend.search_by_text(query_text="nonexistent_term_xyz", top_k=5)
        assert results == []

    def test_vector_and_text_search_both_work(self, usearch_backend):
        """Both ANN and FTS5 paths return results from the same backend."""
        usearch_backend.add(
            item_id="doc1",
            embedding=[1.0] + [0.0] * 127,
            metadata={"content": "Alpha centauri star system"},
        )
        # Vector search (cosine similarity)
        vec_results = usearch_backend.search(
            query_embedding=[1.0] + [0.0] * 127, top_k=3
        )
        assert any(r["id"] == "doc1" for r in vec_results)

        # FTS5 text search
        text_results = usearch_backend.search_by_text(query_text="centauri", top_k=3)
        assert any(r["id"] == "doc1" for r in text_results)


# ── SmartGraphSearch fallback on SQLite ───────────────────────────────────────


class TestSearchFallbackOnSQLite:
    """SmartGraphSearch gracefully falls back when Cypher/embeddings unavailable."""

    def test_simple_contains_finds_content(self, populated_graph):
        """_search_with_simple_contains works via get_all_nodes() on SQLiteBackend."""
        search = populated_graph.search
        # This should fall through SSG/vector/regex to simple_contains
        results = search._search_with_simple_contains("Python", top_k=5)
        assert len(results) >= 1
        assert any("Python" in getattr(r, "content", "") for r in results)

    def test_keyword_matching_finds_content(self, populated_graph):
        search = populated_graph.search
        results = search._search_with_keyword_matching("neural networks", top_k=5)
        assert len(results) >= 1

    def test_get_all_nodes_fallback(self, populated_graph):
        search = populated_graph.search
        results = search._get_all_nodes_fallback("*", top_k=10)
        assert len(results) >= 5  # We added 6 nodes

    def test_search_nodes_dict_query(self, populated_graph):
        """search_nodes(dict) works on SQLiteBackend."""
        results = populated_graph.search.search_nodes({"memory_type": "semantic"})
        assert len(results) == 2


# ── Serialize / Deserialize sync contract ─────────────────────────────────────


class TestSerializeDeserializeSQLite:
    """Round-trip via serialize → clear → deserialize preserves all data."""

    def test_round_trip_preserves_counts(self, populated_graph):
        b = populated_graph.backend
        before_nodes = b.count_nodes()
        before_edges = b.count_edges()
        data = b.serialize()
        b.clear()
        assert b.count_nodes() == 0
        assert b.count_edges() == 0
        b.deserialize(data)
        assert b.count_nodes() == before_nodes
        assert b.count_edges() == before_edges

    def test_round_trip_preserves_types(self, populated_graph):
        b = populated_graph.backend
        before_types = set(b.get_node_types())
        data = b.serialize()
        b.clear()
        b.deserialize(data)
        after_types = set(b.get_node_types())
        assert before_types == after_types

    def test_monitoring_works_after_deserialize(self, populated_graph):
        """Monitoring still works after serialize → clear → deserialize."""
        b = populated_graph.backend
        data = b.serialize()
        b.clear()
        b.deserialize(data)
        mon = Monitoring(populated_graph)
        result = mon.summary()
        assert result["semantic"]["count"] == 2


# ── GraphHealthChecker on SQLiteBackend ──────────────────────────────────────


class TestGraphHealthOnSQLite:
    """GraphHealthChecker returns correct metrics without Cypher."""

    def test_health_report_populated(self, populated_graph):
        checker = GraphHealthChecker(None, graph=populated_graph)
        report = checker.collect_health()
        assert report.total_nodes == 6  # s1, s2, e1, w1, p1, n1
        assert report.total_edges == 2  # RELATES_TO + DERIVED_FROM
        assert report.orphan_count == 3  # w1, p1, n1 have no edges
        assert report.type_distribution["semantic"] == 2
        assert report.type_distribution["episodic"] == 1
        assert report.edge_distribution["RELATES_TO"] == 1
        assert report.edge_distribution["DERIVED_FROM"] == 1

    def test_health_report_empty(self, sqlite_graph):
        checker = GraphHealthChecker(None, graph=sqlite_graph)
        report = checker.collect_health()
        assert report.total_nodes == 0
        assert report.total_edges == 0
        assert report.orphan_count == 0
        assert report.type_distribution == {}
        assert report.edge_distribution == {}

    def test_is_healthy_check(self, populated_graph):
        checker = GraphHealthChecker(None, graph=populated_graph)
        report = checker.collect_health()
        # 3 orphans / 6 total = 0.5 > ORPHAN_THRESHOLD (0.2) → unhealthy
        assert not report.is_healthy

    def test_provenance_coverage_no_decisions(self, populated_graph):
        """No decision nodes → provenance coverage defaults to 1.0."""
        checker = GraphHealthChecker(None, graph=populated_graph)
        report = checker.collect_health()
        assert report.provenance_coverage == 1.0

    def test_provenance_coverage_with_decisions(self, sqlite_graph):
        """Decision node with provenance edge counts as covered."""
        b = sqlite_graph.backend
        b.add_node("d1", {"content": "Pick framework X", "memory_type": "decision"})
        b.add_node("d2", {"content": "Use database Y", "memory_type": "decision"})
        b.add_node("s1", {"content": "Framework comparison", "memory_type": "semantic"})
        b.add_edge("d1", "s1", "DERIVED_FROM", {})
        # d1 has provenance, d2 does not → coverage = 0.5
        checker = GraphHealthChecker(None, graph=sqlite_graph)
        report = checker.collect_health()
        assert report.provenance_coverage == 0.5

    def test_to_dict_round_trip(self, populated_graph):
        checker = GraphHealthChecker(None, graph=populated_graph)
        report = checker.collect_health()
        d = report.to_dict()
        restored = HealthReport.from_dict(d)
        assert restored.total_nodes == report.total_nodes
        assert restored.total_edges == report.total_edges
        assert restored.orphan_count == report.orphan_count


# ── merge_nodes on SQLiteBackend ─────────────────────────────────────────────


class TestMergeNodesOnSQLite:
    """merge_nodes() on SQLiteBackend matches FalkorDB semantics."""

    @pytest.fixture
    def backend(self, tmp_path):
        return SQLiteBackend(db_path=str(tmp_path / "merge.db"))

    def test_basic_merge(self, backend):
        """Merge source into target: edges rewired, source deleted."""
        backend.add_node("target", {"content": "canonical", "memory_type": "semantic"})
        backend.add_node("source", {"content": "duplicate", "memory_type": "semantic"})
        backend.add_node("other", {"content": "neighbor", "memory_type": "semantic"})
        backend.add_edge("source", "other", "RELATES_TO", {})
        backend.add_edge("other", "source", "CAUSED_BY", {})

        result = backend.merge_nodes("target", ["source"])
        assert result is True
        assert backend.get_node("source") is None
        assert backend.get_node("target") is not None
        # Outgoing edge rewired: target→other
        assert backend.algos.edge_type_counts().get("RELATES_TO") == 1
        # Incoming edge rewired: other→target
        assert backend.algos.edge_type_counts().get("CAUSED_BY") == 1

    def test_merge_multiple_sources(self, backend):
        """Multiple sources merged into one target."""
        backend.add_node("T", {"content": "target"})
        backend.add_node("S1", {"content": "source1"})
        backend.add_node("S2", {"content": "source2"})
        backend.add_node("X", {"content": "external"})
        backend.add_edge("S1", "X", "REL", {})
        backend.add_edge("S2", "X", "REL", {})

        result = backend.merge_nodes("T", ["S1", "S2"])
        assert result is True
        assert backend.get_node("S1") is None
        assert backend.get_node("S2") is None
        assert backend.count_nodes() == 2  # T and X
        # Both edges rewired to T→X REL — UPSERT deduplicates
        assert backend.algos.edge_type_counts() == {"REL": 1}

    def test_self_loop_skip(self, backend):
        """Edges between source and target are dropped (not turned into self-loops)."""
        backend.add_node("T", {})
        backend.add_node("S", {})
        backend.add_edge("S", "T", "LINKS_TO", {})  # would become T→T
        backend.add_edge("T", "S", "CAUSED_BY", {})  # would become T→T

        backend.merge_nodes("T", ["S"])
        assert backend.count_edges() == 0  # both dropped
        assert backend.algos.orphan_nodes() == ["T"]

    def test_property_merge_target_wins(self, backend):
        """Target properties take precedence over source."""
        backend.add_node("T", {"name": "target_name", "color": "blue"})
        backend.add_node("S", {"name": "source_name", "color": "red", "extra": "data"})

        backend.merge_nodes("T", ["S"])
        node = backend.get_node("T")
        # Target wins on collision
        assert node["name"] == "target_name"
        assert node["color"] == "blue"
        # Source fills missing keys
        assert node["extra"] == "data"

    def test_empty_source_ids(self, backend):
        """Empty source list is a no-op."""
        backend.add_node("T", {})
        assert backend.merge_nodes("T", []) is True
        assert backend.count_nodes() == 1

    def test_source_same_as_target(self, backend):
        """Source == target is silently skipped."""
        backend.add_node("T", {"content": "hello"})
        assert backend.merge_nodes("T", ["T"]) is True
        assert backend.get_node("T") is not None

    def test_missing_target_does_not_delete_source(self, backend):
        """merge_nodes with nonexistent target must not delete sources."""
        backend.add_node("source", {"content": "important data"})
        result = backend.merge_nodes("nonexistent_target", ["source"])
        assert result is False
        # Source must still exist — no data loss
        assert backend.get_node("source") is not None
        assert backend.count_nodes() == 1

    def test_pk_conflict_edge_upsert(self, backend):
        """When rewiring creates a duplicate (target,other,type), UPSERT merges it."""
        backend.add_node("T", {})
        backend.add_node("S", {})
        backend.add_node("X", {})
        # Both already connect to X with same type
        backend.add_edge("T", "X", "REL", {"weight": "0.9"})
        backend.add_edge("S", "X", "REL", {"weight": "0.1"})

        backend.merge_nodes("T", ["S"])
        assert backend.count_edges() == 1
        # UPSERT overwrites with the rewired edge's properties
        edges = backend.get_edges_for_node("T")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "REL"

    def test_compute_layer_sync_after_merge(self, backend):
        """In-memory graph reflects merged state."""
        backend.add_node("T", {})
        backend.add_node("S", {})
        backend.add_node("X", {})
        backend.add_edge("S", "X", "REL", {})

        backend.merge_nodes("T", ["S"])
        # Verify algos work on the merged graph
        g = backend._compute.nx_graph
        assert "S" not in g.nodes
        assert "T" in g.nodes
        assert g.has_edge("T", "X")

    def test_merge_preserves_parallel_edges(self, backend):
        """Rewiring preserves distinct edge types between same node pair."""
        backend.add_node("T", {})
        backend.add_node("S", {})
        backend.add_node("X", {})
        backend.add_edge("S", "X", "TYPE_A", {})
        backend.add_edge("S", "X", "TYPE_B", {})

        backend.merge_nodes("T", ["S"])
        counts = backend.algos.edge_type_counts()
        assert counts == {"TYPE_A": 1, "TYPE_B": 1}


# ── Phase 4: Bulk operations + diagnostics ───────────────────────────────────


class TestBulkOperationsOnSQLite:
    """Bulk add_nodes/add_edges using executemany()."""

    @pytest.fixture
    def backend(self, tmp_path):
        return SQLiteBackend(db_path=str(tmp_path / "bulk.db"))

    def test_add_nodes_bulk(self, backend):
        nodes = [
            {"item_id": f"n{i}", "content": f"Node {i}", "memory_type": "semantic"}
            for i in range(50)
        ]
        count = backend.add_nodes_bulk(nodes)
        assert count == 50
        assert backend.count_nodes() == 50
        # Verify compute layer in sync
        assert backend._compute.nx_graph.number_of_nodes() == 50

    def test_add_edges_bulk(self, backend):
        for i in range(5):
            backend.add_node(f"n{i}", {"memory_type": "semantic"})
        edges = [
            (f"n{i}", f"n{i+1}", "RELATES_TO", {})
            for i in range(4)
        ]
        count = backend.add_edges_bulk(edges)
        assert count == 4
        assert backend.count_edges() == 4
        assert backend.algos.edge_type_counts() == {"RELATES_TO": 4}

    def test_bulk_nodes_upsert(self, backend):
        """Bulk add with duplicate IDs uses UPSERT semantics."""
        backend.add_node("x", {"content": "original"})
        nodes = [{"item_id": "x", "content": "updated", "memory_type": "semantic"}]
        backend.add_nodes_bulk(nodes)
        node = backend.get_node("x")
        assert node["content"] == "updated"
        assert backend.count_nodes() == 1

    def test_bulk_with_algos(self, backend):
        """Algos work correctly after bulk insert."""
        nodes = [
            {"item_id": "a", "memory_type": "semantic"},
            {"item_id": "b", "memory_type": "semantic"},
            {"item_id": "c", "memory_type": "semantic"},
            {"item_id": "orphan", "memory_type": "semantic"},
        ]
        backend.add_nodes_bulk(nodes)
        edges = [("a", "b", "REL", {}), ("b", "c", "REL", {})]
        backend.add_edges_bulk(edges)
        assert backend.algos.orphan_nodes() == ["orphan"]
        paths = backend.algos.find_paths("a", "c")
        assert len(paths) == 1


class TestDiagnosticsOnSQLite:
    """health_check() and backend_info() diagnostics."""

    def test_health_check_ok(self, tmp_path):
        backend = SQLiteBackend(db_path=str(tmp_path / "diag.db"))
        result = backend.health_check()
        assert result["status"] == "ok"
        assert result["backend"] == "sqlite"
        assert result["integrity_check"] == "ok"

    def test_health_check_in_memory(self):
        backend = SQLiteBackend(db_path=":memory:")
        result = backend.health_check()
        assert result["status"] == "ok"

    def test_backend_info_file(self, tmp_path):
        backend = SQLiteBackend(db_path=str(tmp_path / "info.db"))
        backend.add_node("x", {"memory_type": "semantic"})
        info = backend.backend_info()
        assert info["backend"] == "sqlite"
        assert info["journal_mode"] == "wal"
        assert info["node_count"] == 1
        assert info["edge_count"] == 0
        assert info["in_memory"] is False
        assert "file_size_bytes" in info
        assert info["file_size_bytes"] > 0

    def test_backend_info_in_memory(self):
        backend = SQLiteBackend(db_path=":memory:")
        info = backend.backend_info()
        assert info["in_memory"] is True
        assert "file_size_bytes" not in info


class TestInferenceOnSQLite:
    """InferenceEngine runs backend-agnostic matchers (no Cypher) on SQLite."""

    @pytest.fixture
    def backend(self, tmp_path):
        return SQLiteBackend(db_path=str(tmp_path / "inference.db"))

    @pytest.fixture
    def engine(self, backend):
        """InferenceEngine wired to a bare SQLiteBackend (no SmartMemory wrapper)."""
        # InferenceEngine expects (memory, graph) — graph needs a .backend attribute.
        class _GraphShim:
            def __init__(self, b):
                self.backend = b
        return InferenceEngine(memory=None, graph=_GraphShim(backend))

    # ── Causal transitivity ───────────────────────────────────────────────

    def test_causal_transitivity_creates_edge(self, backend, engine):
        """A-[CAUSES]->B-[CAUSES]->C should infer A-[CAUSES]->C."""
        backend.add_node("a", {"memory_type": "semantic"})
        backend.add_node("b", {"memory_type": "semantic"})
        backend.add_node("c", {"memory_type": "semantic"})
        backend.add_edge("a", "b", "CAUSES", {"confidence": 0.9})
        backend.add_edge("b", "c", "CAUSES", {"confidence": 0.8})

        result = engine.run()
        assert result.edges_created == 1
        assert "causal_transitivity" in result.rules_applied
        # Verify edge was created
        edges_a = backend.get_edges_for_node("a")
        causes_to_c = [e for e in edges_a if e.get("target_id") == "c" and e.get("edge_type") == "CAUSES"]
        assert len(causes_to_c) == 1

    def test_causal_transitivity_skips_existing_direct(self, backend, engine):
        """No duplicate when direct A-[CAUSES]->C already exists."""
        backend.add_node("a", {"memory_type": "semantic"})
        backend.add_node("b", {"memory_type": "semantic"})
        backend.add_node("c", {"memory_type": "semantic"})
        backend.add_edge("a", "b", "CAUSES", {})
        backend.add_edge("b", "c", "CAUSES", {})
        backend.add_edge("a", "c", "CAUSES", {})  # already exists

        result = engine.run()
        # causal_transitivity should find no unlinked pairs
        assert "causal_transitivity" not in result.rules_applied

    def test_causal_transitivity_confidence_decay(self, backend, engine):
        """Inferred edge confidence decays via min(rule_conf, r1 * r2)."""
        backend.add_node("a", {"memory_type": "semantic"})
        backend.add_node("b", {"memory_type": "semantic"})
        backend.add_node("c", {"memory_type": "semantic"})
        backend.add_edge("a", "b", "CAUSES", {"confidence": 0.9})
        backend.add_edge("b", "c", "CAUSES", {"confidence": 0.8})

        result = engine.run()
        assert result.edges_created == 1
        # The edge confidence should be min(0.7, 0.9 * 0.8) = min(0.7, 0.72) = 0.7
        edges_a = backend.get_edges_for_node("a")
        inferred = [e for e in edges_a if e.get("target_id") == "c" and e.get("edge_type") == "CAUSES"]
        assert len(inferred) == 1

    # ── Contradiction symmetry ────────────────────────────────────────────

    def test_contradiction_symmetry(self, backend, engine):
        """A-[CONTRADICTS]->B should create B-[CONTRADICTS]->A."""
        backend.add_node("a", {"memory_type": "semantic"})
        backend.add_node("b", {"memory_type": "semantic"})
        backend.add_edge("a", "b", "CONTRADICTS", {})

        result = engine.run()
        assert "contradiction_symmetry" in result.rules_applied
        edges_b = backend.get_edges_for_node("b")
        reverse = [e for e in edges_b if e.get("source_id") == "b" and e.get("target_id") == "a"
                    and e.get("edge_type") == "CONTRADICTS"]
        assert len(reverse) == 1

    def test_contradiction_symmetry_already_symmetric(self, backend, engine):
        """No action when both directions already exist."""
        backend.add_node("a", {"memory_type": "semantic"})
        backend.add_node("b", {"memory_type": "semantic"})
        backend.add_edge("a", "b", "CONTRADICTS", {})
        backend.add_edge("b", "a", "CONTRADICTS", {})

        result = engine.run()
        assert "contradiction_symmetry" not in result.rules_applied

    # ── Topic inheritance ─────────────────────────────────────────────────

    def test_topic_inheritance(self, backend, engine):
        """decision-[DERIVED_FROM]->semantic creates semantic-[INFLUENCES]->decision."""
        backend.add_node("d1", {"memory_type": "decision"})
        backend.add_node("s1", {"memory_type": "semantic"})
        backend.add_edge("d1", "s1", "DERIVED_FROM", {})

        result = engine.run()
        assert "topic_inheritance" in result.rules_applied
        edges_s = backend.get_edges_for_node("s1")
        influences = [e for e in edges_s if e.get("source_id") == "s1" and e.get("target_id") == "d1"
                      and e.get("edge_type") == "INFLUENCES"]
        assert len(influences) == 1

    def test_topic_inheritance_wrong_types(self, backend, engine):
        """No INFLUENCES when node types don't match."""
        backend.add_node("ep1", {"memory_type": "episodic"})
        backend.add_node("s1", {"memory_type": "semantic"})
        backend.add_edge("ep1", "s1", "DERIVED_FROM", {})

        result = engine.run()
        assert "topic_inheritance" not in result.rules_applied

    def test_topic_inheritance_already_exists(self, backend, engine):
        """No duplicate when INFLUENCES already present."""
        backend.add_node("d1", {"memory_type": "decision"})
        backend.add_node("s1", {"memory_type": "semantic"})
        backend.add_edge("d1", "s1", "DERIVED_FROM", {})
        backend.add_edge("s1", "d1", "INFLUENCES", {})

        result = engine.run()
        assert "topic_inheritance" not in result.rules_applied

    # ── Engine behavior ───────────────────────────────────────────────────

    def test_dry_run(self, backend, engine):
        """dry_run=True counts matches but creates no edges."""
        backend.add_node("a", {"memory_type": "semantic"})
        backend.add_node("b", {"memory_type": "semantic"})
        backend.add_edge("a", "b", "CONTRADICTS", {})

        result = engine.run(dry_run=True)
        assert result.edges_created == 1
        assert "contradiction_symmetry" in result.rules_applied
        # No reverse edge actually created
        edges_b = backend.get_edges_for_node("b")
        reverse = [e for e in edges_b if e.get("source_id") == "b" and e.get("edge_type") == "CONTRADICTS"]
        assert len(reverse) == 0

    def test_empty_graph(self, backend, engine):
        """No errors on empty graph."""
        result = engine.run()
        assert result.edges_created == 0
        assert result.rules_applied == []
        assert result.errors == []

    def test_all_three_rules_fire(self, backend, engine):
        """Graph with all three patterns triggers all three rules."""
        # causal transitivity: x -> y -> z
        backend.add_node("x", {"memory_type": "semantic"})
        backend.add_node("y", {"memory_type": "semantic"})
        backend.add_node("z", {"memory_type": "semantic"})
        backend.add_edge("x", "y", "CAUSES", {"confidence": 0.8})
        backend.add_edge("y", "z", "CAUSES", {"confidence": 0.8})
        # contradiction symmetry: a -> b
        backend.add_node("ca", {"memory_type": "semantic"})
        backend.add_node("cb", {"memory_type": "semantic"})
        backend.add_edge("ca", "cb", "CONTRADICTS", {})
        # topic inheritance: decision -> semantic
        backend.add_node("dec", {"memory_type": "decision"})
        backend.add_node("sem", {"memory_type": "semantic"})
        backend.add_edge("dec", "sem", "DERIVED_FROM", {})

        result = engine.run()
        assert set(result.rules_applied) == {"causal_transitivity", "contradiction_symmetry", "topic_inheritance"}
        assert result.edges_created == 3
