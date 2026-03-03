"""Integration tests for DIST-LITE-PARITY-1 — verify higher-level subsystems work on SQLiteBackend.

Covers:
- Monitoring: summary(), orphaned_notes(), reflect(), summarize() on SQLite graph
- Analytics: find_similar_items() degrades gracefully, detect_concept_drift() works
- FTS5: UsearchVectorBackend.search_by_text() returns results via SQLite FTS5
- Search fallback: SmartGraphSearch falls back to contains/keyword on SQLite

Requires: no Docker (all SQLite / in-memory)
"""
import pytest

from smartmemory.graph.backends.sqlite import SQLiteBackend
from smartmemory.graph.smartgraph import SmartGraph
from smartmemory.memory.pipeline.stages.monitoring import Monitoring
from smartmemory.memory.pipeline.stages.analytics import MemoryAnalytics


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
