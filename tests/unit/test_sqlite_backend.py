"""Comprehensive tests for SQLiteBackend (DIST-LITE-1 + DIST-LITE-4). No Docker required.

Covers:
- All 10 abstract methods
- Edge cases: None item_id, update-preserves-edges, missing nodes, empty results
- Thread safety (concurrent writes)
- WAL mode (file-based only)
- Properties: complex JSON, special chars
- Cascade behavior: remove_node removes edges
- Bulk operations (inherited from ABC)
- Persistence: data survives across backend instances
- serialize/deserialize: full round-trip including edges
- Foreign key enforcement
"""

import threading
import pytest

from smartmemory.graph.backends.sqlite import SQLiteBackend


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def backend(tmp_path):
    """File-based backend for most tests (WAL, persistence)."""
    return SQLiteBackend(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def mem_backend():
    """In-memory backend for fast isolation tests."""
    return SQLiteBackend(db_path=":memory:")


# ── Core round-trips ─────────────────────────────────────────────────────────


def test_add_and_get_node(backend):
    """add_node → get_node round-trip with memory_type in properties."""
    backend.add_node("item1", {"content": "hello", "memory_type": "semantic"})
    result = backend.get_node("item1")
    assert result is not None
    assert result["item_id"] == "item1"
    assert result["memory_type"] == "semantic"
    assert result["content"] == "hello"


def test_add_node_memory_type_param_takes_priority(backend):
    """memory_type= parameter takes priority over properties dict."""
    backend.add_node("n1", {"content": "x", "memory_type": "episodic"}, memory_type="semantic")
    result = backend.get_node("n1")
    assert result["memory_type"] == "semantic"


def test_add_node_none_item_id_generates_uuid(backend):
    """add_node(None, ...) generates a UUID and returns it in the result."""
    result = backend.add_node(None, {"content": "auto-id", "memory_type": "working"})
    assert "item_id" in result
    item_id = result["item_id"]
    assert len(item_id) == 36  # UUID4 format
    assert backend.get_node(item_id) is not None


def test_add_node_update_preserves_edges(backend):
    """Updating an existing node via add_node MUST NOT cascade-delete its edges (upsert bug fix)."""
    backend.add_node("a", {"content": "original", "memory_type": "semantic"})
    backend.add_node("b", {"content": "B", "memory_type": "semantic"})
    backend.add_edge("a", "b", "linked", {})
    # Update node a — this should NOT remove the edge
    backend.add_node("a", {"content": "updated", "memory_type": "semantic"})
    assert backend.get_node("a")["content"] == "updated"
    neighbors = backend.get_neighbors("a")
    assert any(n["item_id"] == "b" for n in neighbors), "Edge was lost after node update — cascade bug!"


def test_get_node_nonexistent(backend):
    """get_node on unknown item_id returns None."""
    assert backend.get_node("nonexistent-xyz") is None


def test_add_edge_and_get_neighbors(backend):
    """add_edge → get_neighbors returns the target node."""
    backend.add_node("a", {"content": "A", "memory_type": "working"})
    backend.add_node("b", {"content": "B", "memory_type": "working"})
    backend.add_edge("a", "b", "relates_to", {})
    neighbors = backend.get_neighbors("a")
    ids = [n["item_id"] for n in neighbors]
    assert "b" in ids


def test_get_neighbors_empty(backend):
    """get_neighbors on a node with no edges returns empty list."""
    backend.add_node("lone", {"content": "lone", "memory_type": "working"})
    assert backend.get_neighbors("lone") == []


def test_get_neighbors_filtered_by_edge_type(backend):
    """get_neighbors(edge_type=...) returns only matching neighbors."""
    backend.add_node("hub", {"content": "hub", "memory_type": "working"})
    backend.add_node("spoke_a", {"content": "A", "memory_type": "working"})
    backend.add_node("spoke_b", {"content": "B", "memory_type": "working"})
    backend.add_edge("hub", "spoke_a", "type_a", {})
    backend.add_edge("hub", "spoke_b", "type_b", {})
    results = backend.get_neighbors("hub", edge_type="type_a")
    ids = [n["item_id"] for n in results]
    assert "spoke_a" in ids
    assert "spoke_b" not in ids


# ── Remove operations ────────────────────────────────────────────────────────


def test_remove_node_returns_true(backend):
    """remove_node returns True for an existing node."""
    backend.add_node("x", {"content": "X", "memory_type": "episodic"})
    assert backend.remove_node("x") is True
    assert backend.get_node("x") is None


def test_remove_node_nonexistent_returns_false(backend):
    """remove_node on unknown item_id returns False."""
    assert backend.remove_node("does-not-exist") is False


def test_remove_node_cascades_edges(backend):
    """Removing a node removes all edges it participates in."""
    backend.add_node("p", {"content": "P", "memory_type": "working"})
    backend.add_node("q", {"content": "Q", "memory_type": "working"})
    backend.add_edge("p", "q", "linked", {})
    backend.remove_node("p")
    # Edge from p→q should be gone; q should have no incoming neighbors from p
    assert backend.get_neighbors("q") == []  # q has no outgoing edges
    # And trying to get p's neighbors should return [] (node is gone)
    assert backend.get_neighbors("p") == []


def test_remove_edge_returns_true(backend):
    """remove_edge returns True when edge exists."""
    backend.add_node("p", {"content": "P", "memory_type": "working"})
    backend.add_node("q", {"content": "Q", "memory_type": "working"})
    backend.add_edge("p", "q", "linked", {})
    assert backend.remove_edge("p", "q", "linked") is True
    assert backend.get_neighbors("p") == []


def test_remove_edge_nonexistent_returns_false(backend):
    """remove_edge on unknown edge returns False."""
    assert backend.remove_edge("x", "y", "no_such_edge") is False


def test_remove_edge_without_type(backend):
    """remove_edge without edge_type removes all edges between the pair."""
    backend.add_node("a", {"content": "A", "memory_type": "working"})
    backend.add_node("b", {"content": "B", "memory_type": "working"})
    backend.add_edge("a", "b", "type1", {})
    backend.add_edge("a", "b", "type2", {})
    result = backend.remove_edge("a", "b")  # no edge_type — removes both
    assert result is True
    assert backend.get_neighbors("a") == []


# ── Search ───────────────────────────────────────────────────────────────────


def test_search_nodes_by_memory_type(backend):
    """search_nodes({'memory_type': 'semantic'}) returns only semantic nodes."""
    backend.add_node("s1", {"content": "S1", "memory_type": "semantic"})
    backend.add_node("e1", {"content": "E1", "memory_type": "episodic"})
    results = backend.search_nodes({"memory_type": "semantic"})
    ids = [r["item_id"] for r in results]
    assert "s1" in ids
    assert "e1" not in ids


def test_search_nodes_content_match(backend):
    """search_nodes({'content': 'Atlas'}) matches nodes with that substring in properties."""
    backend.add_node("n1", {"content": "Project Atlas is running", "memory_type": "semantic"})
    backend.add_node("n2", {"content": "Nothing here", "memory_type": "semantic"})
    results = backend.search_nodes({"content": "Atlas"})
    ids = [r["item_id"] for r in results]
    assert "n1" in ids
    assert "n2" not in ids


def test_search_nodes_combined_filters(backend):
    """search_nodes with memory_type AND content filters both applied."""
    backend.add_node("match", {"content": "Atlas project", "memory_type": "semantic"})
    backend.add_node("wrong_type", {"content": "Atlas project", "memory_type": "episodic"})
    backend.add_node("wrong_content", {"content": "Nothing", "memory_type": "semantic"})
    results = backend.search_nodes({"memory_type": "semantic", "content": "Atlas"})
    ids = [r["item_id"] for r in results]
    assert "match" in ids
    assert "wrong_type" not in ids
    assert "wrong_content" not in ids


def test_search_nodes_empty_query_returns_all(backend):
    """search_nodes({}) returns all nodes."""
    backend.add_node("a", {"content": "A", "memory_type": "working"})
    backend.add_node("b", {"content": "B", "memory_type": "semantic"})
    results = backend.search_nodes({})
    ids = [r["item_id"] for r in results]
    assert "a" in ids
    assert "b" in ids


# ── Serialize / deserialize ──────────────────────────────────────────────────


def test_serialize_clear_deserialize_preserves_nodes(backend):
    """serialize → clear → deserialize: nodes restored."""
    backend.add_node("persist", {"content": "keep me", "memory_type": "semantic"})
    data = backend.serialize()
    backend.clear()
    assert backend.get_node("persist") is None
    backend.deserialize(data)
    result = backend.get_node("persist")
    assert result is not None
    assert result["item_id"] == "persist"


def test_serialize_clear_deserialize_preserves_edges(backend):
    """serialize → clear → deserialize: edges restored."""
    backend.add_node("x", {"content": "X", "memory_type": "working"})
    backend.add_node("y", {"content": "Y", "memory_type": "working"})
    backend.add_edge("x", "y", "linked", {"weight": 0.9})
    data = backend.serialize()
    backend.clear()
    backend.deserialize(data)
    neighbors = backend.get_neighbors("x")
    assert any(n["item_id"] == "y" for n in neighbors), "Edges not restored after deserialize"


def test_deserialize_empty_data(backend):
    """deserialize({}) does not crash."""
    backend.deserialize({})
    backend.deserialize({"nodes": [], "edges": []})


def test_clear_idempotent(backend):
    """clear() called twice does not raise."""
    backend.add_node("a", {"content": "A", "memory_type": "working"})
    backend.clear()
    backend.clear()  # second call on empty DB must not raise


# ── Properties ───────────────────────────────────────────────────────────────


def test_properties_complex_json(backend):
    """Properties with nested dicts and lists round-trip correctly."""
    props = {
        "content": "test",
        "memory_type": "semantic",
        "tags": ["alpha", "beta"],
        "meta": {"confidence": 0.95, "source": "test"},
    }
    backend.add_node("complex", props)
    result = backend.get_node("complex")
    assert result["tags"] == ["alpha", "beta"]
    assert result["meta"]["confidence"] == 0.95


def test_properties_special_characters(backend):
    """Properties with quotes, newlines, unicode round-trip correctly."""
    props = {
        "content": 'He said "hello"\nNew line here\t tab',
        "memory_type": "semantic",
        "emoji": "\U0001f9e0",
    }
    backend.add_node("special", props)
    result = backend.get_node("special")
    assert result["content"] == props["content"]
    assert result["emoji"] == "\U0001f9e0"


# ── valid_time handling ──────────────────────────────────────────────────────


def test_valid_time_tuple_stored_and_retrieved(backend):
    """valid_time as a tuple has its first element stored as valid_from."""
    backend.add_node("t1", {"content": "timed", "memory_type": "episodic"}, valid_time=("2026-01-01", "2026-12-31"))
    # valid_from should be "2026-01-01" (first element)
    row = backend._conn.execute("SELECT valid_from FROM nodes WHERE item_id='t1'").fetchone()
    assert row[0] == "2026-01-01"


# ── Cascade and FK enforcement ───────────────────────────────────────────────


def test_add_edge_fk_violation_raises(backend):
    """add_edge with a non-existent source_id raises IntegrityError (FK ON)."""
    import sqlite3 as _sqlite3

    backend.add_node("exists", {"content": "E", "memory_type": "working"})
    with pytest.raises(_sqlite3.IntegrityError):
        backend.add_edge("nonexistent", "exists", "linked", {})


# ── Bulk operations (inherited from ABC) ─────────────────────────────────────


def test_add_nodes_bulk(backend):
    """add_nodes_bulk inserts all nodes correctly."""
    nodes = [
        {"item_id": "bulk1", "content": "B1", "memory_type": "working"},
        {"item_id": "bulk2", "content": "B2", "memory_type": "semantic"},
    ]
    backend.add_nodes_bulk(nodes)
    assert backend.get_node("bulk1") is not None
    assert backend.get_node("bulk2") is not None


# ── WAL and persistence ──────────────────────────────────────────────────────


def test_wal_mode_file_backend(backend):
    """File-based backend uses WAL journal mode."""
    assert backend._journal_mode == "wal"


def test_memory_backend_journal_mode(mem_backend):
    """In-memory backend stores whatever journal_mode SQLite returns (not wal)."""
    # SQLite returns "memory" for in-memory databases regardless of what we request
    assert mem_backend._is_memory_db is True
    # No assertion on journal_mode value — just confirm it doesn't crash


def test_persistence_across_instances(tmp_path):
    """Data written to a file-based backend survives reopening."""
    db_path = str(tmp_path / "persistent.db")
    b1 = SQLiteBackend(db_path=db_path)
    b1.add_node("durable", {"content": "still here", "memory_type": "semantic"})
    del b1
    b2 = SQLiteBackend(db_path=db_path)
    result = b2.get_node("durable")
    assert result is not None
    assert result["content"] == "still here"


# ── Thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_writes_no_corruption(tmp_path):
    """Concurrent add_node calls from multiple threads produce consistent state."""
    db_path = str(tmp_path / "threaded.db")
    backend = SQLiteBackend(db_path=db_path)
    errors = []

    def insert(i):
        try:
            backend.add_node(f"thread-{i}", {"content": f"item {i}", "memory_type": "working"})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=insert, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    results = backend.search_nodes({"memory_type": "working"})
    assert len(results) == 20


# ── execute_query guard ──────────────────────────────────────────────────────


def test_execute_query_raises_not_implemented(backend):
    """execute_query raises NotImplementedError with helpful message."""
    with pytest.raises(NotImplementedError, match="FalkorDB"):
        backend.execute_query("MATCH (n) RETURN n")


# ── valid_to storage and upsert ──────────────────────────────────────────────


def test_valid_to_stored_and_retrieved(backend):
    """valid_time tuple's second element is stored as valid_to."""
    backend.add_node(
        "timed2",
        {"content": "expiring", "memory_type": "episodic"},
        valid_time=("2026-01-01", "2026-12-31"),
    )
    row = backend._conn.execute("SELECT valid_from, valid_to FROM nodes WHERE item_id='timed2'").fetchone()
    assert row[0] == "2026-01-01"
    assert row[1] == "2026-12-31"


def test_add_node_upsert_preserves_valid_to(backend):
    """Updating a node via add_node does not silently wipe its valid_to."""
    backend.add_node(
        "expiring",
        {"content": "original", "memory_type": "semantic"},
        valid_time=("2026-01-01", "2026-12-31"),
    )
    # Update content only — valid_to must survive
    backend.add_node(
        "expiring",
        {"content": "updated", "memory_type": "semantic"},
        valid_time=("2026-01-01", "2026-12-31"),
    )
    row = backend._conn.execute("SELECT valid_to FROM nodes WHERE item_id='expiring'").fetchone()
    assert row[0] == "2026-12-31", "valid_to was wiped by upsert"


def test_deserialize_bad_edge_skipped_not_rolled_back(backend):
    """If deserialize encounters an edge with a non-existent node, nodes are still committed."""
    data = {
        "nodes": [{"item_id": "good_node", "memory_type": "working", "properties": {}}],
        "edges": [{"source_id": "nonexistent", "target_id": "good_node", "edge_type": "broken", "properties": {}}],
    }
    backend.deserialize(data)  # must not raise
    # The good node must be committed even though the edge failed
    assert backend.get_node("good_node") is not None


# ── Fix 1: _row_to_node returns temporal fields ──────────────────────────────


def test_get_node_returns_temporal_fields(backend):
    """get_node() includes valid_from, valid_to, created_at in the returned dict."""
    backend.add_node(
        "temporal",
        {"content": "timed", "memory_type": "episodic"},
        valid_time=("2026-01-01", "2026-12-31"),
    )
    result = backend.get_node("temporal")
    assert "valid_from" in result
    assert "valid_to" in result
    assert "created_at" in result
    assert result["valid_from"] == "2026-01-01"
    assert result["valid_to"] == "2026-12-31"


# ── Fix 3: search_nodes content filter targets value, not JSON key names ─────


def test_search_nodes_content_filter_no_false_positives(backend):
    """search_nodes content filter targets the 'content' field value, not JSON key names."""
    backend.add_node("fp", {"content": "", "content_type": "text", "memory_type": "semantic"})
    # Searching for "content" should NOT match just because the key "content_type" exists
    results = backend.search_nodes({"content": "content"})
    ids = [r["item_id"] for r in results]
    assert "fp" not in ids, "False positive: matched JSON key name instead of content value"


# ── Fix 4: add_edge upsert preserves existing memory_type when None passed ───


def test_add_edge_upsert_preserves_existing_memory_type(backend):
    """Re-adding an edge with memory_type=None does not wipe the stored memory_type."""
    backend.add_node("a", {"content": "A", "memory_type": "working"})
    backend.add_node("b", {"content": "B", "memory_type": "working"})
    backend.add_edge("a", "b", "linked", {}, memory_type="semantic")
    # Re-add same edge with memory_type=None — should preserve "semantic"
    backend.add_edge("a", "b", "linked", {"extra": "data"}, memory_type=None)
    row = backend._conn.execute(
        "SELECT memory_type FROM edges WHERE source_id='a' AND target_id='b' AND edge_type='linked'"
    ).fetchone()
    assert row[0] == "semantic", "memory_type was wiped by edge re-add with None"


# ── Fix 2: close() and __del__ ───────────────────────────────────────────────


def test_close_does_not_raise(backend):
    """close() can be called without raising."""
    backend.close()


def test_del_does_not_raise(tmp_path):
    """__del__ does not raise even after connection is already closed."""
    b = SQLiteBackend(db_path=str(tmp_path / "del_test.db"))
    b.close()
    del b  # must not raise


def test_get_neighbors_is_bidirectional(backend):
    """get_neighbors returns neighbors in BOTH directions, matching FalkorDB semantics.

    FalkorDB uses undirected MATCH (n)-[r]-(m) which traverses both outgoing and incoming
    edges. SQLiteBackend implements this via UNION of both edge legs. Callers that assume
    directed-only behavior will see both ends of any edge touching the query node.
    """
    backend.add_node("source", {"content": "S", "memory_type": "working"})
    backend.add_node("target", {"content": "T", "memory_type": "working"})
    backend.add_edge("source", "target", "points_to", {})

    # Outgoing from source: target must appear
    outgoing = backend.get_neighbors("source")
    assert any(n["item_id"] == "target" for n in outgoing)

    # From target's perspective: source is an incoming neighbor and MUST appear
    incoming = backend.get_neighbors("target")
    assert any(n["item_id"] == "source" for n in incoming), (
        "get_neighbors must return incoming neighbors to match FalkorDB undirected semantics"
    )


def test_deserialize_edge_preserves_existing_memory_type(backend):
    """deserialize onto a non-empty graph preserves existing edge memory_type when serialized value is None."""
    backend.add_node("a", {"content": "A", "memory_type": "working"})
    backend.add_node("b", {"content": "B", "memory_type": "working"})
    # Add edge with explicit memory_type
    backend.add_edge("a", "b", "linked", {}, memory_type="semantic")
    # Serialize — memory_type is captured in the snapshot
    data = backend.serialize()
    # Manually null out memory_type in the serialized edge (simulates old export format)
    for edge in data["edges"]:
        edge["memory_type"] = None
    # Deserialize back — existing memory_type should be preserved via COALESCE
    backend.deserialize(data)
    row = backend._conn.execute(
        "SELECT memory_type FROM edges WHERE source_id='a' AND target_id='b' AND edge_type='linked'"
    ).fetchone()
    assert row[0] == "semantic", "deserialize wiped memory_type via INSERT OR REPLACE"


# ── DIST-LITE-4: new read methods ────────────────────────────────────────────


class TestGetAllEdges:
    def test_returns_all_edges(self, mem_backend):
        mem_backend.add_node("a", {"memory_type": "semantic"})
        mem_backend.add_node("b", {"memory_type": "semantic"})
        mem_backend.add_node("c", {"memory_type": "semantic"})
        mem_backend.add_edge("a", "b", "relates_to", {})
        mem_backend.add_edge("b", "c", "links_to", {})
        edges = mem_backend.get_all_edges()
        assert len(edges) == 2
        types = {e["edge_type"] for e in edges}
        assert types == {"relates_to", "links_to"}

    def test_edge_dict_has_all_eight_keys(self, mem_backend):
        mem_backend.add_node("x", {"memory_type": "semantic"})
        mem_backend.add_node("y", {"memory_type": "semantic"})
        mem_backend.add_edge("x", "y", "rel", {"weight": 0.9})
        edges = mem_backend.get_all_edges()
        assert len(edges) == 1
        e = edges[0]
        for key in (
            "source_id",
            "target_id",
            "edge_type",
            "memory_type",
            "valid_from",
            "valid_to",
            "created_at",
            "properties",
        ):
            assert key in e, f"missing key: {key}"
        assert e["properties"]["weight"] == 0.9

    def test_empty_graph_returns_empty_list(self, mem_backend):
        assert mem_backend.get_all_edges() == []


class TestGetEdgesForNode:
    def test_returns_edges_for_source_and_target(self, mem_backend):
        mem_backend.add_node("a", {"memory_type": "semantic"})
        mem_backend.add_node("b", {"memory_type": "semantic"})
        mem_backend.add_node("c", {"memory_type": "semantic"})
        mem_backend.add_edge("a", "b", "rel", {})
        mem_backend.add_edge("b", "c", "rel", {})
        # B is source of one edge and target of another
        edges = mem_backend.get_edges_for_node("b")
        assert len(edges) == 2

    def test_node_with_no_edges_returns_empty(self, mem_backend):
        mem_backend.add_node("lone", {"memory_type": "semantic"})
        assert mem_backend.get_edges_for_node("lone") == []

    def test_nonexistent_node_returns_empty(self, mem_backend):
        assert mem_backend.get_edges_for_node("does-not-exist") == []


class TestGetCounts:
    def test_empty_graph(self, mem_backend):
        counts = mem_backend.get_counts()
        assert counts == {"node_count": 0, "edge_count": 0}

    def test_counts_after_ingest(self, mem_backend):
        mem_backend.add_node("a", {"memory_type": "semantic"})
        mem_backend.add_node("b", {"memory_type": "semantic"})
        mem_backend.add_edge("a", "b", "rel", {})
        counts = mem_backend.get_counts()
        assert counts["node_count"] == 2
        assert counts["edge_count"] == 1


# ── DIST-LITE-DEGRADE-1a: direction parameter ────────────────────────────────


class TestGetNeighborsDirection:
    """Tests for get_neighbors(direction=...) — DIST-LITE-DEGRADE-1a."""

    @pytest.fixture(autouse=True)
    def setup_triangle(self, mem_backend):
        """Create a→b→c graph for directional queries."""
        self.backend = mem_backend
        mem_backend.add_node("a", {"content": "A", "memory_type": "working"})
        mem_backend.add_node("b", {"content": "B", "memory_type": "working"})
        mem_backend.add_node("c", {"content": "C", "memory_type": "working"})
        mem_backend.add_edge("a", "b", "LINKS", {})
        mem_backend.add_edge("b", "c", "LINKS", {})

    def test_outgoing_returns_only_targets(self):
        # a→b: outgoing from a should be [b]
        result = self.backend.get_neighbors("a", direction="outgoing")
        ids = [n["item_id"] for n in result]
        assert ids == ["b"]

    def test_incoming_returns_only_sources(self):
        # a→b: incoming to b should be [a]
        result = self.backend.get_neighbors("b", direction="incoming")
        ids = [n["item_id"] for n in result]
        assert "a" in ids
        assert "c" not in ids

    def test_both_returns_union(self):
        # b has outgoing to c and incoming from a
        result = self.backend.get_neighbors("b", direction="both")
        ids = {n["item_id"] for n in result}
        assert ids == {"a", "c"}

    def test_outgoing_with_edge_type_filter(self):
        self.backend.add_node("d", {"content": "D", "memory_type": "working"})
        self.backend.add_edge("a", "d", "OTHER", {})
        result = self.backend.get_neighbors("a", edge_type="LINKS", direction="outgoing")
        ids = [n["item_id"] for n in result]
        assert "b" in ids
        assert "d" not in ids

    def test_outgoing_empty_when_no_outgoing_edges(self):
        # c has no outgoing edges
        result = self.backend.get_neighbors("c", direction="outgoing")
        assert result == []

    def test_incoming_empty_when_no_incoming_edges(self):
        # a has no incoming edges
        result = self.backend.get_neighbors("a", direction="incoming")
        assert result == []

    def test_default_direction_is_both(self):
        """Omitting direction behaves the same as direction='both'."""
        explicit = self.backend.get_neighbors("b", direction="both")
        default = self.backend.get_neighbors("b")
        assert {n["item_id"] for n in explicit} == {n["item_id"] for n in default}
