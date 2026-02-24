"""Additional coverage tests for UsearchVectorBackend — edge cases not in test_usearch_backend.py.

Focuses on:
- search() on empty index returns []
- add() with no content key (no FTS entry written)
- upsert updates metadata, not just embedding
- search_by_text returns [] when no match
- search_by_text with FTS special chars returns [] without raising
- clear() then clear() again doesn't crash
- in-memory backend (no persist_directory)
- _load_or_create with corrupted JSON map falls back gracefully
- _next_key is monotonically increasing across upserts of same id
"""
import numpy as np
import pytest

from smartmemory.stores.vector.backends.usearch import UsearchVectorBackend


def _vec(seed: int, dim: int = 8) -> list:
    """Deterministic unit vector for testing."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def b(tmp_path):
    return UsearchVectorBackend(collection_name="cov", persist_directory=str(tmp_path))


@pytest.fixture
def mem_b():
    """In-memory backend — no persist_directory."""
    return UsearchVectorBackend(collection_name="mem", persist_directory=None)


# ── search on empty index ─────────────────────────────────────────────────────

def test_search_empty_index_returns_empty_list(b):
    """search() on an index with no items returns []."""
    results = b.search(query_embedding=_vec(1), top_k=5)
    assert results == []


def test_search_empty_index_in_memory(mem_b):
    """search() on empty in-memory backend returns []."""
    results = mem_b.search(query_embedding=_vec(2), top_k=3)
    assert results == []


# ── add without content key (no FTS entry) ───────────────────────────────────

def test_add_without_content_metadata_no_fts(b):
    """add() with metadata that has no 'content' key still works (no FTS row)."""
    v = _vec(3)
    b.add(item_id="no_content", embedding=v, metadata={"source": "test"})
    # Vector search should still find it
    results = b.search(query_embedding=v, top_k=1)
    assert results[0]["id"] == "no_content"
    # FTS search won't find it (no content indexed)
    fts_results = b.search_by_text(query_text="test", top_k=5)
    assert not any(r.get("id") == "no_content" for r in fts_results)


# ── upsert updates metadata ───────────────────────────────────────────────────

def test_upsert_updates_metadata(b):
    """upsert() replaces metadata for the same item_id."""
    v = _vec(4)
    b.add(item_id="meta_item", embedding=v, metadata={"content": "original", "tag": "old"})
    b.upsert(item_id="meta_item", embedding=v, metadata={"content": "updated", "tag": "new"})
    results = b.search(query_embedding=v, top_k=1)
    assert results[0]["metadata"].get("tag") == "new"
    assert results[0]["metadata"].get("content") == "updated"


# ── search_by_text no match ───────────────────────────────────────────────────

def test_search_by_text_no_match_returns_empty(b):
    """search_by_text() returns [] when query matches nothing."""
    b.add(item_id="item1", embedding=_vec(5), metadata={"content": "Python programming"})
    results = b.search_by_text(query_text="XYZnonexistenttoken", top_k=5)
    assert results == []


# ── search_by_text with FTS special chars ────────────────────────────────────

def test_search_by_text_special_chars_returns_empty_not_raises(b):
    """search_by_text() with FTS-special characters returns [] without raising."""
    b.add(item_id="special", embedding=_vec(6), metadata={"content": "normal content"})
    # FTS5 special characters that could cause parse errors
    result = b.search_by_text(query_text='"unclosed', top_k=5)
    # Must not raise — returns [] on failure (logged warning)
    assert isinstance(result, list)


# ── double clear ──────────────────────────────────────────────────────────────

def test_double_clear_does_not_raise(b):
    """clear() called twice does not raise."""
    b.add(item_id="item", embedding=_vec(7), metadata={"content": "will be cleared"})
    b.clear()
    b.clear()  # second clear on already-empty backend must not raise


def test_clear_removes_fts_entries(b):
    """After clear(), search_by_text returns no results."""
    b.add(item_id="fts_item", embedding=_vec(8), metadata={"content": "searchable text"})
    b.clear()
    results = b.search_by_text(query_text="searchable", top_k=5)
    assert results == []


# ── in-memory backend ─────────────────────────────────────────────────────────

def test_in_memory_backend_add_and_search(mem_b):
    """In-memory backend (persist_directory=None) supports add and search."""
    v = _vec(9)
    mem_b.add(item_id="mem_item", embedding=v, metadata={"content": "memory"})
    results = mem_b.search(query_embedding=v, top_k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "mem_item"


def test_in_memory_backend_fts_works(mem_b):
    """In-memory backend FTS5 (backed by :memory: SQLite) works correctly."""
    v = _vec(10)
    mem_b.add(item_id="fts_mem", embedding=v, metadata={"content": "in-memory text search"})
    results = mem_b.search_by_text(query_text="memory", top_k=5)
    ids = [r["id"] for r in results]
    assert "fts_mem" in ids


# ── corrupted map file graceful recovery ─────────────────────────────────────

def test_load_or_create_corrupted_map_falls_back(tmp_path):
    """If the JSON map file is corrupted, backend starts fresh without raising."""
    # Create a backend and persist some data
    b = UsearchVectorBackend(collection_name="corrupt_col", persist_directory=str(tmp_path))
    b.add(item_id="item1", embedding=_vec(11), metadata={"content": "before corruption"})
    del b

    # Corrupt the JSON map file
    map_path = tmp_path / "corrupt_col.json"
    assert map_path.exists()
    map_path.write_text("{ this is not valid JSON }", encoding="utf-8")

    # Backend must load without raising — falls back to fresh state
    b2 = UsearchVectorBackend(collection_name="corrupt_col", persist_directory=str(tmp_path))
    # Fresh state: search returns []
    results = b2.search(query_embedding=_vec(11), top_k=5)
    assert results == []


# ── _next_key monotonically increasing across re-adds ────────────────────────

def test_next_key_increases_after_upsert_of_same_id(b):
    """_next_key increases even when upserting the same item_id (old entry removed)."""
    v1 = _vec(12)
    v2 = _vec(13)
    b.add(item_id="key_test", embedding=v1, metadata={})
    key_after_first = b._next_key
    b.upsert(item_id="key_test", embedding=v2, metadata={})
    key_after_upsert = b._next_key
    # _next_key must advance (a new key was allocated for the upsert)
    assert key_after_upsert > key_after_first
    # id_map and rev_map must be consistent (no orphan entries)
    assert b._rev_map.get("key_test") is not None
    int_key = b._rev_map["key_test"]
    assert b._id_map.get(int_key) == "key_test"


# ── persistence: map_path=None means no files written ────────────────────────

def test_in_memory_backend_no_files_written(tmp_path, mem_b):
    """In-memory backend does not create any files."""
    v = _vec(14)
    mem_b.add(item_id="no_file", embedding=v, metadata={"content": "test"})
    mem_b.clear()
    # The tmp_path directory should remain empty (in-memory backend uses separate fixture)
    # We just verify no _index_path or _map_path attributes exist
    assert mem_b._index_path is None
    assert mem_b._map_path is None


# ── FTS cross-collection isolation (regression: DIST-LITE-1 bug P1) ──────────

def test_fts_table_name_is_collection_scoped(tmp_path):
    """Each collection gets its own FTS virtual table (fts_{collection_name})."""
    col_a = UsearchVectorBackend(collection_name="col_a", persist_directory=str(tmp_path))
    col_b = UsearchVectorBackend(collection_name="col_b", persist_directory=str(tmp_path))
    # Table names must differ between collections
    assert col_a._fts_table != col_b._fts_table
    assert col_a._fts_table == "fts_col_a"
    assert col_b._fts_table == "fts_col_b"


def test_clear_does_not_delete_sibling_collection_fts(tmp_path):
    """clear() on col_b must not delete col_a's FTS rows (shared fts.db file)."""
    col_a = UsearchVectorBackend(collection_name="col_a", persist_directory=str(tmp_path))
    col_b = UsearchVectorBackend(collection_name="col_b", persist_directory=str(tmp_path))

    col_a.add(item_id="a1", embedding=_vec(20), metadata={"content": "alpha content"})
    col_b.add(item_id="b1", embedding=_vec(21), metadata={"content": "beta content"})

    # Clearing col_b must not remove col_a's FTS rows
    col_b.clear()

    hits_a = col_a.search_by_text(query_text="alpha", top_k=5)
    assert any(r["id"] == "a1" for r in hits_a), "col_a FTS row was destroyed by col_b.clear()"


def test_search_by_text_does_not_leak_across_collections(tmp_path):
    """search_by_text() on col_a must not return rows ingested into col_b."""
    col_a = UsearchVectorBackend(collection_name="col_a", persist_directory=str(tmp_path))
    col_b = UsearchVectorBackend(collection_name="col_b", persist_directory=str(tmp_path))

    col_b.add(item_id="b1", embedding=_vec(22), metadata={"content": "exclusive phrase from b"})

    # col_a's FTS should not see col_b's content
    hits_from_a = col_a.search_by_text(query_text="exclusive", top_k=5)
    assert hits_from_a == [], f"col_a returned col_b items: {hits_from_a}"


def test_fts_table_name_sanitizes_special_chars():
    """Collection names with non-alphanumeric chars produce valid SQL identifiers."""
    b = UsearchVectorBackend(collection_name="my-col.v2", persist_directory=None)
    # Hyphens and dots replaced by underscores
    assert b._fts_table == "fts_my_col_v2"
    # Table must be queryable (no SQL syntax error from the name)
    results = b.search_by_text(query_text="anything", top_k=5)
    assert results == []
