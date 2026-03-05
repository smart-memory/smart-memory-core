"""Tests for UsearchVectorBackend (DIST-LITE-1 Step 2). No Docker required."""
import numpy as np
import pytest

from smartmemory.stores.vector.backends.usearch import UsearchVectorBackend


@pytest.fixture
def backend(tmp_path):
    return UsearchVectorBackend(
        collection_name="test_col",
        persist_directory=str(tmp_path),
    )


def _vec(seed: int, dim: int = 8) -> list:
    """Deterministic unit vector for testing."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_add_and_search(backend):
    """add() -> search() returns the added item."""
    v = _vec(1)
    backend.add(item_id="item_a", embedding=v, metadata={"content": "hello"})
    results = backend.search(query_embedding=v, top_k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "item_a"


def test_upsert_overwrites(backend):
    """upsert() replaces an existing item_id."""
    v1 = _vec(1)
    v2 = _vec(2)
    backend.add(item_id="item_x", embedding=v1, metadata={"content": "old"})
    backend.upsert(item_id="item_x", embedding=v2, metadata={"content": "new"})
    # Search with v2 — item_x should be top result
    results = backend.search(query_embedding=v2, top_k=1)
    assert results[0]["id"] == "item_x"


def test_search_by_text(backend):
    """search_by_text() returns item_id for content containing the keyword."""
    v = _vec(3)
    backend.add(item_id="item_alice", embedding=v, metadata={"content": "Alice works on Project Atlas"})
    results = backend.search_by_text(query_text="Alice", top_k=5)
    ids = [r["id"] if isinstance(r, dict) else r for r in results]
    assert "item_alice" in ids


def test_persistence(tmp_path):
    """Items persist across backend instances with same path."""
    v = _vec(4)
    b1 = UsearchVectorBackend("persist_col", str(tmp_path))
    b1.add(item_id="persist_item", embedding=v, metadata={"content": "remember me"})
    del b1
    b2 = UsearchVectorBackend("persist_col", str(tmp_path))
    results = b2.search(query_embedding=v, top_k=1)
    assert results[0]["id"] == "persist_item"


def test_clear(backend):
    """clear() empties the index — search returns empty list."""
    v = _vec(5)
    backend.add(item_id="to_clear", embedding=v, metadata={"content": "gone"})
    backend.clear()
    results = backend.search(query_embedding=v, top_k=5)
    assert len(results) == 0


def test_registered_in_factory(tmp_path):
    """'usearch' key resolves to UsearchVectorBackend via the factory."""
    from smartmemory.stores.vector.backends.base import create_backend

    b = create_backend("usearch", "factory_col", str(tmp_path))
    assert isinstance(b, UsearchVectorBackend)


def test_add_with_numpy_embedding_persists(tmp_path):
    """add() with np.ndarray embedding must JSON-serialize on save (DIST-QA-1 F2)."""
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    b1 = UsearchVectorBackend("np_col", str(tmp_path))
    # Should not raise TypeError on _save()
    b1.add(item_id="np_item", embedding=vec, metadata={"content": "numpy test"})
    del b1
    # Reload from disk — proves JSON round-trip worked
    b2 = UsearchVectorBackend("np_col", str(tmp_path))
    result = b2.get("np_item")
    assert result is not None
    assert result["id"] == "np_item"
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) == 4
