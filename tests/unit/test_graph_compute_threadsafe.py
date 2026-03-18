"""Tests for GraphComputeLayer thread-safety (CORE-EVO-LIVE-1 T1)."""

import threading
import time

import pytest


@pytest.fixture
def compute():
    """Create a GraphComputeLayer with a mock backend."""
    from unittest.mock import MagicMock

    from smartmemory.graph.compute import GraphComputeLayer

    backend = MagicMock()
    backend.get_all_nodes.return_value = []
    backend.get_all_edges.return_value = []
    return GraphComputeLayer(backend)


class TestThreadSafety:
    def test_concurrent_add_and_read(self, compute):
        """Two threads: one adding nodes, one reading — no crashes."""
        errors = []

        def writer():
            try:
                for i in range(100):
                    compute.sync_add_node(f"w-{i}", {"val": i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    compute.node_count()
                    compute.get_node("w-0")
                    compute.get_neighbors("w-0")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert compute.node_count() == 100

    def test_concurrent_add_and_remove(self, compute):
        """Writer adds nodes, remover removes them — no crashes."""
        # Pre-populate
        for i in range(50):
            compute.sync_add_node(f"n-{i}", {"val": i})

        errors = []

        def adder():
            try:
                for i in range(50, 100):
                    compute.sync_add_node(f"n-{i}", {"val": i})
            except Exception as e:
                errors.append(e)

        def remover():
            try:
                for i in range(50):
                    compute.sync_remove_node(f"n-{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=remover)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        # 50 added, 50 removed = 50 remaining
        assert compute.node_count() == 50

    def test_snapshot_is_independent_copy(self, compute):
        """Snapshot doesn't see mutations after it's taken."""
        compute.sync_add_node("a", {"val": 1})
        compute.sync_add_node("b", {"val": 2})

        snap = compute.snapshot()
        assert snap.number_of_nodes() == 2

        # Mutate after snapshot
        compute.sync_add_node("c", {"val": 3})
        assert snap.number_of_nodes() == 2  # unchanged
        assert compute.node_count() == 3  # live graph updated


class TestLockedAccessors:
    def test_get_node_existing(self, compute):
        compute.sync_add_node("x", {"name": "test", "val": 42})
        result = compute.get_node("x")
        assert result is not None
        assert result["name"] == "test"
        assert result["val"] == 42

    def test_get_node_missing(self, compute):
        assert compute.get_node("nonexistent") is None

    def test_get_neighbors(self, compute):
        compute.sync_add_node("a", {"name": "alpha"})
        compute.sync_add_node("b", {"name": "beta"})
        compute.sync_add_edge("a", "b", "RELATES_TO", {})

        neighbors = compute.get_neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0]["item_id"] == "b"
        assert neighbors[0]["name"] == "beta"

    def test_get_neighbors_missing_node(self, compute):
        assert compute.get_neighbors("nonexistent") == []

    def test_has_node(self, compute):
        compute.sync_add_node("x", {})
        assert compute.has_node("x")
        assert not compute.has_node("y")

    def test_node_count(self, compute):
        assert compute.node_count() == 0
        compute.sync_add_node("a", {})
        compute.sync_add_node("b", {})
        assert compute.node_count() == 2

    def test_edge_count(self, compute):
        compute.sync_add_node("a", {})
        compute.sync_add_node("b", {})
        assert compute.edge_count() == 0
        compute.sync_add_edge("a", "b", "KNOWS", {})
        assert compute.edge_count() == 1

    def test_sync_clear(self, compute):
        compute.sync_add_node("a", {})
        compute.sync_add_node("b", {})
        compute.sync_clear()
        assert compute.node_count() == 0
