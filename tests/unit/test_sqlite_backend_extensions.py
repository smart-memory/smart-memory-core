"""Unit tests for CORE-EVO-LIVE-1: SQLiteBackend set_properties and transaction_context."""

from smartmemory.graph.backends.sqlite import SQLiteBackend


class TestSetProperties:
    def test_set_properties_merges(self):
        backend = SQLiteBackend(":memory:")
        backend.add_node("item-1", {"content": "hello", "score": 0.5}, memory_type="semantic")

        result = backend.set_properties("item-1", {"score": 0.9, "extra": "new"})
        assert result is True

        node = backend.get_node("item-1")
        assert node["score"] == 0.9
        assert node["extra"] == "new"
        assert node["content"] == "hello"  # Preserved

    def test_set_properties_nonexistent_returns_false(self):
        backend = SQLiteBackend(":memory:")
        assert backend.set_properties("no-such-id", {"x": 1}) is False

    def test_set_properties_empty_dict(self):
        backend = SQLiteBackend(":memory:")
        backend.add_node("item-1", {"content": "hello"}, memory_type="semantic")
        assert backend.set_properties("item-1", {}) is True

    def test_set_properties_syncs_compute_layer(self):
        backend = SQLiteBackend(":memory:")
        backend.add_node("item-1", {"content": "hello"}, memory_type="semantic")
        backend.set_properties("item-1", {"score": 0.9})

        # Compute layer should reflect the update
        node = backend._compute.get_node("item-1")
        assert node is not None
        assert node.get("score") == 0.9


class TestTransactionContext:
    def test_transaction_context_commits(self):
        backend = SQLiteBackend(":memory:")
        with backend.transaction_context():
            backend._conn.execute(
                "INSERT INTO nodes (item_id, memory_type, properties) VALUES (?, ?, ?)",
                ("tx-1", "semantic", '{"content": "test"}'),
            )
        # Should be committed
        node = backend.get_node("tx-1")
        assert node is not None

    def test_transaction_context_rolls_back_on_error(self):
        backend = SQLiteBackend(":memory:")
        try:
            with backend.transaction_context():
                backend._conn.execute(
                    "INSERT INTO nodes (item_id, memory_type, properties) VALUES (?, ?, ?)",
                    ("tx-2", "semantic", '{}'),
                )
                raise ValueError("intentional error")
        except ValueError:
            pass
        # Should have been rolled back
        node = backend.get_node("tx-2")
        assert node is None
