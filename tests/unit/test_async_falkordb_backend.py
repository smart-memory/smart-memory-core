"""Unit tests for AsyncFalkorDBBackend and AsyncSmartGraphBackend ABC.

CI-safe: no Docker or real FalkorDB required.
All async graph I/O is mocked at the AsyncGraph level.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartmemory.graph.backends.async_backend import AsyncSmartGraphBackend
from smartmemory.graph.backends.async_falkordb import AsyncFalkorDBBackend
from smartmemory.graph.backends.backend import SmartGraphBackend
from smartmemory.scope_provider import DefaultScopeProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_set(rows: list) -> MagicMock:
    """Return a mock QueryResult with a populated result_set."""
    qr = MagicMock()
    qr.result_set = rows
    return qr


def _make_node(props: dict) -> MagicMock:
    """Return a mock FalkorDB Node object."""
    node = MagicMock()
    node.properties = props
    # make keys()/values() work for serialize()
    node.keys.return_value = list(props.keys())
    node.values.return_value = list(props.values())
    return node


def _make_backend(scope_provider=None, **kwargs) -> AsyncFalkorDBBackend:
    b = AsyncFalkorDBBackend(
        host="localhost",
        port=9010,
        graph_name="test_graph",
        scope_provider=scope_provider or DefaultScopeProvider(),
        **kwargs,
    )
    # Inject a mock graph so we never touch the network
    b._graph = MagicMock()
    b._graph.query = AsyncMock(return_value=_make_result_set([]))
    b._graph.ro_query = AsyncMock(return_value=_make_result_set([]))
    b._graph.delete = AsyncMock()
    # Inject a mock db connection for close()
    b._db = MagicMock()
    b._db.connection = MagicMock()
    b._db.connection.aclose = AsyncMock()
    return b


# ---------------------------------------------------------------------------
# 1. ABC method parity
# ---------------------------------------------------------------------------

class TestAbcMethodParity:
    """AsyncSmartGraphBackend exposes the same 10 + 2 methods as SmartGraphBackend."""

    ABSTRACT_METHODS = {
        "add_node", "add_edge", "get_node", "get_neighbors",
        "remove_node", "remove_edge", "search_nodes", "serialize",
        "deserialize", "clear",
    }
    BULK_METHODS = {"add_nodes_bulk", "add_edges_bulk"}

    def test_all_abstract_methods_present(self):
        async_abstract = {
            name for name, method in inspect.getmembers(AsyncSmartGraphBackend)
            if getattr(method, "__isabstractmethod__", False)
        }
        assert self.ABSTRACT_METHODS == async_abstract

    def test_bulk_methods_present(self):
        for name in self.BULK_METHODS:
            assert hasattr(AsyncSmartGraphBackend, name), f"Missing bulk method: {name}"

    def test_abstract_methods_are_coroutines(self):
        """All abstract methods on the ABC must be declared as async."""
        for name in self.ABSTRACT_METHODS:
            fn = getattr(AsyncSmartGraphBackend, name)
            assert inspect.iscoroutinefunction(fn), f"{name} must be async def"

    def test_bulk_methods_are_coroutines(self):
        for name in self.BULK_METHODS:
            fn = getattr(AsyncSmartGraphBackend, name)
            assert inspect.iscoroutinefunction(fn), f"{name} must be async def"

    def test_sync_and_async_abc_share_same_method_names(self):
        sync_names = {
            name for name, _ in inspect.getmembers(SmartGraphBackend)
            if not name.startswith("_") and callable(getattr(SmartGraphBackend, name))
        }
        async_names = {
            name for name, _ in inspect.getmembers(AsyncSmartGraphBackend)
            if not name.startswith("_") and callable(getattr(AsyncSmartGraphBackend, name))
        }
        # Core abstract + bulk must all be present in the async ABC
        for method in self.ABSTRACT_METHODS | self.BULK_METHODS:
            assert method in async_names, f"{method} missing from AsyncSmartGraphBackend"


# ---------------------------------------------------------------------------
# 2. Connection lifecycle
# ---------------------------------------------------------------------------

class TestConnectionLifecycle:
    async def test_connect_creates_client_and_selects_graph(self):
        backend = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name="myg")
        mock_graph = MagicMock()
        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph

        with patch("smartmemory.graph.backends.async_falkordb.AsyncFalkorDB", return_value=mock_db):
            await backend.connect()

        mock_db.select_graph.assert_called_once_with("myg")
        assert backend._graph is mock_graph

    async def test_close_calls_aclose_on_connection(self):
        backend = _make_backend()
        # Capture reference before close() nulls _db
        mock_aclose = backend._db.connection.aclose
        await backend.close()
        mock_aclose.assert_called_once()

    async def test_close_is_idempotent(self):
        backend = _make_backend()
        await backend.close()
        await backend.close()  # second call must not raise

    async def test_context_manager_calls_connect_and_close(self):
        backend = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name="g")
        mock_graph = MagicMock()
        mock_db = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_db.connection = MagicMock()
        mock_db.connection.aclose = AsyncMock()

        with patch("smartmemory.graph.backends.async_falkordb.AsyncFalkorDB", return_value=mock_db):
            async with backend as b:
                assert b is backend
                assert backend._graph is mock_graph

        mock_db.connection.aclose.assert_called_once()

    def test_operations_before_connect_raise(self):
        backend = AsyncFalkorDBBackend(host="localhost", port=9010, graph_name="g")
        # _graph is None — _ensure_connected should raise
        with pytest.raises(RuntimeError, match="connect()"):
            backend._ensure_connected()


# ---------------------------------------------------------------------------
# 3. Scope injection on writes
# ---------------------------------------------------------------------------

class TestScopeInjection:
    async def test_add_node_injects_write_context(self):
        scope = DefaultScopeProvider(workspace_id="ws-1", user_id="u-1")
        backend = _make_backend(scope_provider=scope)
        # Patch node_exists to return False (new node path)
        backend._graph.ro_query.return_value = _make_result_set([[0]])

        await backend.add_node("node-1", {"content": "hello"}, memory_type="semantic")

        call_args = backend._graph.query.call_args
        cypher, params = call_args[0]

        # Write context must appear as Cypher params
        assert params.get("prop_workspace_id") == "ws-1"
        assert params.get("prop_user_id") == "u-1"

    async def test_add_node_global_skips_scope(self):
        scope = DefaultScopeProvider(workspace_id="ws-1", user_id="u-1")
        backend = _make_backend(scope_provider=scope)
        backend._graph.ro_query.return_value = _make_result_set([[0]])

        await backend.add_node("node-g", {"content": "shared"}, memory_type="semantic", is_global=True)

        call_args = backend._graph.query.call_args
        _, params = call_args[0]

        assert "prop_workspace_id" not in params
        assert "prop_user_id" not in params

    async def test_add_edge_injects_workspace_in_query(self):
        scope = DefaultScopeProvider(workspace_id="ws-2")
        backend = _make_backend(scope_provider=scope)
        # First ro_query = edge verify (0 count), second = source/target checks
        backend._graph.ro_query.side_effect = [
            _make_result_set([[1]]),  # edge_count = 1 => success
        ]

        result = await backend.add_edge("a", "b", "LINKS", {})

        call_args = backend._graph.query.call_args
        cypher, params = call_args[0]
        assert "workspace_id" in cypher
        assert params.get("ws_id") == "ws-2"
        assert result is True


# ---------------------------------------------------------------------------
# 4. Read operations use ro_query
# ---------------------------------------------------------------------------

class TestReadOperationsUseRoQuery:
    async def test_get_node_uses_ro_query(self):
        backend = _make_backend()
        node = _make_node({"item_id": "n1", "content": "hello"})
        backend._graph.ro_query.return_value = _make_result_set([[node]])

        result = await backend.get_node("n1")

        backend._graph.ro_query.assert_called()
        backend._graph.query.assert_not_called()
        assert result["content"] == "hello"

    async def test_search_nodes_uses_ro_query(self):
        backend = _make_backend()
        node = _make_node({"item_id": "n2", "type": "fact"})
        backend._graph.ro_query.return_value = _make_result_set([[node]])

        await backend.search_nodes({"type": "fact"})

        backend._graph.ro_query.assert_called()

    async def test_get_neighbors_uses_ro_query(self):
        backend = _make_backend()
        backend._graph.ro_query.return_value = _make_result_set([])

        await backend.get_neighbors("n1")

        backend._graph.ro_query.assert_called()
        backend._graph.query.assert_not_called()

    async def test_node_exists_uses_ro_query(self):
        backend = _make_backend()
        backend._graph.ro_query.return_value = _make_result_set([[1]])

        result = await backend.node_exists("n1")

        backend._graph.ro_query.assert_called()
        assert result is True


# ---------------------------------------------------------------------------
# 5. Search nodes — isolation filters
# ---------------------------------------------------------------------------

class TestSearchNodesIsolation:
    async def test_scoped_search_appends_isolation_filters(self):
        scope = DefaultScopeProvider(workspace_id="ws-3", user_id="u-3")
        backend = _make_backend(scope_provider=scope)
        backend._graph.ro_query.return_value = _make_result_set([])

        await backend.search_nodes({"memory_type": "semantic"})

        call_args = backend._graph.ro_query.call_args
        cypher, params = call_args[0]
        assert "ws-3" in params.values()
        assert "u-3" in params.values()
        assert "WHERE" in cypher

    async def test_unscoped_search_has_no_isolation_params(self):
        backend = _make_backend()  # DefaultScopeProvider with no ids
        backend._graph.ro_query.return_value = _make_result_set([])

        await backend.search_nodes({})

        call_args = backend._graph.ro_query.call_args
        _, params = call_args[0]
        # No scope keys injected
        assert not any(k.startswith("ctx_") for k in params)


# ---------------------------------------------------------------------------
# 6. Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    async def test_query_failure_propagates(self):
        backend = _make_backend()
        backend._graph.query.side_effect = Exception("FalkorDB down")

        with pytest.raises(Exception, match="FalkorDB down"):
            await backend._query("MATCH (n) RETURN n")

    async def test_add_edge_failure_returns_false_and_logs(self):
        backend = _make_backend()
        backend._graph.query.side_effect = RuntimeError("network error")

        result = await backend.add_edge("a", "b", "REL", {})

        assert result is False

    async def test_node_exists_returns_false_on_exception(self):
        backend = _make_backend()
        backend._graph.ro_query.side_effect = Exception("timeout")

        result = await backend.node_exists("n1")

        assert result is False


# ---------------------------------------------------------------------------
# 7. Property serialization parity
# ---------------------------------------------------------------------------

class TestPropertySerializationParity:
    """_is_valid_property and _serialize_value must behave identically to sync."""

    from smartmemory.graph.backends.async_falkordb import _is_valid_property, _serialize_value
    from smartmemory.graph.backends.falkordb import FalkorDBBackend as _SyncBackend

    def test_valid_property_types_match_sync(self):
        from smartmemory.graph.backends.async_falkordb import _is_valid_property
        sync = self._SyncBackend.__new__(self._SyncBackend)
        pairs = [
            ("key", "value", True),
            ("key", 42, True),
            ("key", 3.14, True),
            ("key", True, True),
            ("key", None, False),
            ("embedding", [0.1, 0.2], False),
            ("key", "", False),
        ]
        for key, value, expected in pairs:
            assert _is_valid_property(key, value) == expected, f"Mismatch for ({key!r}, {value!r})"

    def test_serialize_datetime_to_iso(self):
        from smartmemory.graph.backends.async_falkordb import _serialize_value
        from datetime import datetime, timezone
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _serialize_value(dt)
        assert result == dt.isoformat()

    def test_serialize_list_to_json_string(self):
        from smartmemory.graph.backends.async_falkordb import _serialize_value
        result = _serialize_value(["a", "b"])
        assert result == '["a", "b"]'

    def test_serialize_passthrough_for_primitives(self):
        from smartmemory.graph.backends.async_falkordb import _serialize_value
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(42) == 42
        assert _serialize_value(True) is True


# ---------------------------------------------------------------------------
# 8. Clear
# ---------------------------------------------------------------------------

class TestClear:
    async def test_clear_calls_graph_delete(self):
        backend = _make_backend()
        await backend.clear()
        backend._graph.delete.assert_called_once()

    async def test_clear_ignores_empty_key_error(self):
        backend = _make_backend()
        backend._graph.delete.side_effect = Exception("Invalid graph operation on empty key")
        await backend.clear()  # must not raise

    async def test_clear_reraises_other_errors(self):
        backend = _make_backend()
        backend._graph.delete.side_effect = Exception("connection refused")
        with pytest.raises(Exception, match="connection refused"):
            await backend.clear()


# ---------------------------------------------------------------------------
# 9. Bulk methods
# ---------------------------------------------------------------------------

class TestBulkMethods:
    async def test_add_nodes_bulk_returns_zero_for_empty(self):
        backend = _make_backend()
        result = await backend.add_nodes_bulk([])
        assert result == 0

    async def test_add_edges_bulk_returns_zero_for_empty(self):
        backend = _make_backend()
        result = await backend.add_edges_bulk([])
        assert result == 0

    async def test_add_nodes_bulk_uses_unwind(self):
        backend = _make_backend()
        backend._graph.query.return_value = _make_result_set([[2]])
        nodes = [
            {"item_id": "n1", "memory_type": "semantic", "content": "a"},
            {"item_id": "n2", "memory_type": "semantic", "content": "b"},
        ]
        total = await backend.add_nodes_bulk(nodes)
        call_args = backend._graph.query.call_args
        cypher, params = call_args[0]
        assert "UNWIND" in cypher
        assert "batch" in params

    async def test_add_nodes_bulk_skips_nodes_without_item_id(self):
        backend = _make_backend()
        backend._graph.query.return_value = _make_result_set([[1]])
        nodes = [
            {"item_id": "n1", "memory_type": "semantic"},
            {"memory_type": "semantic"},  # no item_id — must skip
        ]
        await backend.add_nodes_bulk(nodes)
        call_args = backend._graph.query.call_args
        _, params = call_args[0]
        batch = params["batch"]
        assert len(batch) == 1
        assert batch[0]["item_id"] == "n1"
