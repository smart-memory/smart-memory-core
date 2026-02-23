"""Unit tests for smartmemory.observability.retrieval_tracking.

Verifies:
- emit_retrieval_get() calls XADD with correct stream key and MAXLEN
- emit_retrieval_search() calls XADD with batched results (one event, not per result)
- SADD smartmemory:active_workspaces called on each emission
- Exceptions in XADD do not propagate to the caller
- retrieval_source context var is captured into the event ``source`` field
- Workspace ID is extracted correctly from scope_provider
"""

import json
from contextvars import copy_context
from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit

from smartmemory.observability.retrieval_tracking import (
    emit_retrieval_get,
    emit_retrieval_search,
    retrieval_source,
)
from smartmemory.models.memory_item import MemoryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(item_id: str = "item-1", score: float = 0.9) -> MemoryItem:
    item = MemoryItem(item_id=item_id, content="test", memory_type="episodic")
    item.score = score
    return item


def _make_scope(workspace_id: str = "ws-test") -> MagicMock:
    scope = MagicMock()
    scope.get_write_context.return_value = {"workspace_id": workspace_id}
    return scope


def _make_redis_mock() -> MagicMock:
    mock = MagicMock()
    mock.xadd.return_value = "1234567890-0"
    mock.sadd.return_value = 1
    return mock


# ---------------------------------------------------------------------------
# emit_retrieval_get tests
# ---------------------------------------------------------------------------


class TestEmitRetrievalGet:
    """Tests for emit_retrieval_get()."""

    def test_xadd_called_with_correct_stream_key(self):
        item = _make_item("item-abc")
        scope = _make_scope("ws-42")
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        xadd_calls = redis_mock.xadd.call_args_list
        assert len(xadd_calls) == 1
        stream_key = xadd_calls[0][0][0]
        assert stream_key == "smartmemory:retrievals:ws-42"

    def test_xadd_called_with_maxlen_50000(self):
        item = _make_item()
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        _, kwargs = redis_mock.xadd.call_args
        assert kwargs.get("maxlen") == 50_000

    def test_xadd_called_with_approximate_true(self):
        item = _make_item()
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        _, kwargs = redis_mock.xadd.call_args
        assert kwargs.get("approximate") is True

    def test_event_type_is_get(self):
        item = _make_item("item-abc")
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        fields = redis_mock.xadd.call_args[0][1]
        assert fields["type"] == "get"

    def test_event_includes_item_id(self):
        item = _make_item("my-item-id")
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        fields = redis_mock.xadd.call_args[0][1]
        assert fields["item_id"] == "my-item-id"

    def test_event_includes_ts(self):
        item = _make_item()
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        fields = redis_mock.xadd.call_args[0][1]
        assert "ts" in fields and fields["ts"]

    def test_sadd_called_with_active_workspaces_key(self):
        item = _make_item()
        scope = _make_scope("ws-99")
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        redis_mock.sadd.assert_called_once_with("smartmemory:active_workspaces", "ws-99")

    def test_xadd_exception_does_not_propagate(self):
        item = _make_item()
        scope = _make_scope()
        redis_mock = _make_redis_mock()
        redis_mock.xadd.side_effect = RuntimeError("Redis down")

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            # Must not raise
            emit_retrieval_get(item, scope)

    def test_none_redis_client_does_not_propagate(self):
        item = _make_item()
        scope = _make_scope()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=None,
        ):
            emit_retrieval_get(item, scope)  # Must not raise

    def test_retrieval_source_captured_in_event(self):
        item = _make_item()
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        token = retrieval_source.set("api.GET /memory/item-1")
        try:
            with patch(
                "smartmemory.observability.retrieval_tracking._get_redis_client",
                return_value=redis_mock,
            ):
                emit_retrieval_get(item, scope)

            fields = redis_mock.xadd.call_args[0][1]
            assert fields["source"] == "api.GET /memory/item-1"
        finally:
            retrieval_source.reset(token)

    def test_default_source_is_internal(self):
        item = _make_item()
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_get(item, scope)

        fields = redis_mock.xadd.call_args[0][1]
        assert fields["source"] == "internal"


# ---------------------------------------------------------------------------
# emit_retrieval_search tests
# ---------------------------------------------------------------------------


class TestEmitRetrievalSearch:
    """Tests for emit_retrieval_search()."""

    def test_single_xadd_call_for_multiple_results(self):
        """One search call → one XADD event regardless of result count."""
        results = [_make_item("item-1"), _make_item("item-2"), _make_item("item-3")]
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search(results, "test query", scope)

        assert redis_mock.xadd.call_count == 1

    def test_xadd_stream_key_per_workspace(self):
        scope = _make_scope("ws-search-42")
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], "q", scope)

        stream_key = redis_mock.xadd.call_args[0][0]
        assert stream_key == "smartmemory:retrievals:ws-search-42"

    def test_results_json_packed_in_single_event(self):
        """All result item_ids appear in results_json in the single event."""
        results = [_make_item("a"), _make_item("b")]
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search(results, "q", scope)

        fields = redis_mock.xadd.call_args[0][1]
        parsed = json.loads(fields["results_json"])
        item_ids = [r["item_id"] for r in parsed]
        assert "a" in item_ids
        assert "b" in item_ids

    def test_ranks_are_1_indexed(self):
        results = [_make_item("first"), _make_item("second")]
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search(results, "q", scope)

        fields = redis_mock.xadd.call_args[0][1]
        parsed = json.loads(fields["results_json"])
        ranks = [r["rank"] for r in parsed]
        assert ranks == [1, 2]

    def test_event_type_is_search(self):
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], "q", scope)

        fields = redis_mock.xadd.call_args[0][1]
        assert fields["type"] == "search"

    def test_query_hash_is_md5(self):
        import hashlib

        scope = _make_scope()
        redis_mock = _make_redis_mock()
        query = "  Test Query  "

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], query, scope)

        fields = redis_mock.xadd.call_args[0][1]
        expected_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()
        assert fields["query_hash"] == expected_hash

    def test_sadd_active_workspaces_called(self):
        scope = _make_scope("ws-s")
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], "q", scope)

        redis_mock.sadd.assert_called_once_with("smartmemory:active_workspaces", "ws-s")

    def test_xadd_exception_does_not_propagate(self):
        scope = _make_scope()
        redis_mock = _make_redis_mock()
        redis_mock.xadd.side_effect = ConnectionError("Redis gone")

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([_make_item()], "q", scope)  # Must not raise

    def test_empty_results_still_emits_event(self):
        """Even empty search results generate one XADD call."""
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], "q", scope)

        assert redis_mock.xadd.call_count == 1

    def test_retrieval_source_captured_in_event(self):
        scope = _make_scope()
        redis_mock = _make_redis_mock()
        token = retrieval_source.set("api.POST /memory/search")
        try:
            with patch(
                "smartmemory.observability.retrieval_tracking._get_redis_client",
                return_value=redis_mock,
            ):
                emit_retrieval_search([], "q", scope)

            fields = redis_mock.xadd.call_args[0][1]
            assert fields["source"] == "api.POST /memory/search"
        finally:
            retrieval_source.reset(token)

    def test_maxlen_is_50000(self):
        scope = _make_scope()
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], "q", scope)

        _, kwargs = redis_mock.xadd.call_args
        assert kwargs.get("maxlen") == 50_000

    def test_workspace_default_when_scope_fails(self):
        """If get_write_context raises, workspace defaults to 'default'."""
        scope = MagicMock()
        scope.get_write_context.side_effect = RuntimeError("auth broken")
        redis_mock = _make_redis_mock()

        with patch(
            "smartmemory.observability.retrieval_tracking._get_redis_client",
            return_value=redis_mock,
        ):
            emit_retrieval_search([], "q", scope)  # Must not raise

        stream_key = redis_mock.xadd.call_args[0][0]
        assert stream_key == "smartmemory:retrievals:default"
