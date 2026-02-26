"""Tests for CORE-OBS-2: trace field kwargs in emit_event() and _emit_span().

Verifies that trace fields (name, trace_id, span_id, parent_span_id, duration_ms)
become top-level Redis Stream fields when passed as kwargs to emit_event(), and
that _emit_span() passes them correctly.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    __getattr__ = dict.get


def _make_spooler(**overrides):
    """Create an EventSpooler with mocked Redis and config.

    Uses patch.dict(sys.modules) rather than patch("redis.Redis") so the test
    works in environments where the redis package is not installed (e.g. a bare
    ``pip install smartmemory`` without [server]).
    """
    mock_redis = MagicMock()
    cfg = _AttrDict(
        {
            "cache": _AttrDict({"redis": _AttrDict({"host": "localhost", "port": 9012})}),
        }
    )
    cfg.get = lambda key, default=None: {} if key in ("observability", "active_namespace") else default

    # Build a minimal fake redis module so that `import redis as _redis` inside
    # EventSpooler succeeds even when the real package is absent.
    mock_redis_mod = MagicMock()
    mock_redis_mod.Redis.return_value = mock_redis
    mock_redis_mod.ResponseError = Exception  # must be a real exception class

    with (
        patch("smartmemory.observability.events.get_config", return_value=cfg),
        patch.dict(sys.modules, {"redis": mock_redis_mod}),
    ):
        from smartmemory.observability.events import EventSpooler

        spooler = EventSpooler(**overrides)
        spooler.redis_client = mock_redis
        spooler._connected = True
        return spooler, mock_redis


def _last_xadd_fields(mock_redis):
    """Extract the fields dict from the most recent xadd call."""
    assert mock_redis.xadd.called, "xadd was never called"
    args, kwargs = mock_redis.xadd.call_args
    return args[1] if len(args) > 1 else kwargs.get("fields", {})


# ---------------------------------------------------------------------------
# emit_event() — trace fields as top-level Stream fields
# ---------------------------------------------------------------------------


class TestEmitEventTraceFields:
    def test_trace_fields_become_top_level(self):
        """CORE-OBS-2: trace kwargs appear as top-level Redis Stream fields."""
        spooler, mock_redis = _make_spooler()
        spooler.emit_event(
            event_type="span",
            component="memory",
            operation="add",
            data={"extra": "stuff"},
            name="memory.add",
            trace_id="abc123",
            span_id="def456",
            parent_span_id="ghi789",
            duration_ms=42.5,
        )
        fields = _last_xadd_fields(mock_redis)
        assert fields["name"] == "memory.add"
        assert fields["trace_id"] == "abc123"
        assert fields["span_id"] == "def456"
        assert fields["parent_span_id"] == "ghi789"
        assert fields["duration_ms"] == "42.5"

    def test_none_trace_fields_omitted(self):
        """Trace fields with None value are not added to Stream fields."""
        spooler, mock_redis = _make_spooler()
        spooler.emit_event(
            event_type="span",
            component="memory",
            operation="add",
        )
        fields = _last_xadd_fields(mock_redis)
        assert "name" not in fields
        assert "trace_id" not in fields
        assert "span_id" not in fields
        assert "parent_span_id" not in fields
        assert "duration_ms" not in fields

    def test_backward_compat_no_trace_kwargs(self):
        """Existing callers without trace kwargs still work."""
        spooler, mock_redis = _make_spooler()
        spooler.emit_event(
            event_type="memory.added",
            component="memory",
            operation="add",
            data={"item_id": "123"},
        )
        fields = _last_xadd_fields(mock_redis)
        assert fields["event_type"] == "memory.added"
        assert fields["component"] == "memory"
        assert "name" not in fields

    def test_duration_ms_zero_included(self):
        """duration_ms=0 (sub-ms span) is included as '0'."""
        spooler, mock_redis = _make_spooler()
        spooler.emit_event(
            event_type="span",
            component="pipeline",
            operation="classify",
            duration_ms=0,
        )
        fields = _last_xadd_fields(mock_redis)
        assert fields["duration_ms"] == "0"

    def test_duration_ms_rounded(self):
        """duration_ms is rounded to 2 decimal places."""
        spooler, mock_redis = _make_spooler()
        spooler.emit_event(
            event_type="span",
            component="pipeline",
            operation="classify",
            duration_ms=42.5678,
        )
        fields = _last_xadd_fields(mock_redis)
        assert fields["duration_ms"] == "42.57"

    def test_data_blob_preserved(self):
        """data blob is still serialized as JSON alongside trace fields."""
        spooler, mock_redis = _make_spooler()
        spooler.emit_event(
            event_type="span",
            component="memory",
            operation="add",
            data={"name": "memory.add", "trace_id": "abc"},
            name="memory.add",
            trace_id="abc",
        )
        fields = _last_xadd_fields(mock_redis)
        data = json.loads(fields["data"])
        assert data["name"] == "memory.add"
        assert fields["name"] == "memory.add"  # Also at top level

    def test_metadata_param_preserved(self):
        """Existing metadata parameter still works alongside trace fields."""
        spooler, mock_redis = _make_spooler()
        spooler.unified_keys = True
        spooler.emit_event(
            event_type="span",
            component="memory",
            operation="add",
            metadata={"domain": "memory", "category": "crud", "action": "add"},
            trace_id="abc",
        )
        fields = _last_xadd_fields(mock_redis)
        assert fields["domain"] == "memory"
        assert fields["trace_id"] == "abc"


# ---------------------------------------------------------------------------
# _emit_span() — passes trace fields as kwargs
# ---------------------------------------------------------------------------


class TestEmitSpanKwargs:
    def test_emit_span_passes_trace_kwargs(self):
        """_emit_span() passes trace fields as explicit kwargs to emit_event."""
        mock_spooler = MagicMock()
        with patch("smartmemory.observability.tracing._get_spooler", return_value=mock_spooler):
            from smartmemory.observability.tracing import _emit_span

            _emit_span(
                {
                    "event_type": "span",
                    "component": "memory",
                    "operation": "add",
                    "name": "memory.add",
                    "trace_id": "abc123",
                    "span_id": "def456",
                    "parent_span_id": "ghi789",
                    "duration_ms": 42.5,
                }
            )

        mock_spooler.emit_event.assert_called_once()
        call_kwargs = mock_spooler.emit_event.call_args
        # Check keyword arguments
        assert call_kwargs.kwargs["name"] == "memory.add"
        assert call_kwargs.kwargs["trace_id"] == "abc123"
        assert call_kwargs.kwargs["span_id"] == "def456"
        assert call_kwargs.kwargs["parent_span_id"] == "ghi789"
        assert call_kwargs.kwargs["duration_ms"] == 42.5

    def test_emit_span_missing_fields_pass_none(self):
        """_emit_span() passes None for missing trace fields."""
        mock_spooler = MagicMock()
        with patch("smartmemory.observability.tracing._get_spooler", return_value=mock_spooler):
            from smartmemory.observability.tracing import _emit_span

            _emit_span(
                {
                    "event_type": "span",
                    "component": "memory",
                    "operation": "add",
                }
            )

        call_kwargs = mock_spooler.emit_event.call_args
        assert call_kwargs.kwargs["name"] is None
        assert call_kwargs.kwargs["trace_id"] is None
