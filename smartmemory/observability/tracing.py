"""Unified observability via trace_span() -- single API for all event emission."""

import logging
import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanContext:
    """Context for a single traced span."""

    trace_id: str = ""
    span_id: str = ""
    parent_span_id: Optional[str] = None
    name: str = ""
    start_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def emit_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Emit a point-in-time span event attached to this span.

        Unlike trace_span(), records an instantaneous occurrence with no duration.
        Inherits trace_id from this span; uses this span's span_id as parent reference.

        Args:
            name: Event name in dot notation (e.g., "graph.add_node").
            attributes: Key-value payload for the event.
        """
        if not self.trace_id:  # no-op when observability is disabled (empty SpanContext)
            return
        resolved_component = name.split(".")[0] if name else "unknown"
        operation = name.split(".", 1)[1] if "." in name else name
        event_data = {
            **(attributes or {}),
            "event_type": "span_event",
            "component": resolved_component,
            "operation": operation,
            "name": name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }
        try:
            _emit_span(event_data)
        except Exception:
            pass


_current_span: ContextVar[Optional[SpanContext]] = ContextVar("_current_span", default=None)


def _is_enabled() -> bool:
    """Read observability toggle from env at call time.

    Default: enabled. Set SMARTMEMORY_OBSERVABILITY=false to disable.
    Reading at call time (instead of module load) allows SmartMemory(observability=False)
    to set the env var after import and have it take effect immediately.
    """
    return os.environ.get("SMARTMEMORY_OBSERVABILITY", "true").strip().lower() not in (
        "false",
        "0",
        "no",
        "off",
    )


_spooler = None
_spooler_lock = threading.Lock()
_SPOOLER_FAILED = object()  # Sentinel: distinguishes "never tried" from "tried and failed"


def _get_spooler():
    global _spooler
    if _spooler is _SPOOLER_FAILED:
        return None
    if _spooler is not None:
        return _spooler
    with _spooler_lock:
        if _spooler is _SPOOLER_FAILED:
            return None
        if _spooler is not None:
            return _spooler
        try:
            from smartmemory.observability.events import EventSpooler

            _spooler = EventSpooler()
            return _spooler
        except Exception as e:
            logger.debug(f"Could not create EventSpooler: {e}")
            _spooler = _SPOOLER_FAILED
            return None


def _emit_span(event_data: dict) -> None:
    spooler = _get_spooler()
    if spooler is None:
        return
    try:
        spooler.emit_event(
            event_type=event_data.get("event_type", "span"),
            component=event_data.get("component", "unknown"),
            operation=event_data.get("operation", event_data.get("name", "unknown")),
            data=event_data,
            # CORE-OBS-2: promote trace fields to top-level Stream fields
            name=event_data.get("name"),
            trace_id=event_data.get("trace_id"),
            span_id=event_data.get("span_id"),
            parent_span_id=event_data.get("parent_span_id"),
            duration_ms=event_data.get("duration_ms"),
        )
    except Exception as e:
        logger.debug(f"Failed to emit span: {e}")


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    component: Optional[str] = None,
) -> Generator[SpanContext, None, None]:
    """Create a traced span with automatic parent-child nesting.

    Args:
        name: Span name using dot notation (e.g., "pipeline.classify").
              First segment used as component if component not specified.
        attributes: Key-value pairs attached to the span event.
        component: Override component name (default: derived from name).

    Yields:
        SpanContext with mutable attributes dict. Add attributes during
        the span and they will be included in the emitted event.
    """
    if not _is_enabled():
        yield SpanContext()
        return

    parent = _current_span.get()
    trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
    span_id = uuid.uuid4().hex[:12]
    parent_span_id = parent.span_id if parent else None

    span = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        start_time=perf_counter(),
        attributes=dict(attributes) if attributes else {},
    )

    token = _current_span.set(span)
    error_info = None

    try:
        yield span
    except Exception as e:
        error_info = e
        raise
    finally:
        duration_ms = (perf_counter() - span.start_time) * 1000.0
        resolved_component = component or (name.split(".")[0] if name else "unknown")
        operation = name.split(".", 1)[1] if "." in name else name

        # Spread user attributes first, then structural fields on top
        # to prevent user-provided keys from clobbering span identity.
        event_data = {
            **span.attributes,
            "event_type": "span",
            "component": resolved_component,
            "operation": operation,
            "name": name,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "duration_ms": round(duration_ms, 2),
        }

        if error_info is not None:
            event_data["error"] = str(error_info)
            event_data["status"] = "error"

        try:
            _emit_span(event_data)
        except Exception:
            pass

        _current_span.reset(token)


def current_trace_id() -> Optional[str]:
    """Return the active trace_id, or None if no span is active."""
    span = _current_span.get()
    return span.trace_id if span else None


def current_span() -> Optional[SpanContext]:
    """Return the active SpanContext, or None if no span is active."""
    return _current_span.get()


def trace_log(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Emit a standalone log record with current span trace context.

    Unlike trace_span(), records an instantaneous occurrence with no duration
    and does not require a parent span.  Inherits trace_id / span_id from the
    active span when one exists, so log records are correlatable to the
    pipeline stage that emitted them.

    Args:
        name: Log name in dot notation (e.g., "pipeline.log").
              First segment used as component.
        attributes: Key-value payload for the log record.
    """
    if not _is_enabled():
        return
    span = _current_span.get()
    resolved_component = name.split(".")[0] if name else "unknown"
    operation = name.split(".", 1)[1] if "." in name else name
    event_data = {
        **(attributes or {}),
        "event_type": "log",
        "component": resolved_component,
        "operation": operation,
        "name": name,
        "trace_id": span.trace_id if span else "",
        "span_id": span.span_id if span else "",
        "parent_span_id": span.parent_span_id if span else None,
    }
    try:
        _emit_span(event_data)
    except Exception:
        pass


__all__ = ["trace_span", "trace_log", "current_trace_id", "current_span", "SpanContext"]
