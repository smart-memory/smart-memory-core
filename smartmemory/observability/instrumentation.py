"""
Instrumentation helpers to make observability emissions concise and consistent.

Features:
- Global observability context via contextvars (trace_id, request_id, user_id, etc.)
- Decorator for graph mutation stats: @emit_graph_stats(operation, component='graph', extra_fn=None)
  - Captures counts before/after a method call on SmartGraph
  - Computes deltas and emits a graph_stats_update event
  - Merges in global context and optional per-call extra metadata

Notes:
- Designed to be best-effort: all errors are swallowed to avoid impacting core flows.
- Uses backend fast count helpers when available; falls back to safe counting.
"""

import contextvars
import functools
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union


def _is_observability_enabled() -> bool:
    """Read observability toggle from env at call time.

    Default: enabled. Set SMARTMEMORY_OBSERVABILITY=false to disable.
    Reading at call time allows SmartMemory(observability=False) to set the env var
    after import and have it take effect immediately.
    """
    return os.getenv("SMARTMEMORY_OBSERVABILITY", "true").lower() in ("true", "1", "yes", "on")


# ---- Global context -------------------------------------------------------

_obs_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("obs_context", default={})


def set_obs_context(ctx: Dict[str, Any]) -> None:
    """Replace the current observability context with ctx.

    .. deprecated::
        Use ``trace_span`` attributes from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "set_obs_context is deprecated, use trace_span attributes from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(ctx, dict):
        return
    _obs_context.set(dict(ctx))


def update_obs_context(values: Dict[str, Any]) -> None:
    """Merge values into the current observability context.

    .. deprecated::
        Use ``trace_span`` attributes from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "update_obs_context is deprecated, use trace_span attributes from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(values, dict):
        return
    current = dict(_obs_context.get() or {})
    current.update(values)
    _obs_context.set(current)


def get_obs_context() -> Dict[str, Any]:
    """Get a shallow copy of the current observability context.

    .. deprecated::
        Use ``trace_span`` attributes from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "get_obs_context is deprecated, use trace_span attributes from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dict(_obs_context.get() or {})


def clear_obs_context() -> None:
    """Clear the current observability context.

    .. deprecated::
        Use ``trace_span`` attributes from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "clear_obs_context is deprecated, use trace_span attributes from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )
    _obs_context.set({})


# ---- Context decorator ---------------------------------------------------


def with_obs_context(
    ctx_or_fn: Optional[Union[Dict[str, Any], Callable[..., Optional[Dict[str, Any]]]]] = None,
    *,
    merge: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to run a function with an updated observability context.

    Parameters:
    - ctx_or_fn: a dict to merge into the current context, or a callable
      that receives (*args, **kwargs) and returns a dict of values to add.
    - merge: if True (default), merge with existing context; if False, replace
      the context for the duration of the function call.

    Works with both synchronous and asynchronous functions and always restores
    the previous context state.

    .. deprecated::
        Use ``trace_span`` attributes from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "with_obs_context is deprecated, use trace_span attributes from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )

    def _resolve(args: tuple, kwargs: dict) -> Dict[str, Any]:
        try:
            if callable(ctx_or_fn):
                data = ctx_or_fn(*args, **kwargs)
                return dict(data or {})
            if isinstance(ctx_or_fn, dict):
                return dict(ctx_or_fn)
        except Exception:
            return {}
        return {}

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Since we're now fully synchronous, we only need the sync wrapper
        @functools.wraps(fn)
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            base = get_obs_context()
            new = _resolve(args, kwargs)
            effective = {**base, **new} if merge else new
            token = _obs_context.set(effective)
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    _obs_context.reset(token)
                except Exception:
                    _obs_context.set(base)

        return _wrap

    return _decorator


# ---- Concise emit helpers -------------------------------------------------


def emit_ctx(
    event_type: str,
    *,
    component: str,
    operation: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    include_context: bool = True,
    key: Optional[str] = None,
) -> None:
    """Emit an event with optional automatic context merging.

    Parameters:
    - event_type/component/operation: legacy metadata preserved for compatibility
    - key: optional hierarchical key "domain.category.action"; when provided, domain/category/action
      are included in the envelope (behind the unified_keys config flag)

    Usage:
        emit_ctx("vector_operation", component="vector_store", operation="add", data={"count": 3})
        emit_ctx("vector_operation", component="vector_store", operation="add", data={"count": 3}, key="vector.operation.add")

    .. deprecated::
        Use ``trace_span`` from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "emit_ctx is deprecated, use trace_span from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )
    payload: Dict[str, Any] = dict(data or {})
    if include_context:
        ctx = get_obs_context()
        if ctx:
            payload["context"] = ctx

    metadata: Optional[Dict[str, Any]] = None
    if isinstance(key, str) and key.count(".") == 2:
        try:
            domain, category, action = key.split(".", 2)
            if domain and category and action:
                metadata = {"domain": domain, "category": category, "action": action}
        except Exception:
            metadata = None
    if not _is_observability_enabled():
        return
    try:
        from smartmemory.observability.events import emit_event

        emit_event(event_type=event_type, component=component, operation=operation, data=payload, metadata=metadata)
    except Exception:
        # best-effort; never raise
        pass


def make_emitter(
    *,
    component: str,
    default_type: Optional[str] = None,
    default_operation: Optional[str] = None,
    include_context: bool = True,
    default_key: Optional[str] = None,
) -> Callable[[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str]], None]:
    """Create a pre-configured emitter to reduce repetition at call sites.

    Example:
        vec_emit = make_emitter(component="vector", default_type="vector_op")
        vec_emit("vector_op", "upsert", {"count": 1})
        # or rely on defaults:
        vec_emit(None, None, {"count": 1})

    .. deprecated::
        Use ``trace_span`` from ``smartmemory.observability.tracing`` instead.
    """
    warnings.warn(
        "make_emitter is deprecated, use trace_span from smartmemory.observability.tracing instead",
        DeprecationWarning,
        stacklevel=2,
    )

    def _emit(
        et: Optional[str] = None,
        op: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> None:
        et_final = et or default_type or "custom_event"
        op_final = op or default_operation
        emit_ctx(
            et_final,
            component=component,
            operation=op_final,
            data=data,
            include_context=include_context,
            key=(key or default_key),
        )

    return _emit


def emit_http_perf(request: Any, status_code: Optional[int], start: Any, *, component: str = "service") -> None:
    """Emit standard HTTP performance metrics given a FastAPI/Starlette request, status, and start time.

    Accesses request.method, request.url.path, and request.url.query via getattr to avoid hard dependency.
    """
    try:
        from time import perf_counter as _pc
    except Exception:
        _pc = None  # type: ignore

    method = getattr(request, "method", None)
    url = getattr(request, "url", None)
    path = getattr(url, "path", None) if url is not None else None
    query = str(getattr(url, "query", "")) if url is not None else None

    duration_ms = None
    if _pc is not None and start is not None:
        try:
            duration_ms = (_pc() - start) * 1000.0
        except Exception:
            duration_ms = None

    op = f"{method} {path}" if method and path else None
    emit_ctx(
        "performance_metrics",
        component=component,
        operation=op,
        data={
            "method": method,
            "path": path,
            "query": query,
            "status_code": status_code,
            "duration_ms": duration_ms,
        },
        key="api.request.handle",
    )


def emit_system_health(operation: str, *, data: Optional[Dict[str, Any]] = None, component: str = "service") -> None:
    """Emit a system_health event with minimal boilerplate."""
    # Hierarchical key follows system.health.<operation>
    key = f"system.health.{operation}" if isinstance(operation, str) and operation else None
    emit_ctx("system_health", component=component, operation=operation, data=data or {}, key=key)


# ---- Graph stats decorator ------------------------------------------------


def _safe_get_counts(graph_self: Any) -> Tuple[int, int]:
    """Return (node_count, edge_count) best-effort from a SmartGraph-like instance.

    Prefers backend.get_counts() or backend.get_node_count()/get_edge_count().
    Falls back to counting via high-level APIs if necessary.
    """
    try:
        backend = getattr(graph_self, "_backend", None)
        if backend is not None:
            # Prefer fast combined counts when available
            if hasattr(backend, "get_counts") and callable(backend.get_counts):
                counts = backend.get_counts()
                if isinstance(counts, dict):
                    return int(counts.get("nodes", 0)), int(counts.get("edges", 0))
            # Fallback to individual fast getters
            nodes = int(backend.get_node_count()) if hasattr(backend, "get_node_count") else None
            edges = int(backend.get_edge_count()) if hasattr(backend, "get_edge_count") else None
            if nodes is not None and edges is not None:
                return nodes, edges
    except Exception:
        pass

    # Last-resort safe fallbacks
    nodes = 0
    edges = 0
    try:
        # get_all_nodes is expected on SmartGraph
        all_nodes = graph_self.get_all_nodes() if hasattr(graph_self, "get_all_nodes") else []
        nodes = len(all_nodes) if all_nodes is not None else 0
    except Exception:
        nodes = 0
    try:
        # If we can cheaply retrieve edge counts in aggregate, do so; otherwise skip expensive traverse
        backend = getattr(graph_self, "_backend", None)
        if backend is not None and hasattr(backend, "get_edge_count"):
            edges = int(backend.get_edge_count())
        else:
            # Conservative: try to approximate by summing outgoing edges if available
            edges = 0
            if hasattr(graph_self, "get_edges_for_node") and callable(graph_self.get_edges_for_node):
                # Only iterate a subset if node count is large to avoid O(N^2); here we just skip
                pass
    except Exception:
        edges = 0
    return nodes, edges


def emit_graph_stats(
    operation: str,
    *,
    component: str = "graph",
    extra_fn: Optional[Callable[[Any, tuple, dict, Any], Optional[Dict[str, Any]]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for SmartGraph mutation methods to emit graph_stats_update concisely.

    Parameters:
    - operation: logical operation label, e.g., 'add_node', 'remove_edge', 'clear'.
    - component: component name for the event (default 'graph').
    - extra_fn: optional callback (self, args, kwargs, result) -> dict to add extra payload fields.

    Usage:
      @emit_graph_stats('add_edge')
      def add_edge(self, src, dst, rel, properties=None):
          return self._backend.add_edge(src, dst, rel, properties or {})
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                before_nodes, before_edges = _safe_get_counts(self)
            except Exception:
                before_nodes, before_edges = 0, 0

            result: Any = fn(self, *args, **kwargs)

            try:
                after_nodes, after_edges = _safe_get_counts(self)
                delta_nodes = after_nodes - before_nodes
                delta_edges = after_edges - before_edges

                data: Dict[str, Any] = {
                    "backend": type(getattr(self, "_backend", None)).__name__
                    if hasattr(self, "_backend")
                    else "UnknownBackend",
                    "node_count": after_nodes,
                    "edge_count": after_edges,
                    "delta_nodes": delta_nodes,
                    "delta_edges": delta_edges,
                }

                if callable(extra_fn):
                    try:
                        extra = extra_fn(self, args, kwargs, result)
                        if isinstance(extra, dict):
                            data.update(extra)
                    except Exception:
                        pass

                ctx = get_obs_context()
                if ctx:
                    data["context"] = ctx

                emit_ctx(
                    "graph_stats_update",
                    component=component,
                    operation=operation,
                    data=data,
                    key="graph.stats.update",
                )
            except Exception:
                # Never let observability break core logic
                pass

            return result

        return _wrapper

    return _decorator


__all__ = [
    "set_obs_context",
    "update_obs_context",
    "get_obs_context",
    "clear_obs_context",
    "with_obs_context",
    "emit_ctx",
    "make_emitter",
    "emit_http_perf",
    "emit_system_health",
    "emit_graph_stats",
]
