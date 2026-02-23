"""Retrieval tracking — emits per-retrieval events to Redis Streams (CORE-EVO-ENH-2).

Accumulates retrieval signals from ``SmartMemory.get()`` and ``SmartMemory.search()``
into per-workspace Redis Streams so the ``RetrievalFlushConsumer`` (smart-memory-worker)
can aggregate them into ``metadata["retrieval__profile"]`` for use by
``RetrievalBasedStrengtheningEvolver``.

Stream: ``smartmemory:retrievals:{workspace_id}`` on Redis DB 2.
MAXLEN: ~ 50,000 events per workspace (approximate, O(1) amortized).
Active workspaces: ``smartmemory:active_workspaces`` set on Redis DB 2.

Design decisions:
- All emission functions are wrapped in try/except — observability failure must
  never affect ``get()`` or ``search()`` return values.
- One event per ``search()`` call (not per result) — results are JSON-packed
  in ``results_json`` to preserve per-result rank and score.
- ``retrieval_source`` context var carries the caller identity injected by
  the service layer; defaults to ``"internal"`` for SDK/Maya/direct callers.
- Lazy singleton Redis client follows the pattern from ``pipeline_producer.py``
  exactly — same env vars, same DB, same failure-suppression idiom.
"""

import hashlib
import json
import logging
import os
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variable — set by service layer, readable by core
# ---------------------------------------------------------------------------

retrieval_source: ContextVar[str] = ContextVar("retrieval_source", default="internal")

# ---------------------------------------------------------------------------
# Redis singleton — mirrors pipeline_producer.py exactly
# ---------------------------------------------------------------------------

_REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.getenv("REDIS_PORT", "9012"))
_REDIS_DB = int(os.getenv("WORKER_REDIS_DB", "2"))

_MAXLEN = 50_000
_ACTIVE_WORKSPACES_KEY = "smartmemory:active_workspaces"

_redis_client: Optional[object] = None
_redis_lock = threading.Lock()
_redis_failed = False


def _get_redis_client():
    """Return lazy singleton sync Redis client for retrieval event emission.

    Returns None if Redis is unavailable or the ``redis`` package is missing.
    All exceptions are caught and logged at DEBUG level.
    """
    global _redis_client, _redis_failed

    if _redis_failed:
        return None
    if _redis_client is not None:
        return _redis_client

    with _redis_lock:
        if _redis_failed:
            return None
        if _redis_client is not None:
            return _redis_client
        try:
            import redis as _redis_module

            client = _redis_module.Redis(
                host=_REDIS_HOST,
                port=_REDIS_PORT,
                db=_REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()
            _redis_client = client
            logger.debug(
                "retrieval_tracking: Redis client connected (%s:%d db=%d)",
                _REDIS_HOST,
                _REDIS_PORT,
                _REDIS_DB,
            )
        except Exception as exc:
            logger.debug(
                "retrieval_tracking: Redis unavailable — retrieval events will not be emitted: %s",
                exc,
            )
            _redis_failed = True

    return _redis_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace_id_from_scope(scope_provider) -> str:
    """Extract workspace_id from scope_provider, falling back to 'default'."""
    try:
        ctx = scope_provider.get_write_context()
        ws = ctx.get("workspace_id") or getattr(scope_provider, "workspace_id", None)
        return str(ws) if ws else "default"
    except Exception:
        return "default"


def _emit_to_stream(stream_key: str, workspace_id: str, fields: dict) -> None:
    """Write one event to a per-workspace stream and register the workspace.

    Args:
        stream_key: Full Redis key, e.g. ``smartmemory:retrievals:{ws_id}``.
        workspace_id: Workspace identifier — added to the active workspaces set.
        fields: Event fields (all values must be strings for Redis Streams).
    """
    client = _get_redis_client()
    if client is None:
        return
    client.xadd(stream_key, fields, maxlen=_MAXLEN, approximate=True)
    client.sadd(_ACTIVE_WORKSPACES_KEY, workspace_id)


# ---------------------------------------------------------------------------
# Public emission functions
# ---------------------------------------------------------------------------


def emit_retrieval_get(item: "MemoryItem", scope_provider) -> None:
    """Emit a ``get`` retrieval event for a single memory item.

    Called by ``SmartMemory.get()`` after a successful fetch. Fully wrapped in
    try/except — a Redis failure must never propagate to the caller.

    Args:
        item: The fetched ``MemoryItem`` (must not be None).
        scope_provider: ``ScopeProvider`` instance attached to the ``SmartMemory``
            object — used to extract the workspace_id.
    """
    try:
        workspace_id = _workspace_id_from_scope(scope_provider)
        stream_key = f"smartmemory:retrievals:{workspace_id}"
        fields = {
            "type": "get",
            "item_id": str(item.item_id or ""),
            "source": retrieval_source.get(),
            "ts": _utcnow_iso(),
        }
        _emit_to_stream(stream_key, workspace_id, fields)
        logger.debug("retrieval_tracking: emitted get event for item_id=%s", item.item_id)
    except Exception as exc:
        logger.debug("retrieval_tracking: emit_retrieval_get failed (non-fatal): %s", exc)


def emit_retrieval_search(results: list, query: str, scope_provider) -> None:
    """Emit a batched ``search`` retrieval event for a full search call.

    One event per search call (not per result). Per-result rank and score are
    JSON-packed into ``results_json``. Fully wrapped in try/except.

    Args:
        results: List of ``MemoryItem`` objects returned by ``search()``.
        query: The original search query string.
        scope_provider: ``ScopeProvider`` instance for workspace_id extraction.
    """
    try:
        workspace_id = _workspace_id_from_scope(scope_provider)
        stream_key = f"smartmemory:retrievals:{workspace_id}"

        query_hash = hashlib.md5((query or "").strip().lower().encode()).hexdigest()

        results_list = [
            {
                "item_id": str(getattr(r, "item_id", "") or ""),
                "rank": i + 1,
                "score": float(getattr(r, "score", 0) or 0),
            }
            for i, r in enumerate(results or [])
        ]

        fields = {
            "type": "search",
            "query_hash": query_hash,
            "results_json": json.dumps(results_list),
            "source": retrieval_source.get(),
            "ts": _utcnow_iso(),
        }
        _emit_to_stream(stream_key, workspace_id, fields)
        logger.debug(
            "retrieval_tracking: emitted search event for %d results (ws=%s)",
            len(results_list),
            workspace_id,
        )
    except Exception as exc:
        logger.debug("retrieval_tracking: emit_retrieval_search failed (non-fatal): %s", exc)
