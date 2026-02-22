"""Pipeline stream producer — emits post-link events to ``smartmemory:pipeline`` (CORE-BG-1).

Called from ``SmartMemory.ingest()`` after the pipeline run completes (post-link stage).
Fire-and-forget: all exceptions are swallowed so observability failures never break ingestion.

Stream: ``smartmemory:pipeline`` on Redis DB 2.
MAXLEN: ~ 10,000 events (approximate trimming, O(1) amortized).

Event fields (all strings, as required by Redis Streams):
    item_id:       Memory item ID returned by the pipeline
    workspace_id:  Workspace identifier from PipelineConfig (empty string if unknown)
    memory_type:   Memory type string (e.g. "semantic", "episodic")
    ts:            ISO 8601 UTC timestamp of emission
"""

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Redis configuration — same env vars as the rest of the project
_REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.getenv("REDIS_PORT", "9012"))
_REDIS_DB = int(os.getenv("WORKER_REDIS_DB", "2"))

# Stream parameters
_STREAM_KEY = "smartmemory:pipeline"
_MAXLEN = 10_000

# Lazy singleton Redis client with thread lock
_redis_client: Optional[object] = None
_redis_lock = threading.Lock()
_redis_failed = False  # Once connection fails, stop retrying in-process


def _get_redis_client():
    """Return a lazy singleton sync Redis client for pipeline event emission.

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
            # Ping to verify connection is alive
            client.ping()
            _redis_client = client
            logger.debug("pipeline_producer: Redis client connected (%s:%d db=%d)", _REDIS_HOST, _REDIS_PORT, _REDIS_DB)
        except Exception as exc:
            logger.debug("pipeline_producer: Redis unavailable — pipeline events will not be emitted: %s", exc)
            _redis_failed = True

    return _redis_client


def emit_pipeline_event(
    item_id: str,
    workspace_id: Optional[str],
    memory_type: str,
) -> None:
    """Emit a pipeline event to ``smartmemory:pipeline`` after the link stage.

    This is a fire-and-forget call. All exceptions are swallowed — pipeline event
    emission must never break ``ingest()``.

    Args:
        item_id: The memory item ID returned by the pipeline StoreStage.
        workspace_id: Workspace identifier from PipelineConfig. May be None.
        memory_type: Memory type string (e.g. ``"semantic"``).
    """
    try:
        client = _get_redis_client()
        if client is None:
            return

        fields = {
            "item_id": str(item_id or ""),
            "workspace_id": str(workspace_id or ""),
            "memory_type": str(memory_type or "semantic"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        client.xadd(_STREAM_KEY, fields, maxlen=_MAXLEN, approximate=True)
        logger.debug("pipeline_producer: emitted event for item_id=%s", item_id)

    except Exception as exc:
        # Non-fatal: observability failure must never break ingestion
        logger.debug("pipeline_producer: failed to emit event (non-fatal): %s", exc)
