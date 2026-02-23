"""
Centralized observability event emission utilities.
Defaults to Redis logical DB 1 and the stream 'smartmemory:events'.

This module is also the home for a light Redis Streams queue helper that can be
used for background job dispatch/consumption (e.g., enrichment/grounding).

Enhancements:
- Config-first controls for observability emission (enabled, sampling, retention)
- Enriched event envelope fields while preserving backward compatibility
  (keeps existing fields: event_type, component, operation, data)
"""

import json
import os
import random
import redis
import socket
import uuid
from typing import Any, Dict, Optional, Iterator, List

from smartmemory.utils import get_config, now


def _is_observability_enabled() -> bool:
    """Read observability toggle from env at call time.

    Default: enabled. Set SMARTMEMORY_OBSERVABILITY=false to disable.
    Reading at call time allows SmartMemory(observability=False) to set the env var
    after import and have it take effect immediately.
    """
    return os.getenv("SMARTMEMORY_OBSERVABILITY", "true").lower() in ("true", "1", "yes", "on")


# Defaults for observability stream
STREAM_NAME = "smartmemory:events"
REDIS_DB_EVENTS = 1


class EventSpooler:
    """Canonical event spooler for SmartMemory observability.
    Emits to a configurable Redis DB/stream (defaults preserved).
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        session_id: Optional[str] = None,
        *,
        stream_name: str = STREAM_NAME,
        db: int = REDIS_DB_EVENTS,
    ):
        config = get_config()

        # Simple direct access - crashes with clear error if config is missing
        redis_config = config.cache.redis

        # Resolve connection with explicit defaults
        host = redis_host or redis_config.host
        port = int(redis_port or redis_config.port)

        # Optional observability event settings
        obs_config = config.get("observability") or {}
        events_config = obs_config.get("events") or {}

        # Compute effective db (optional setting)
        eff_db = int(events_config.get("db", db)) if "db" in events_config else db

        # Compute effective stream name with optional namespace suffix
        base_stream = events_config.get("stream_name", stream_name) if "stream_name" in events_config else stream_name
        active_namespace = config.get("active_namespace")

        if active_namespace:
            eff_stream = f"{base_stream}:{active_namespace}"
        else:
            eff_stream = base_stream

        self.redis_client = redis.Redis(host=host, port=port, db=eff_db, decode_responses=True)
        self.stream_name = eff_stream
        self.db = eff_db
        self.namespace = active_namespace
        self.session_id = session_id or str(uuid.uuid4())

        # Observability controls
        self.obs_enabled: bool = bool(obs_config.get("enabled", True))
        # Feature flag: emit hierarchical keys in envelope (domain/category/action)
        self.unified_keys: bool = bool(obs_config.get("unified_keys", False))
        sampling_cfg = obs_config.get("event_sampling") or {} if "event_sampling" in obs_config else {}
        # Provide sensible defaults if not configured
        self.sampling: Dict[str, float] = {
            "system_health": 1.0,
            "graph_stats_update": 1.0,
            "performance_summary": 1.0,
            "performance_metrics": 0.5,
            "vector_operation": 0.2,
            "background_process": 1.0,
            "job_lifecycle": 1.0,
            **({k: float(v) for k, v in sampling_cfg.items()} if isinstance(sampling_cfg, dict) else {}),
        }
        retention_cfg = obs_config.get("retention") or {} if "retention" in obs_config else {}
        self.maxlen: Optional[int] = None
        try:
            ml = retention_cfg.get("maxlen") if "maxlen" in retention_cfg else None
            if ml is not None:
                self.maxlen = int(ml)
            else:
                self.maxlen = 100_000
        except Exception:
            self.maxlen = 100_000
        # Static tags to enrich events
        self.static_tags = obs_config.get("tags") or {} if "tags" in obs_config else {}

    def emit_event(
        self,
        event_type: str,
        component: str,
        operation: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ):
        """Emit an event to the configured Redis Stream.

        Backward compatible fields preserved: event_type, component, operation, data
        Enriched envelope fields added: event_version, timestamp (UTC), namespace, host, pid, tags
        Honors config-driven enabled flag, per-event sampling, and MAXLEN trimming.

        CORE-OBS-2: trace field kwargs become top-level Redis Stream fields when
        provided. Existing callers that don't pass them are unaffected.
        """
        if not self.obs_enabled:
            return

        # Per-event sampling
        rate = float(self.sampling.get(event_type, 1.0))
        if rate < 1.0 and random.random() > max(0.0, min(rate, 1.0)):
            return

        # Build envelope
        payload = data or {}
        try:
            data_json = json.dumps(payload)
        except Exception:
            # Ensure we never fail emission due to serialization issues
            data_json = json.dumps({"_nonserializable": True})

        event_data: Dict[str, Any] = {
            # Backward-compatible fields
            "event_type": event_type,
            "component": component,
            "operation": operation,
            "data": data_json,
            "session_id": self.session_id,
            "timestamp": now().isoformat(),
            # Enriched envelope
            "event_version": 1,
            "namespace": self.namespace or "",
            "stream_name": self.stream_name,
            "db": self.db,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        # CORE-OBS-2: promote trace fields to top-level Stream fields
        if name is not None:
            event_data["name"] = name
        if trace_id is not None:
            event_data["trace_id"] = trace_id
        if span_id is not None:
            event_data["span_id"] = span_id
        if parent_span_id is not None:
            event_data["parent_span_id"] = str(parent_span_id)
        if duration_ms is not None:
            event_data["duration_ms"] = str(round(duration_ms, 2))
        # Optionally include hierarchical fields when enabled and provided
        if self.unified_keys and isinstance(metadata, dict):
            try:
                d = metadata.get("domain")
                c = metadata.get("category")
                a = metadata.get("action")
                if isinstance(d, str) and isinstance(c, str) and isinstance(a, str):
                    event_data["domain"] = d
                    event_data["category"] = c
                    event_data["action"] = a
            except Exception:
                pass
        # Optional static tags (flattened into top-level under 'tags')
        if isinstance(self.static_tags, dict) and self.static_tags:
            try:
                event_data["tags"] = json.dumps(self.static_tags)
            except Exception:
                pass

        try:
            # Apply MAXLEN trimming if configured
            if self.maxlen:
                self.redis_client.xadd(self.stream_name, event_data, maxlen=self.maxlen, approximate=True)
            else:
                self.redis_client.xadd(self.stream_name, event_data)
        except Exception:
            # Observability must never break core flows
            pass


def emit_event(
    event_type: str,
    component: str,
    operation: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    stream_name: str = STREAM_NAME,
    db: int = REDIS_DB_EVENTS,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Functional helper to emit a single event without managing an EventSpooler instance.

    Auto-attaches the current trace_id from the active span context (if any)
    so that all events emitted within a pipeline trace are groupable.
    """
    if not _is_observability_enabled():
        return
    # Import here to avoid circular dependency (tracing imports events)
    from smartmemory.observability.tracing import current_trace_id

    spooler = EventSpooler(
        redis_host=redis_host,
        redis_port=redis_port,
        session_id=session_id,
        stream_name=stream_name,
        db=db,
    )
    spooler.emit_event(
        event_type,
        component,
        operation,
        data,
        metadata,
        trace_id=current_trace_id(),
    )


class EventStream:
    """Canonical read-side interface for observability events.
    Reads from a configurable Redis DB/stream (defaults preserved).
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        *,
        stream_name: str = STREAM_NAME,
        db: int = REDIS_DB_EVENTS,
    ):
        config = get_config()

        # Simple direct access - crashes with clear error if config is missing
        redis_config = config.cache.redis

        # Resolve connection with explicit defaults
        host = redis_host or redis_config.host
        port = int(redis_port or redis_config.port)

        # Optional observability event settings
        obs_config = config.get("observability") or {}
        events_config = obs_config.get("events") or {}

        # Compute effective db (optional setting)
        eff_db = int(events_config.get("db", db)) if "db" in events_config else db

        # Compute effective stream name with optional namespace suffix
        base_stream = events_config.get("stream_name", stream_name) if "stream_name" in events_config else stream_name
        active_namespace = config.get("active_namespace")

        if active_namespace:
            eff_stream = f"{base_stream}:{active_namespace}"
        else:
            eff_stream = base_stream

        self.redis_client = redis.Redis(host=host, port=port, db=eff_db, decode_responses=True)
        self.stream_name = eff_stream

    @staticmethod
    def _parse_event(message_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize structure to match emitter schema
        evt: Dict[str, Any] = {
            "id": message_id,
            "timestamp": fields.get("timestamp"),
            "session_id": fields.get("session_id"),
            "event_type": fields.get("event_type"),
            "component": fields.get("component"),
            "operation": fields.get("operation"),
            "data": {},
        }
        data_raw = fields.get("data")
        if isinstance(data_raw, str):
            try:
                evt["data"] = json.loads(data_raw)
            except json.JSONDecodeError:
                evt["data"] = {"_raw": data_raw}
        elif isinstance(data_raw, dict):
            evt["data"] = data_raw
        # Preserve any extra fields not in the standard schema
        for k, v in fields.items():
            if k not in {"timestamp", "session_id", "event_type", "component", "operation", "data"}:
                evt.setdefault("extras", {})[k] = v
        return evt

    def read_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read all events, optionally limited to the most recent N."""
        try:
            messages = self.redis_client.xrange(self.stream_name, min="-", max="+")
            events = [self._parse_event(mid, fields) for mid, fields in messages]
            return events[-limit:] if limit else events
        except Exception:
            return []

    def read_since(self, last_id: str) -> List[Dict[str, Any]]:
        """Read events strictly after the provided message id."""
        try:
            messages = self.redis_client.xrange(self.stream_name, min=f"({last_id}", max="+")
            return [self._parse_event(mid, fields) for mid, fields in messages]
        except Exception:
            return []

    def stream_new(self, last_id: str = "$", block_ms: int = 1000, count: int = 10) -> Iterator[Dict[str, Any]]:
        """Yield new events as they arrive. Use last_id to resume from a position.
        Note: '$' starts from only new messages.
        """
        current_id = last_id
        while True:
            try:
                result = self.redis_client.xread({self.stream_name: current_id}, count=count, block=block_ms)
                if result:
                    _, messages = result[0]
                    for mid, fields in messages:
                        yield self._parse_event(mid, fields)
                    current_id = messages[-1][0]
            except Exception:
                # Back off briefly on errors
                break

    def clear(self) -> bool:
        """Delete the stream. For test cleanup only."""
        try:
            self.redis_client.delete(self.stream_name)
            return True
        except Exception:
            return False


class RedisStreamQueue:
    """Light helper for Redis Streams-based job queues with consumer groups.

    Usage:
      - Producer: q = RedisStreamQueue(stream_name="smartmemory:jobs:enrich"); q.enqueue(payload)
      - Consumer: q = RedisStreamQueue(stream_name, group="enrich-workers", consumer="worker-1");
                  q.ensure_group(); for mid, fields in q.read_group(): ...; q.ack(mid)
    """

    def __init__(
        self,
        *,
        stream_name: Optional[str] = None,
        group: Optional[str] = None,
        consumer: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        db: Optional[int] = None,
    ) -> None:
        config = get_config()

        # Simple direct access - crashes with clear error if config is missing
        redis_config = config.cache.redis

        # Resolve connection
        host = redis_host or redis_config.host
        port = int(redis_port or redis_config.port)

        # Resolve stream name
        if stream_name is None:
            # Use prefix + kind if available; default to generic jobs stream
            prefix = config.get("background") or {}.get("queues") or {}.get("stream_prefix", "smartmemory:jobs")
            # If no specific kind is provided here, rely on helpers or caller to set
            base_stream = f"{prefix}:default"
        else:
            base_stream = stream_name
        ns = config.get("active_namespace")
        eff_stream = f"{base_stream}:{ns}" if ns else base_stream
        self.redis = redis.Redis(host=host, port=port, db=db or 2, decode_responses=True)
        self.stream_name = eff_stream
        self.group = group
        self.consumer = consumer or f"consumer-{uuid.uuid4().hex[:6]}"

    def ensure_group(self) -> None:
        if not self.group:
            return
        try:
            # MKSTREAM creates stream if absent; $ starts from new messages
            self.redis.xgroup_create(self.stream_name, self.group, id="$", mkstream=True)
        except redis.ResponseError as e:
            # Group already exists -> ignore
            if "BUSYGROUP" not in str(e):
                raise

    def enqueue(self, payload: Dict[str, Any]) -> str:
        data = {"payload": json.dumps(payload)}
        return self.redis.xadd(self.stream_name, data)

    def read_group(self, *, block_ms: int = 1000, count: int = 10) -> List[tuple[str, Dict[str, Any]]]:
        if not self.group:
            raise ValueError("Consumer group is required for read_group")
        res = self.redis.xreadgroup(self.group, self.consumer, {self.stream_name: ">"}, count=count, block=block_ms)
        if not res:
            return []
        _, messages = res[0]
        return messages

    def ack(self, message_id: str) -> int:
        if not self.group:
            raise ValueError("Consumer group is required for ack")
        return self.redis.xack(self.stream_name, self.group, message_id)

    def move_to_dlq(self, message_id: str, fields: Dict[str, Any], reason: str = "error") -> str:
        dlq_stream = f"{self.stream_name}:dlq"
        payload = fields.copy()
        payload["error_reason"] = reason
        return self.redis.xadd(dlq_stream, payload)

    # ---- Helpers for config-derived queues ----
    @classmethod
    def _compute_stream(cls, kind: str) -> tuple[str, int]:
        config = get_config()
        redis_config = config.cache.redis
        bg_cfg = config.get("background") or {}
        queues_cfg = bg_cfg.get("queues") or {}
        # DB
        db = int(queues_cfg.get("db", 2))
        # Build base stream
        prefix = queues_cfg.get("stream_prefix", "smartmemory:jobs")
        # Allow explicit streams per kind
        kind_key = f"{kind}_stream"
        suffix = queues_cfg.get(kind_key, kind)
        base_stream = f"{prefix}:{suffix}"
        # Namespace suffix
        ns = config.get("active_namespace")
        stream = f"{base_stream}:{ns}" if ns else base_stream
        return stream, db

    @classmethod
    def for_enrich(cls, group: Optional[str] = None, consumer: Optional[str] = None) -> "RedisStreamQueue":
        stream, db = cls._compute_stream("enrich")
        return cls(stream_name=stream, group=group or "enrich-workers", consumer=consumer, db=db)

    @classmethod
    def for_promote(cls, group: Optional[str] = None, consumer: Optional[str] = None) -> "RedisStreamQueue":
        stream, db = cls._compute_stream("promote")
        return cls(stream_name=stream, group=group or "promote-workers", consumer=consumer, db=db)
