"""Pipeline metrics consumer — aggregates stage events into time buckets.

Reads stage_complete / pipeline_complete events from the metrics Redis
Stream and rolls them into 5-minute buckets stored in Redis Hashes for
fast dashboard reads.

Bucket key format:
    smartmemory:metrics:agg:{metric_type}:{bucket_ts}

where bucket_ts is the Unix timestamp (floored to BUCKET_SIZE_SECONDS).

Usage:
    consumer = MetricsConsumer()
    processed = consumer.run()            # process all pending events
    data = consumer.get_aggregated("stage", hours=6)  # read for dashboards
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis

from smartmemory.utils import get_config

METRICS_STREAM = "smartmemory:metrics:pipeline"

logger = logging.getLogger(__name__)

BUCKET_SIZE_SECONDS = 300  # 5 minutes
RETENTION_HOURS = 24
AGG_KEY_PREFIX = "smartmemory:metrics:agg"


class MetricsConsumer:
    """Consume pipeline metric events and aggregate into time buckets."""

    def __init__(
        self,
        redis_client: redis.Redis | None = None,  # annotation is a deferred string (PEP 563)
        stream_name: str = METRICS_STREAM,
    ):
        self._redis_available = False
        self.stream_name = stream_name
        self._last_id = "0-0"

        if redis_client is not None:
            self.redis = redis_client
            self._redis_available = True
        else:
            try:
                import redis as _redis
                config = get_config()
                rc = config.cache.redis
                self.redis = _redis.Redis(
                    host=rc.host, port=int(rc.port), db=1, decode_responses=True
                )
                self._redis_available = True
            except (ImportError, Exception) as exc:
                logger.debug("Redis unavailable for metrics consumer: %s", exc)
                self.redis = None

        if self._redis_available:
            self._restore_cursor()

    # -- cursor persistence --------------------------------------------------

    def _cursor_key(self) -> str:
        return f"{AGG_KEY_PREFIX}:cursor:{self.stream_name}"

    def _restore_cursor(self) -> None:
        """Resume from last acknowledged position."""
        try:
            saved = self.redis.get(self._cursor_key())
            if saved:
                self._last_id = saved
        except Exception:
            pass

    def _save_cursor(self) -> None:
        try:
            self.redis.set(self._cursor_key(), self._last_id)
        except Exception:
            pass

    # -- bucket helpers -------------------------------------------------------

    @staticmethod
    def _bucket_ts(epoch: float) -> int:
        """Floor epoch to the nearest bucket boundary."""
        return int(epoch // BUCKET_SIZE_SECONDS) * BUCKET_SIZE_SECONDS

    @staticmethod
    def _bucket_key(metric_type: str, bucket_ts: int) -> str:
        return f"{AGG_KEY_PREFIX}:{metric_type}:{bucket_ts}"

    def _ttl_seconds(self) -> int:
        return RETENTION_HOURS * 3600 + BUCKET_SIZE_SECONDS

    # -- aggregation ----------------------------------------------------------

    def _aggregate_stage_event(self, data: dict[str, Any], ts_epoch: float) -> None:
        """Aggregate a single stage_complete event into its bucket."""
        stage = data.get("stage_name") or data.get("stage", "unknown")
        elapsed = float(data.get("elapsed_ms", 0))
        status = data.get("status", "success")
        entity_count = int(data.get("entity_count", 0))
        relation_count = int(data.get("relation_count", 0))

        bucket = self._bucket_ts(ts_epoch)
        key = self._bucket_key("stage", bucket)
        ttl = self._ttl_seconds()

        pipe = self.redis.pipeline(transaction=False)
        pipe.hincrby(key, f"{stage}:count", 1)
        pipe.hincrbyfloat(key, f"{stage}:total_ms", elapsed)
        pipe.hincrbyfloat(key, f"{stage}:max_ms", 0)  # placeholder, updated below
        pipe.hincrby(key, f"{stage}:entities", entity_count)
        pipe.hincrby(key, f"{stage}:relations", relation_count)
        if status == "error":
            pipe.hincrby(key, f"{stage}:errors", 1)
        pipe.expire(key, ttl)
        pipe.execute()

        # Update max (need read-then-set, but keep non-critical)
        try:
            cur_max = float(self.redis.hget(key, f"{stage}:max_ms") or 0)
            if elapsed > cur_max:
                self.redis.hset(key, f"{stage}:max_ms", str(elapsed))
        except Exception:
            pass

    def _aggregate_pipeline_event(self, data: dict[str, Any], ts_epoch: float) -> None:
        """Aggregate a pipeline_complete event."""
        total_ms = float(data.get("total_ms", 0))
        stages_completed = int(data.get("stages_completed", 0))
        entity_count = int(data.get("entity_count", 0))
        relation_count = int(data.get("relation_count", 0))

        bucket = self._bucket_ts(ts_epoch)
        key = self._bucket_key("pipeline", bucket)
        ttl = self._ttl_seconds()

        pipe = self.redis.pipeline(transaction=False)
        pipe.hincrby(key, "count", 1)
        pipe.hincrbyfloat(key, "total_ms", total_ms)
        pipe.hincrby(key, "total_entities", entity_count)
        pipe.hincrby(key, "total_relations", relation_count)
        pipe.hincrby(key, "total_stages", stages_completed)
        pipe.expire(key, ttl)
        pipe.execute()

    # -- run loop -------------------------------------------------------------

    def run(self, max_iterations: int | None = None) -> int:
        """Process pending metric events. Returns count processed.

        Args:
            max_iterations: Cap on events to process (None = all pending).
        """
        if not self._redis_available:
            return 0
        processed = 0
        batch_size = 100

        while True:
            if max_iterations is not None and processed >= max_iterations:
                break

            try:
                messages = self.redis.xrange(
                    self.stream_name, min=f"({self._last_id}", max="+", count=batch_size
                )
            except Exception:
                logger.exception("Failed to read metrics stream")
                break

            if not messages:
                break

            for msg_id, fields in messages:
                try:
                    self._process_message(fields)
                except Exception:
                    logger.debug("Skipping malformed metric event %s", msg_id)

                self._last_id = msg_id
                processed += 1

                if max_iterations is not None and processed >= max_iterations:
                    break

        self._save_cursor()
        return processed

    def _process_message(self, fields: dict[str, str]) -> None:
        """Route a single message to the appropriate aggregator."""
        event_type = fields.get("event_type", "")
        ts_str = fields.get("timestamp", "")

        # Parse timestamp to epoch
        ts_epoch = self._parse_timestamp(ts_str)

        # Data may be JSON-encoded or flat in fields
        raw_data = fields.get("data")
        if raw_data:
            try:
                data = json.loads(raw_data)
            except (json.JSONDecodeError, TypeError):
                data = dict(fields)
        else:
            data = dict(fields)

        if event_type == "stage_complete":
            self._aggregate_stage_event(data, ts_epoch)
        elif event_type == "pipeline_complete":
            self._aggregate_pipeline_event(data, ts_epoch)

    @staticmethod
    def _parse_timestamp(ts_str: str) -> float:
        """Convert ISO timestamp to epoch seconds. Falls back to now()."""
        if not ts_str:
            return time.time()
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return time.time()

    # -- read pre-aggregated data ---------------------------------------------

    def get_aggregated(self, metric_type: str, hours: int = 1) -> list[dict[str, Any]]:
        """Read pre-aggregated buckets for dashboard consumption.

        Args:
            metric_type: "stage" or "pipeline".
            hours: How many hours of history to return.

        Returns:
            List of bucket dicts sorted by timestamp ascending.
        """
        if not self._redis_available:
            return []
        now_ts = time.time()
        start_bucket = self._bucket_ts(now_ts - hours * 3600)
        end_bucket = self._bucket_ts(now_ts)

        results: list[dict[str, Any]] = []
        bucket = start_bucket

        while bucket <= end_bucket:
            key = self._bucket_key(metric_type, bucket)
            try:
                raw = self.redis.hgetall(key)
            except Exception:
                raw = {}

            if raw:
                parsed = self._parse_bucket(metric_type, bucket, raw)
                results.extend(parsed)

            bucket += BUCKET_SIZE_SECONDS

        return results

    def _parse_bucket(
        self, metric_type: str, bucket_ts: int, raw: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Parse a Redis hash bucket into structured dicts."""
        from datetime import datetime, timezone

        ts_iso = datetime.fromtimestamp(bucket_ts, tz=timezone.utc).isoformat()

        if metric_type == "stage":
            return self._parse_stage_bucket(ts_iso, raw)
        elif metric_type == "pipeline":
            return [self._parse_pipeline_bucket(ts_iso, raw)]
        return []

    @staticmethod
    def _parse_stage_bucket(ts_iso: str, raw: dict[str, str]) -> list[dict[str, Any]]:
        """Parse stage bucket hash into per-stage records."""
        stages: dict[str, dict[str, Any]] = {}

        for field_key, value in raw.items():
            parts = field_key.rsplit(":", 1)
            if len(parts) != 2:
                continue
            stage_name, metric = parts
            stages.setdefault(stage_name, {"stage": stage_name, "timestamp": ts_iso})
            try:
                stages[stage_name][metric] = (
                    float(value) if "." in str(value) else int(value)
                )
            except (ValueError, TypeError):
                stages[stage_name][metric] = value

        # Compute derived fields
        results = []
        for stage_data in stages.values():
            count = stage_data.get("count", 0)
            total_ms = stage_data.get("total_ms", 0.0)
            max_ms = stage_data.get("max_ms", 0.0)
            errors = stage_data.get("errors", 0)

            avg_ms = round(total_ms / count, 2) if count > 0 else 0.0
            # Approximate p95 from max (true p95 needs reservoir sampling)
            p95_ms = round(max_ms * 0.95, 2) if max_ms else 0.0
            error_rate = round(errors / count, 4) if count > 0 else 0.0

            results.append(
                {
                    "timestamp": stage_data["timestamp"],
                    "stage": stage_data["stage"],
                    "count": count,
                    "avg_latency_ms": avg_ms,
                    "p95_latency_ms": p95_ms,
                    "error_count": errors,
                    "error_rate": error_rate,
                    "total_entities": stage_data.get("entities", 0),
                    "total_relations": stage_data.get("relations", 0),
                }
            )

        return results

    @staticmethod
    def _parse_pipeline_bucket(ts_iso: str, raw: dict[str, str]) -> dict[str, Any]:
        """Parse pipeline bucket hash into a summary record."""
        count = int(raw.get("count", 0))
        total_ms = float(raw.get("total_ms", 0))
        return {
            "timestamp": ts_iso,
            "count": count,
            "avg_total_ms": round(total_ms / count, 2) if count > 0 else 0.0,
            "total_entities": int(raw.get("total_entities", 0)),
            "total_relations": int(raw.get("total_relations", 0)),
            "avg_stages": (
                round(int(raw.get("total_stages", 0)) / count, 1) if count > 0 else 0
            ),
        }

    # -- cleanup --------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove buckets older than RETENTION_HOURS. Returns count deleted."""
        cutoff = self._bucket_ts(time.time() - RETENTION_HOURS * 3600)
        deleted = 0
        for metric_type in ("stage", "pipeline"):
            pattern = f"{AGG_KEY_PREFIX}:{metric_type}:*"
            try:
                for key in self.redis.scan_iter(match=pattern, count=100):
                    try:
                        ts_part = key.rsplit(":", 1)[-1]
                        bucket_ts = int(ts_part)
                        if bucket_ts < cutoff:
                            self.redis.delete(key)
                            deleted += 1
                    except (ValueError, IndexError):
                        continue
            except Exception:
                continue
        return deleted
