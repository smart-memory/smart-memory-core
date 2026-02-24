"""Tests for MetricsConsumer — pipeline metrics aggregation."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit

from smartmemory.pipeline.metrics_consumer import (
    AGG_KEY_PREFIX,
    MetricsConsumer,
)


@pytest.fixture
def mock_redis():
    """Create a mock Redis client with basic behavior."""
    r = MagicMock()
    r.get.return_value = None
    r.xrange.return_value = []
    r.hgetall.return_value = {}
    r.pipeline.return_value = MagicMock()
    r.pipeline.return_value.execute.return_value = []
    return r


@pytest.fixture
def consumer(mock_redis):
    """Create a MetricsConsumer with mocked Redis."""
    with patch("smartmemory.pipeline.metrics_consumer.get_config"):
        return MetricsConsumer(redis_client=mock_redis)


class TestBucketHelpers:
    def test_bucket_ts_floors_to_boundary(self):
        # 300s boundaries: 0, 300, 600, ...
        assert MetricsConsumer._bucket_ts(0) == 0
        assert MetricsConsumer._bucket_ts(150) == 0
        assert MetricsConsumer._bucket_ts(299) == 0
        assert MetricsConsumer._bucket_ts(300) == 300
        assert MetricsConsumer._bucket_ts(450) == 300
        assert MetricsConsumer._bucket_ts(600) == 600

    def test_bucket_key_format(self):
        key = MetricsConsumer._bucket_key("stage", 300)
        assert key == f"{AGG_KEY_PREFIX}:stage:300"

    def test_bucket_key_pipeline(self):
        key = MetricsConsumer._bucket_key("pipeline", 600)
        assert key == f"{AGG_KEY_PREFIX}:pipeline:600"


class TestRunProcessing:
    def test_run_empty_stream_returns_zero(self, consumer, mock_redis):
        mock_redis.xrange.return_value = []
        assert consumer.run() == 0

    def test_run_processes_stage_complete_event(self, consumer, mock_redis):
        ts = time.time()
        event_data = json.dumps({
            "event_type": "stage_complete",
            "stage_name": "entity_ruler",
            "elapsed_ms": 4.2,
            "status": "success",
            "entity_count": 5,
            "relation_count": 3,
        })
        mock_redis.xrange.side_effect = [
            [("1-0", {
                "event_type": "stage_complete",
                "component": "pipeline",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ts)),
                "data": event_data,
            })],
            [],  # Second call returns empty (end of stream)
        ]
        mock_redis.hget.return_value = "0"

        count = consumer.run()
        assert count == 1
        mock_redis.pipeline.assert_called()

    def test_run_processes_pipeline_complete_event(self, consumer, mock_redis):
        ts = time.time()
        event_data = json.dumps({
            "event_type": "pipeline_complete",
            "total_ms": 120.5,
            "stages_completed": 11,
            "entity_count": 8,
            "relation_count": 5,
        })
        mock_redis.xrange.side_effect = [
            [("2-0", {
                "event_type": "pipeline_complete",
                "component": "pipeline",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ts)),
                "data": event_data,
            })],
            [],
        ]

        count = consumer.run()
        assert count == 1

    def test_run_respects_max_iterations(self, consumer, mock_redis):
        events = [
            (f"{i}-0", {
                "event_type": "stage_complete",
                "timestamp": "",
                "data": json.dumps({"stage_name": "classify", "elapsed_ms": 1.0, "status": "success", "entity_count": 0, "relation_count": 0}),
            })
            for i in range(10)
        ]
        mock_redis.xrange.return_value = events
        mock_redis.hget.return_value = "0"

        count = consumer.run(max_iterations=3)
        assert count == 3

    def test_run_skips_malformed_events(self, consumer, mock_redis):
        mock_redis.xrange.side_effect = [
            [("3-0", {"event_type": "stage_complete", "timestamp": "", "data": "not-json{{"})],
            [],
        ]

        # Should not raise, just skip
        count = consumer.run()
        assert count == 1

    def test_cursor_persisted_after_run(self, consumer, mock_redis):
        mock_redis.xrange.side_effect = [
            [("100-0", {"event_type": "stage_complete", "timestamp": "", "data": json.dumps({"stage_name": "store", "elapsed_ms": 2.0, "status": "success", "entity_count": 0, "relation_count": 0})})],
            [],
        ]
        mock_redis.hget.return_value = "0"
        consumer.run()
        mock_redis.set.assert_called()


class TestGetAggregated:
    def test_get_aggregated_empty(self, consumer, mock_redis):
        mock_redis.hgetall.return_value = {}
        result = consumer.get_aggregated("stage", hours=1)
        assert isinstance(result, list)

    def test_get_aggregated_returns_parsed_stage_data(self, consumer, mock_redis):
        bucket_ts = MetricsConsumer._bucket_ts(time.time())
        key = MetricsConsumer._bucket_key("stage", bucket_ts)

        mock_redis.hgetall.return_value = {
            "entity_ruler:count": "10",
            "entity_ruler:total_ms": "42.0",
            "entity_ruler:max_ms": "8.1",
            "entity_ruler:errors": "1",
            "entity_ruler:entities": "50",
            "entity_ruler:relations": "30",
        }

        result = consumer.get_aggregated("stage", hours=1)
        assert len(result) >= 1
        stage_data = [r for r in result if r.get("stage") == "entity_ruler"]
        assert len(stage_data) > 0
        record = stage_data[0]
        assert record["count"] == 10
        assert record["avg_latency_ms"] == 4.2
        assert record["error_count"] == 1
        assert record["total_entities"] == 50

    def test_get_aggregated_pipeline_type(self, consumer, mock_redis):
        bucket_ts = MetricsConsumer._bucket_ts(time.time())
        mock_redis.hgetall.return_value = {
            "count": "5",
            "total_ms": "500.0",
            "total_entities": "25",
            "total_relations": "15",
            "total_stages": "55",
        }

        result = consumer.get_aggregated("pipeline", hours=1)
        assert len(result) >= 1
        record = [r for r in result if r.get("count", 0) > 0]
        assert len(record) > 0
        assert record[0]["count"] == 5
        assert record[0]["avg_total_ms"] == 100.0


class TestTimestampParsing:
    def test_parse_iso_timestamp(self):
        ts = MetricsConsumer._parse_timestamp("2026-02-06T12:00:00+00:00")
        assert isinstance(ts, float)
        assert ts > 0

    def test_parse_empty_timestamp_falls_back(self):
        ts = MetricsConsumer._parse_timestamp("")
        assert abs(ts - time.time()) < 5

    def test_parse_invalid_timestamp_falls_back(self):
        ts = MetricsConsumer._parse_timestamp("not-a-date")
        assert abs(ts - time.time()) < 5
