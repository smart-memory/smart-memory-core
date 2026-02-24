"""Unit tests for EventBusTransport — Redis Streams async pipeline execution."""

import dataclasses
import json

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.transport.event_bus import (
    EventBusTransport,
    STATUS_QUEUED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    _serialize_config,
    _deserialize_config,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class MockStage:
    def __init__(self, name, side_effect=None):
        self._name = name
        self.side_effect = side_effect
        self.executed = False

    @property
    def name(self):
        return self._name

    def execute(self, state, config):
        if self.side_effect:
            raise self.side_effect
        self.executed = True
        return dataclasses.replace(state, entities=[{"name": f"from_{self._name}"}])

    def undo(self, state):
        return dataclasses.replace(state)


@pytest.fixture
def mock_redis():
    r = MagicMock()
    r.hset = MagicMock()
    r.hgetall = MagicMock(return_value={})
    r.hget = MagicMock(return_value="default")
    r.xadd = MagicMock(return_value="1-0")
    r.xrange = MagicMock(return_value=[])
    r.xdel = MagicMock()
    r.expire = MagicMock()
    r.scan = MagicMock(return_value=(0, []))
    return r


@pytest.fixture
def transport(mock_redis):
    return EventBusTransport(redis_client=mock_redis, stream_prefix="test:pipeline")


# ------------------------------------------------------------------ #
# Submit tests
# ------------------------------------------------------------------ #


class TestSubmit:
    def test_submit_returns_run_id(self, transport, mock_redis):
        state = PipelineState(text="hello", workspace_id="ws1")
        config = PipelineConfig(workspace_id="ws1")

        run_id = transport.submit(state, config, stages=["classify", "store"])

        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format

    def test_submit_publishes_to_first_stage_stream(self, transport, mock_redis):
        state = PipelineState(text="hello", workspace_id="ws1")
        config = PipelineConfig(workspace_id="ws1")

        transport.submit(state, config, stages=["classify", "store"])

        # Should xadd to the first stage stream
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "test:pipeline:ws1:classify"

    def test_submit_stores_status_as_queued(self, transport, mock_redis):
        state = PipelineState(text="hello", workspace_id="ws1")
        config = PipelineConfig(workspace_id="ws1")

        run_id = transport.submit(state, config, stages=["classify"])

        # Should hset status
        hset_calls = mock_redis.hset.call_args_list
        assert len(hset_calls) >= 1
        status_data = hset_calls[0].kwargs.get("mapping", hset_calls[0][1].get("mapping", {}))
        assert status_data["status"] == STATUS_QUEUED

    def test_submit_default_stages(self, transport, mock_redis):
        state = PipelineState(text="hello", workspace_id="ws1")
        config = PipelineConfig(workspace_id="ws1")

        transport.submit(state, config)

        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "test:pipeline:ws1:classify"


# ------------------------------------------------------------------ #
# Status polling tests
# ------------------------------------------------------------------ #


class TestGetStatus:
    def test_get_status_returns_data(self, transport, mock_redis):
        mock_redis.hgetall.return_value = {
            "run_id": "abc-123",
            "status": STATUS_RUNNING,
            "current_stage": "extract",
            "error": "",
        }

        result = transport.get_status("abc-123")

        assert result["status"] == STATUS_RUNNING
        assert result["current_stage"] == "extract"

    def test_get_status_unknown_run(self, transport, mock_redis):
        mock_redis.hgetall.return_value = {}

        result = transport.get_status("nonexistent")

        assert result["status"] == "unknown"


# ------------------------------------------------------------------ #
# Result retrieval tests
# ------------------------------------------------------------------ #


class TestGetResult:
    def test_get_result_when_completed(self, transport, mock_redis):
        mock_redis.hgetall.return_value = {
            "run_id": "abc-123",
            "status": STATUS_COMPLETED,
            "current_stage": "done",
            "error": "",
        }
        mock_redis.hget.return_value = "ws1"

        state_dict = PipelineState(text="hello", item_id="item-1").to_dict()
        mock_redis.xrange.return_value = [
            ("1-0", {"run_id": "abc-123", "state": json.dumps(state_dict)})
        ]

        result = transport.get_result("abc-123")

        assert result is not None
        assert result.item_id == "item-1"

    def test_get_result_when_not_completed(self, transport, mock_redis):
        mock_redis.hgetall.return_value = {
            "run_id": "abc-123",
            "status": STATUS_RUNNING,
            "current_stage": "extract",
            "error": "",
        }

        result = transport.get_result("abc-123")

        assert result is None


# ------------------------------------------------------------------ #
# Publish stage result tests
# ------------------------------------------------------------------ #


class TestPublishStageResult:
    def test_publish_to_next_stage(self, transport, mock_redis):
        state = PipelineState(text="hello", workspace_id="ws1")
        config = PipelineConfig(workspace_id="ws1")

        transport.publish_stage_result("run-1", state, config, ["classify", "store"], stage_index=0)

        # Should publish to "store" stream (next stage)
        xadd_call = mock_redis.xadd.call_args
        assert xadd_call[0][0] == "test:pipeline:ws1:store"

    def test_publish_to_results_when_done(self, transport, mock_redis):
        state = PipelineState(text="hello", workspace_id="ws1")
        config = PipelineConfig(workspace_id="ws1")

        transport.publish_stage_result("run-1", state, config, ["classify", "store"], stage_index=1)

        # Should publish to results stream
        xadd_call = mock_redis.xadd.call_args
        assert xadd_call[0][0] == "test:pipeline:ws1:results"

        # Should mark as completed
        hset_call = mock_redis.hset.call_args
        assert hset_call.kwargs["mapping"]["status"] == STATUS_COMPLETED


# ------------------------------------------------------------------ #
# Mark failed tests
# ------------------------------------------------------------------ #


class TestMarkFailed:
    def test_mark_failed(self, transport, mock_redis):
        transport.mark_failed("run-1", "extract", "LLM timeout")

        hset_call = mock_redis.hset.call_args
        mapping = hset_call.kwargs["mapping"]
        assert mapping["status"] == STATUS_FAILED
        assert mapping["error"] == "LLM timeout"


# ------------------------------------------------------------------ #
# Serialization roundtrip tests
# ------------------------------------------------------------------ #


class TestSerialization:
    def test_state_roundtrip(self):
        state = PipelineState(
            text="Test content",
            workspace_id="ws1",
            entities=[{"name": "Python", "type": "Technology"}],
            item_id="item-42",
        )
        d = state.to_dict()
        restored = PipelineState.from_dict(d)

        assert restored.text == "Test content"
        assert restored.workspace_id == "ws1"
        assert restored.item_id == "item-42"
        assert len(restored.entities) == 1

    def test_config_serialize_deserialize(self):
        config = PipelineConfig(workspace_id="ws1", mode="async")
        serialized = _serialize_config(config)

        assert serialized["workspace_id"] == "ws1"
        assert serialized["mode"] == "async"

        restored = _deserialize_config(serialized)
        assert restored.workspace_id == "ws1"
