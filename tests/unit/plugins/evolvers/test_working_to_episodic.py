"""
Unit tests for WorkingToEpisodicEvolver.

Tests metadata, config defaults, and evolve() behavior with mocked memory.
"""

from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.working_to_episodic import (
    WorkingToEpisodicEvolver,
    WorkingToEpisodicConfig,
)


class TestWorkingToEpisodicMetadata:
    """Tests for plugin metadata."""

    def test_metadata_name(self):
        meta = WorkingToEpisodicEvolver.metadata()
        assert meta.name == "working_to_episodic"

    def test_metadata_plugin_type(self):
        meta = WorkingToEpisodicEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version(self):
        meta = WorkingToEpisodicEvolver.metadata()
        assert meta.version == "1.0.0"


class TestWorkingToEpisodicConfig:
    """Tests for config dataclass defaults and custom values."""

    def test_default_threshold(self):
        config = WorkingToEpisodicConfig()
        assert config.threshold == 40

    def test_custom_threshold(self):
        config = WorkingToEpisodicConfig(threshold=100)
        assert config.threshold == 100


class TestWorkingToEpisodicEvolve:
    """Tests for evolve() method with mocked memory."""

    def _make_evolver(self, threshold=40):
        """Create an evolver with typed config."""
        config = WorkingToEpisodicConfig(threshold=threshold)
        evolver = WorkingToEpisodicEvolver(config=config)
        return evolver

    def _make_memory(self, buffer_items=None, summary=None):
        """Create a mock memory with working and episodic sub-managers."""
        memory = Mock()
        memory.working = Mock()
        memory.episodic = Mock()
        memory.working.get_buffer = Mock(return_value=buffer_items or [])
        memory.working.summarize_buffer = Mock(return_value=summary or Mock(item_id="summary-1"))
        memory.working.clear_buffer = Mock()
        memory.episodic.add = Mock()
        return memory

    def test_below_threshold_does_nothing(self):
        """When buffer size is below threshold, no promotion occurs."""
        evolver = self._make_evolver(threshold=40)
        items = [Mock() for _ in range(10)]
        memory = self._make_memory(buffer_items=items)

        evolver.evolve(memory)

        memory.working.get_buffer.assert_called_once()
        memory.working.summarize_buffer.assert_not_called()
        memory.episodic.add.assert_not_called()
        memory.working.clear_buffer.assert_not_called()

    def test_at_threshold_promotes(self):
        """When buffer size equals threshold, promotion should happen."""
        evolver = self._make_evolver(threshold=5)
        items = [Mock() for _ in range(5)]
        summary_item = Mock(item_id="summary-abc")
        memory = self._make_memory(buffer_items=items, summary=summary_item)

        evolver.evolve(memory)

        memory.working.summarize_buffer.assert_called_once()
        memory.episodic.add.assert_called_once_with(summary_item)
        memory.working.clear_buffer.assert_called_once()

    def test_above_threshold_promotes(self):
        """When buffer size exceeds threshold, promotion should happen."""
        evolver = self._make_evolver(threshold=3)
        items = [Mock() for _ in range(10)]
        summary_item = Mock(item_id="summary-xyz")
        memory = self._make_memory(buffer_items=items, summary=summary_item)

        evolver.evolve(memory)

        memory.working.summarize_buffer.assert_called_once()
        memory.episodic.add.assert_called_once_with(summary_item)

    def test_clear_buffer_called_with_archive_reason(self):
        """After promotion, clear_buffer should be called with an archive reason."""
        evolver = self._make_evolver(threshold=2)
        items = [Mock() for _ in range(5)]
        summary_item = Mock(item_id="summary-42")
        memory = self._make_memory(buffer_items=items, summary=summary_item)

        evolver.evolve(memory)

        memory.working.clear_buffer.assert_called_once()
        call_kwargs = memory.working.clear_buffer.call_args
        assert "archive_reason" in call_kwargs.kwargs
        assert "promoted_to_episodic" in call_kwargs.kwargs["archive_reason"]
        assert "summary-42" in call_kwargs.kwargs["archive_reason"]

    def test_logger_called_on_promotion(self):
        """If a logger is provided, info() should be called on promotion."""
        evolver = self._make_evolver(threshold=2)
        items = [Mock() for _ in range(5)]
        memory = self._make_memory(buffer_items=items)
        logger = Mock()

        evolver.evolve(memory, logger=logger)

        logger.info.assert_called_once()

    def test_no_logger_does_not_raise(self):
        """Evolve should work without a logger."""
        evolver = self._make_evolver(threshold=2)
        items = [Mock() for _ in range(5)]
        memory = self._make_memory(buffer_items=items)

        evolver.evolve(memory, logger=None)

    def test_defaults_to_typed_config_when_empty(self):
        """CORE-EVO-LIVE-1: Passing config={} defaults to WorkingToEpisodicConfig."""
        evolver = WorkingToEpisodicEvolver(config={})
        # Should not raise — defaults to typed config with threshold=40
        assert hasattr(evolver.config, "threshold")
        assert evolver.config.threshold == 40
