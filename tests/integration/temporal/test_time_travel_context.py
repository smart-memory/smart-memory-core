"""
Integration tests for time-travel context manager.

Tests the time_travel() context manager functionality with SmartMemory
instances that depend on graph backend infrastructure.

Relocated from tests/unit/temporal/ because these tests instantiate
SmartMemory and exercise memory.add(), memory.time_travel(), and
temporal query methods that require graph backend (FalkorDB).
"""

import pytest


pytestmark = [pytest.mark.integration]
from datetime import datetime, timedelta, UTC
from smartmemory import SmartMemory, MemoryItem
from smartmemory.temporal.context import time_travel, TemporalContext


@pytest.fixture
def memory():
    """Provide clean SmartMemory instance."""
    mem = SmartMemory()
    yield mem
    try:
        mem.clear()
    except:
        pass


class TestTimeTravelContext:
    """Tests for TemporalContext class."""

    def test_temporal_context_creation(self, memory):
        """Test creating a TemporalContext."""
        ctx = TemporalContext(memory, "2024-01-01")

        assert ctx.memory == memory
        assert ctx.time_point == "2024-01-01"
        assert ctx.original_time is None

    def test_temporal_context_enter(self, memory):
        """Test entering temporal context."""
        ctx = TemporalContext(memory, "2024-01-01")

        with ctx:
            # Should set temporal context on memory
            assert hasattr(memory, '_temporal_context')
            assert memory._temporal_context == "2024-01-01"

    def test_temporal_context_exit(self, memory):
        """Test exiting temporal context."""
        # Set initial context
        memory._temporal_context = "original"

        ctx = TemporalContext(memory, "2024-01-01")

        with ctx:
            assert memory._temporal_context == "2024-01-01"

        # Should restore original context
        assert memory._temporal_context == "original"

    def test_temporal_context_exit_with_none(self, memory):
        """Test exiting context when original was None."""
        ctx = TemporalContext(memory, "2024-01-01")

        with ctx:
            assert memory._temporal_context == "2024-01-01"

        # Should restore to None
        assert memory._temporal_context is None

    def test_temporal_context_exception_handling(self, memory):
        """Test that context exits properly even with exception."""
        memory._temporal_context = "original"

        ctx = TemporalContext(memory, "2024-01-01")

        try:
            with ctx:
                assert memory._temporal_context == "2024-01-01"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore original context
        assert memory._temporal_context == "original"


class TestTimeTravelFunction:
    """Tests for time_travel() function."""

    def test_time_travel_function_exists(self):
        """Test that time_travel function exists."""
        from smartmemory.temporal.context import time_travel
        assert callable(time_travel)

    def test_time_travel_returns_context_manager(self, memory):
        """Test that time_travel returns a context manager."""
        ctx = time_travel(memory, "2024-01-01")

        # Should be a context manager
        assert hasattr(ctx, '__enter__')
        assert hasattr(ctx, '__exit__')

    def test_time_travel_sets_context(self, memory):
        """Test that time_travel sets temporal context."""
        with time_travel(memory, "2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

    def test_time_travel_restores_context(self, memory):
        """Test that time_travel restores original context."""
        memory._temporal_context = "original"

        with time_travel(memory, "2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

        assert memory._temporal_context == "original"


class TestSmartMemoryTimeTravelMethod:
    """Tests for SmartMemory.time_travel() method."""

    def test_memory_has_time_travel_method(self, memory):
        """Test that SmartMemory has time_travel method."""
        assert hasattr(memory, 'time_travel')
        assert callable(memory.time_travel)

    def test_memory_time_travel_returns_context_manager(self, memory):
        """Test that memory.time_travel() returns context manager."""
        ctx = memory.time_travel("2024-01-01")

        assert hasattr(ctx, '__enter__')
        assert hasattr(ctx, '__exit__')

    def test_memory_time_travel_sets_context(self, memory):
        """Test that memory.time_travel() sets context."""
        with memory.time_travel("2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

    def test_memory_time_travel_accepts_iso_format(self, memory):
        """Test that time_travel accepts ISO format dates."""
        # Should not raise exception
        with memory.time_travel("2024-01-01T00:00:00"):
            assert memory._temporal_context == "2024-01-01T00:00:00"

        with memory.time_travel("2024-01-01"):
            assert memory._temporal_context == "2024-01-01"


class TestTimeTravelNesting:
    """Tests for nested time-travel contexts."""

    def test_nested_time_travel(self, memory):
        """Test nested time-travel contexts."""
        with memory.time_travel("2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

            with memory.time_travel("2024-02-01"):
                assert memory._temporal_context == "2024-02-01"

            # Should restore to first context
            assert memory._temporal_context == "2024-01-01"

        # Should restore to None
        assert memory._temporal_context is None

    def test_triple_nested_time_travel(self, memory):
        """Test triple-nested time-travel contexts."""
        with memory.time_travel("2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

            with memory.time_travel("2024-02-01"):
                assert memory._temporal_context == "2024-02-01"

                with memory.time_travel("2024-03-01"):
                    assert memory._temporal_context == "2024-03-01"

                assert memory._temporal_context == "2024-02-01"

            assert memory._temporal_context == "2024-01-01"

        assert memory._temporal_context is None


class TestTimeTravelWithQueries:
    """Tests for time-travel with actual queries."""

    def test_time_travel_with_add(self, memory):
        """Test adding memories within time-travel context."""
        with memory.time_travel("2024-01-01"):
            item = MemoryItem(content="Test in past")
            item_id = memory.add(item)

            assert item_id is not None

    def test_time_travel_with_get(self, memory):
        """Test retrieving memories within time-travel context."""
        # Add item in present
        item = MemoryItem(content="Test")
        memory.add(item)

        # Try to get in past context
        with memory.time_travel("2024-01-01"):
            # Item might not exist in past
            # This tests that the context is set, not the actual time-travel logic
            assert memory._temporal_context == "2024-01-01"

    def test_time_travel_with_search(self, memory):
        """Test searching within time-travel context."""
        # Add item
        item = MemoryItem(content="Python programming")
        memory.add(item)

        # Search in time-travel context
        with memory.time_travel("2024-01-01"):
            # Context should be set
            assert memory._temporal_context == "2024-01-01"

            # Context is set - actual search behavior depends on implementation
            # Just verify context is active
            pass


class TestTimeTravelEdgeCases:
    """Edge case tests for time-travel."""

    def test_time_travel_with_empty_string(self, memory):
        """Test time-travel with empty string."""
        with memory.time_travel(""):
            assert memory._temporal_context == ""

    def test_time_travel_with_future_date(self, memory):
        """Test time-travel to future date."""
        future = (datetime.now(UTC) + timedelta(days=365)).isoformat()

        with memory.time_travel(future):
            assert memory._temporal_context == future

    def test_time_travel_with_past_date(self, memory):
        """Test time-travel to past date."""
        past = (datetime.now(UTC) - timedelta(days=365)).isoformat()

        with memory.time_travel(past):
            assert memory._temporal_context == past

    def test_time_travel_multiple_times_same_context(self, memory):
        """Test entering same time-travel context multiple times."""
        with memory.time_travel("2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

        assert memory._temporal_context is None

        with memory.time_travel("2024-01-01"):
            assert memory._temporal_context == "2024-01-01"

        assert memory._temporal_context is None


class TestTimeTravelIntegration:
    """Integration tests for time-travel functionality."""

    def test_time_travel_workflow(self, memory):
        """Test complete time-travel workflow."""
        # Add item in present
        item = MemoryItem(
            content="Present content",
            metadata={"timestamp": datetime.now(UTC).isoformat()}
        )
        memory.add(item)

        # Time travel to past
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        with memory.time_travel(past):
            # Context should be set
            assert memory._temporal_context == past

        # Context should be restored
        assert memory._temporal_context is None

    def test_time_travel_with_temporal_queries(self, memory):
        """Test time-travel with temporal query methods."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        with memory.time_travel(past):
            # Temporal queries should still work
            history = memory.temporal.get_history(item_id)
            assert isinstance(history, list)

            trail = memory.temporal.get_audit_trail(item_id)
            assert isinstance(trail, list)
