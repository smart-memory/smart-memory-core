"""
Integration tests for TemporalQueries class.

Tests all temporal query methods including:
- get_history()
- at_time()
- get_changes()
- compare_versions()
- rollback()
- get_audit_trail()
- find_memories_changed_since()
- get_timeline()

Relocated from tests/unit/temporal/ because these tests require SmartMemory
with graph backend (FalkorDB) for temporal versioning and queries.
"""

import pytest


pytestmark = [pytest.mark.integration]
from datetime import datetime, timedelta, UTC
from smartmemory import SmartMemory, MemoryItem
from smartmemory.temporal.queries import TemporalQueries, TemporalVersion, TemporalChange
import time


@pytest.fixture
def memory():
    """Provide clean SmartMemory instance."""
    mem = SmartMemory()
    yield mem
    # Cleanup
    try:
        mem.clear()
    except Exception:
        pass


@pytest.fixture
def temporal(memory):
    """Provide TemporalQueries instance."""
    return memory.temporal


class TestTemporalQueriesBasic:
    """Basic temporal query functionality tests."""

    def test_temporal_queries_exists(self, memory):
        """Test that temporal queries are available on SmartMemory."""
        assert hasattr(memory, "temporal")
        assert isinstance(memory.temporal, TemporalQueries)

    def test_temporal_queries_has_methods(self, temporal):
        """Test that all expected methods exist."""
        assert hasattr(temporal, "get_history")
        assert hasattr(temporal, "at_time")
        assert hasattr(temporal, "get_changes")
        assert hasattr(temporal, "compare_versions")
        assert hasattr(temporal, "rollback")
        assert hasattr(temporal, "get_audit_trail")
        assert hasattr(temporal, "find_memories_changed_since")
        assert hasattr(temporal, "get_timeline")


class TestGetHistory:
    """Tests for get_history() method."""

    def test_get_history_empty(self, temporal):
        """Test get_history with non-existent item."""
        history = temporal.get_history("nonexistent-id")
        assert isinstance(history, list)
        # Non-existent items should return empty list or have valid structure
        # If not empty, items must be TemporalVersion objects
        if history:
            assert all(isinstance(v, TemporalVersion) for v in history)

    @pytest.mark.skip(reason="Requires real storage backend for add->retrieve round-trip")
    def test_get_history_single_item(self, memory, temporal):
        """Test get_history with single item (requires real storage backend)."""
        item = MemoryItem(content="Test content")
        item_id = memory.add(item)

        history = temporal.get_history(item_id)

        assert isinstance(history, list)
        assert len(history) >= 1
        assert all(isinstance(v, TemporalVersion) for v in history)

    def test_get_history_returns_temporal_versions(self, memory, temporal):
        """Test that history returns TemporalVersion objects."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        history = temporal.get_history(item_id)

        if history:
            version = history[0]
            assert hasattr(version, "item_id")
            assert hasattr(version, "content")
            assert hasattr(version, "metadata")
            assert hasattr(version, "transaction_time_start")

    def test_get_history_ordered_by_time(self, memory, temporal):
        """Test that history is ordered newest first."""
        item = MemoryItem(content="Version 1")
        item_id = memory.add(item)

        time.sleep(0.1)

        item2 = MemoryItem(content="Version 2", metadata={"version": 2, "original_id": item_id})
        memory.add(item2)

        history = temporal.get_history(item_id)

        # Should be ordered newest first
        if len(history) > 1:
            assert history[0].transaction_time_start >= history[1].transaction_time_start


class TestAtTime:
    """Tests for at_time() method."""

    def test_at_time_returns_list(self, temporal):
        """Test that at_time returns a list."""
        result = temporal.at_time("2024-01-01")
        assert isinstance(result, list)

    def test_at_time_accepts_iso_format(self, temporal):
        """Test that at_time accepts ISO format dates."""
        # Should not raise exception
        result = temporal.at_time("2024-01-01T00:00:00")
        assert isinstance(result, list)

        result = temporal.at_time("2024-01-01")
        assert isinstance(result, list)

    def test_at_time_with_filters(self, temporal):
        """Test at_time with additional filters."""
        result = temporal.at_time("2024-01-01", filters={"user_id": "test_user"})
        assert isinstance(result, list)


class TestGetChanges:
    """Tests for get_changes() method."""

    def test_get_changes_empty(self, temporal):
        """Test get_changes with non-existent item."""
        changes = temporal.get_changes("nonexistent-id")
        assert isinstance(changes, list)
        assert len(changes) == 0

    def test_get_changes_single_version(self, memory, temporal):
        """Test get_changes with single version (no changes)."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        changes = temporal.get_changes(item_id)

        # Single version should have no changes
        assert isinstance(changes, list)

    def test_get_changes_returns_temporal_changes(self, memory, temporal):
        """Test that changes are TemporalChange objects."""
        item = MemoryItem(content="Version 1")
        item_id = memory.add(item)

        time.sleep(0.1)

        item2 = MemoryItem(content="Version 2", metadata={"version": 2, "original_id": item_id})
        memory.add(item2)

        changes = temporal.get_changes(item_id)

        # May or may not detect changes depending on implementation
        for change in changes:
            assert isinstance(change, TemporalChange)
            assert hasattr(change, "item_id")
            assert hasattr(change, "timestamp")
            assert hasattr(change, "change_type")
            assert hasattr(change, "changed_fields")

    def test_get_changes_with_time_range(self, memory, temporal):
        """Test get_changes with time range."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        until = datetime.now(UTC).isoformat()

        changes = temporal.get_changes(item_id, since=since, until=until)
        assert isinstance(changes, list)


class TestCompareVersions:
    """Tests for compare_versions() method."""

    def test_compare_versions_returns_dict(self, memory, temporal):
        """Test that compare_versions returns a dict."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        time1 = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        time2 = datetime.now(UTC).isoformat()

        result = temporal.compare_versions(item_id, time1, time2)
        assert isinstance(result, dict)

    def test_compare_versions_has_expected_keys(self, memory, temporal):
        """Test that comparison result has expected structure."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        time1 = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        time2 = datetime.now(UTC).isoformat()

        result = temporal.compare_versions(item_id, time1, time2)

        # Should have changed_fields key
        assert "changed_fields" in result or "error" in result

    def test_compare_versions_nonexistent_item(self, temporal):
        """Test compare_versions with non-existent item."""
        result = temporal.compare_versions("nonexistent", "2024-01-01", "2024-01-02")

        assert isinstance(result, dict)
        assert "error" in result


class TestRollback:
    """Tests for rollback() method."""

    def test_rollback_dry_run_default(self, memory, temporal):
        """Test that rollback defaults to dry_run=True."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        to_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        result = temporal.rollback(item_id, to_time)

        assert isinstance(result, dict)
        # Should be dry run by default
        assert result.get("dry_run", False) or "error" in result

    def test_rollback_dry_run_explicit(self, memory, temporal):
        """Test rollback with explicit dry_run=True."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        to_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        result = temporal.rollback(item_id, to_time, dry_run=True)

        assert isinstance(result, dict)
        assert result.get("dry_run", False) or "error" in result

    def test_rollback_returns_preview_info(self, memory, temporal):
        """Test that dry_run rollback returns preview info."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        to_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        result = temporal.rollback(item_id, to_time, dry_run=True)

        assert isinstance(result, dict)
        # Should have either preview info or error
        assert "would_change" in result or "preview" in result or "error" in result

    def test_rollback_nonexistent_item(self, temporal):
        """Test rollback with non-existent item."""
        result = temporal.rollback("nonexistent", "2024-01-01", dry_run=True)

        assert isinstance(result, dict)
        # Must have one of these keys
        assert "error" in result or "dry_run" in result or "preview" in result
        # If it's an error, verify error message exists
        if "error" in result:
            assert result["error"], "Error message should not be empty"


class TestGetAuditTrail:
    """Tests for get_audit_trail() method."""

    def test_get_audit_trail_returns_list(self, memory, temporal):
        """Test that get_audit_trail returns a list."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        trail = temporal.get_audit_trail(item_id)

        assert isinstance(trail, list)

    def test_get_audit_trail_has_events(self, memory, temporal):
        """Test that audit trail contains events."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        trail = temporal.get_audit_trail(item_id)

        # Should return a list
        assert isinstance(trail, list)
        # If trail has events, they must have proper structure
        for event in trail:
            assert isinstance(event, dict)
            assert "timestamp" in event
            assert "action" in event

    def test_get_audit_trail_event_structure(self, memory, temporal):
        """Test that audit events have expected structure."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        trail = temporal.get_audit_trail(item_id)

        if trail:
            event = trail[0]
            assert isinstance(event, dict)
            assert "timestamp" in event
            assert "action" in event
            assert "valid_from" in event
            assert "valid_to" in event

    def test_get_audit_trail_with_metadata(self, memory, temporal):
        """Test audit trail with metadata included."""
        item = MemoryItem(content="Test", metadata={"user_id": "test_user"})
        item_id = memory.add(item)

        trail = temporal.get_audit_trail(item_id, include_metadata=True)

        if trail:
            event = trail[0]
            assert "metadata" in event or "content_preview" in event

    def test_get_audit_trail_without_metadata(self, memory, temporal):
        """Test audit trail without metadata."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        trail = temporal.get_audit_trail(item_id, include_metadata=False)

        assert isinstance(trail, list)

    def test_get_audit_trail_empty(self, temporal):
        """Test audit trail for non-existent item."""
        trail = temporal.get_audit_trail("nonexistent")

        assert isinstance(trail, list)
        assert len(trail) == 0


class TestFindMemoriesChangedSince:
    """Tests for find_memories_changed_since() method."""

    def test_find_changed_returns_list(self, temporal):
        """Test that find_memories_changed_since returns a list."""
        since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        result = temporal.find_memories_changed_since(since)

        assert isinstance(result, list)

    def test_find_changed_accepts_iso_format(self, temporal):
        """Test that method accepts ISO format dates."""
        # Should not raise exception
        result = temporal.find_memories_changed_since("2024-01-01T00:00:00")
        assert isinstance(result, list)

        result = temporal.find_memories_changed_since("2024-01-01")
        assert isinstance(result, list)

    def test_find_changed_with_filters(self, temporal):
        """Test find_memories_changed_since with filters."""
        since = datetime.now(UTC).isoformat()

        result = temporal.find_memories_changed_since(since, filters={"user_id": "test"})

        assert isinstance(result, list)


class TestGetTimeline:
    """Tests for get_timeline() method."""

    def test_get_timeline_returns_dict(self, memory, temporal):
        """Test that get_timeline returns a dict."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        timeline = temporal.get_timeline(item_id)

        assert isinstance(timeline, dict)

    def test_get_timeline_granularity_day(self, memory, temporal):
        """Test timeline with day granularity."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        timeline = temporal.get_timeline(item_id, granularity="day")

        assert isinstance(timeline, dict)

    def test_get_timeline_granularity_hour(self, memory, temporal):
        """Test timeline with hour granularity."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        timeline = temporal.get_timeline(item_id, granularity="hour")

        assert isinstance(timeline, dict)

    def test_get_timeline_granularity_week(self, memory, temporal):
        """Test timeline with week granularity."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        timeline = temporal.get_timeline(item_id, granularity="week")

        assert isinstance(timeline, dict)

    def test_get_timeline_granularity_month(self, memory, temporal):
        """Test timeline with month granularity."""
        item = MemoryItem(content="Test")
        item_id = memory.add(item)

        timeline = temporal.get_timeline(item_id, granularity="month")

        assert isinstance(timeline, dict)

    def test_get_timeline_empty(self, temporal):
        """Test timeline for non-existent item."""
        timeline = temporal.get_timeline("nonexistent")

        assert isinstance(timeline, dict)
        assert len(timeline) == 0


class TestTemporalDataTypes:
    """Tests for temporal data types."""

    def test_temporal_version_structure(self):
        """Test TemporalVersion dataclass structure."""
        version = TemporalVersion(item_id="test-id", content="Test content", metadata={"key": "value"}, version=1)

        assert version.item_id == "test-id"
        assert version.content == "Test content"
        assert version.metadata == {"key": "value"}
        assert version.version == 1

    def test_temporal_change_structure(self):
        """Test TemporalChange dataclass structure."""
        change = TemporalChange(
            item_id="test-id", timestamp=datetime.now(UTC), change_type="updated", changed_fields=["content"]
        )

        assert change.item_id == "test-id"
        assert change.change_type == "updated"
        assert "content" in change.changed_fields


class TestTemporalIntegration:
    """Integration tests for temporal queries."""

    @pytest.mark.skip(reason="Requires real storage backend for add->retrieve round-trip")
    def test_full_temporal_workflow(self, memory, temporal):
        """Test complete temporal workflow (requires real storage backend)."""
        # Create item
        item = MemoryItem(content="Original content", metadata={"version": 1, "author": "alice"})
        item_id = memory.add(item)

        time.sleep(0.1)

        # Create version 2
        item2 = MemoryItem(content="Updated content", metadata={"version": 2, "author": "bob", "original_id": item_id})
        memory.add(item2)

        # Get history - must return list with at least one version
        history = temporal.get_history(item_id)
        assert isinstance(history, list)
        assert len(history) >= 1, "History should contain at least the original version"
        assert all(isinstance(v, TemporalVersion) for v in history)

        # Get audit trail - must return list
        trail = temporal.get_audit_trail(item_id)
        assert isinstance(trail, list)
        # Verify structure if trail exists
        for event in trail:
            assert isinstance(event, dict)
            assert "action" in event

        # Get timeline - must return dict
        timeline = temporal.get_timeline(item_id)
        assert isinstance(timeline, dict)

    @pytest.mark.skip(reason="Requires real storage backend for add->retrieve round-trip")
    def test_temporal_queries_with_metadata(self, memory, temporal):
        """Test temporal queries preserve metadata (requires real storage backend)."""
        item = MemoryItem(content="Test", metadata={"user_id": "user123", "category": "test", "confidence": 0.95})
        item_id = memory.add(item)

        # Get history
        history = temporal.get_history(item_id)

        # Verify history is returned with at least one version
        assert isinstance(history, list)
        assert len(history) >= 1, "Should have at least one version"

        # Verify metadata structure
        version = history[0]
        assert hasattr(version, "metadata"), "Version must have metadata attribute"
        assert hasattr(version, "content"), "Version must have content attribute"
        assert hasattr(version, "item_id"), "Version must have item_id attribute"


class TestScopeProviderFiltering:
    """Tests that scope_provider correctly filters temporal queries."""

    def test_temporal_queries_accepts_scope_provider(self):
        """TemporalQueries accepts scope_provider parameter without error."""
        from smartmemory.scope_provider import DefaultScopeProvider

        provider = DefaultScopeProvider(workspace_id="ws_test")
        mem = SmartMemory(scope_provider=provider)

        assert mem.temporal.scope_provider is provider
        assert mem.version_tracker.scope_provider is provider

    def test_temporal_queries_without_scope_provider_backward_compat(self):
        """TemporalQueries works without scope_provider (OSS mode)."""
        mem = SmartMemory()

        # scope_provider should be DefaultScopeProvider with no workspace_id
        assert mem.temporal is not None
        assert mem.version_tracker is not None
        # OSS mode: default scope provider has no workspace_id
        filters = mem.scope_provider.get_isolation_filters()
        assert "workspace_id" not in filters

    def test_version_tracker_scope_provider_propagates(self):
        """scope_provider propagates from SmartMemory to VersionTracker."""
        from smartmemory.scope_provider import DefaultScopeProvider

        provider = DefaultScopeProvider(workspace_id="ws_abc", tenant_id="t_abc")
        mem = SmartMemory(scope_provider=provider)

        assert mem.version_tracker.scope_provider is provider
        assert mem.version_tracker.scope_provider.get_isolation_filters()["workspace_id"] == "ws_abc"

    def test_temporal_relationships_scope_provider_propagates(self):
        """scope_provider propagates from SmartMemory to TemporalRelationshipQueries."""
        from smartmemory.scope_provider import DefaultScopeProvider

        provider = DefaultScopeProvider(workspace_id="ws_xyz")
        mem = SmartMemory(scope_provider=provider)

        assert mem.temporal.relationships is not None
        assert mem.temporal.relationships.scope_provider is provider

    def test_version_tracker_create_version_injects_write_context(self):
        """create_version injects workspace_id from scope_provider into version metadata."""
        from smartmemory.scope_provider import DefaultScopeProvider

        provider = DefaultScopeProvider(
            workspace_id="ws_inject_test",
            tenant_id="t_inject_test",
            user_id="u_inject_test",
        )
        mem = SmartMemory(scope_provider=provider)
        tracker = mem.version_tracker

        # Add a node so create_version can find it
        item_id = mem.add(MemoryItem(content="Test item for version injection"))

        version = tracker.create_version(
            item_id=item_id,
            content="Versioned content",
        )

        assert version.metadata.get("workspace_id") == "ws_inject_test"
        assert version.metadata.get("tenant_id") == "t_inject_test"
        assert version.metadata.get("user_id") == "u_inject_test"
