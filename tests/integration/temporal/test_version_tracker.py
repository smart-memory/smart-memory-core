"""
Integration tests for version tracking system.

Tests VersionTracker interactions with graph backends:
- Mock-based tests: validate call patterns against any backend
- SQLite-based tests (DIST-LITE-DEGRADE-1a): real SQLiteBackend round-trips

Relocated from tests/unit/temporal/ because these tests verify behavior
that depends on graph backend query patterns (node properties, edge
creation, neighbor traversal).
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from smartmemory.temporal.version_tracker import VersionTracker, Version


pytestmark = [pytest.mark.integration]


def _version_to_dict(version: Version) -> dict:
    """Convert a Version to a dict matching backend return format (ref_item_id, flattened metadata)."""
    data = version.to_dict()
    if "item_id" in data:
        data["ref_item_id"] = data.pop("item_id")
    metadata = data.pop("metadata", {})
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if k not in data:
                data[k] = v
    return data


class TestVersion:
    """Test Version dataclass."""

    def test_version_creation(self):
        """Test creating a version."""
        now = datetime.now(timezone.utc)
        version = Version(
            item_id="test123",
            version_number=1,
            content="Test content",
            metadata={"key": "value"},
            valid_time_start=now,
            transaction_time_start=now,
        )

        assert version.item_id == "test123"
        assert version.version_number == 1
        assert version.content == "Test content"
        assert version.metadata == {"key": "value"}
        assert version.valid_time_start == now
        assert version.transaction_time_start == now

    def test_version_to_dict(self):
        """Test converting version to dictionary."""
        now = datetime.now(timezone.utc)
        version = Version(
            item_id="test123",
            version_number=1,
            content="Test content",
            valid_time_start=now,
            transaction_time_start=now,
        )

        data = version.to_dict()

        assert isinstance(data, dict)
        assert data["item_id"] == "test123"
        assert data["version_number"] == 1
        assert isinstance(data["valid_time_start"], str)  # Should be ISO string
        assert isinstance(data["transaction_time_start"], str)

    def test_version_from_dict(self):
        """Test creating version from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "item_id": "test123",
            "version_number": 1,
            "content": "Test content",
            "metadata": {},
            "valid_time_start": now.isoformat(),
            "transaction_time_start": now.isoformat(),
            "valid_time_end": None,
            "transaction_time_end": None,
            "changed_by": None,
            "change_reason": None,
            "previous_version": None,
        }

        version = Version.from_dict(data)

        assert version.item_id == "test123"
        assert version.version_number == 1
        assert isinstance(version.valid_time_start, datetime)
        assert isinstance(version.transaction_time_start, datetime)

    def test_version_is_current(self):
        """Test checking if version is current."""
        now = datetime.now(timezone.utc)

        # Current version (no transaction_time_end)
        current = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            transaction_time_start=now,
            transaction_time_end=None,
        )
        assert current.is_current() is True

        # Closed version
        closed = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            transaction_time_start=now,
            transaction_time_end=now + timedelta(hours=1),
        )
        assert closed.is_current() is False

    def test_version_is_valid_at(self):
        """Test checking if version is valid at a time."""
        now = datetime.now(timezone.utc)

        version = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            valid_time_start=now,
            valid_time_end=now + timedelta(days=1),
        )

        # Before valid time
        assert version.is_valid_at(now - timedelta(hours=1)) is False

        # During valid time
        assert version.is_valid_at(now + timedelta(hours=12)) is True

        # After valid time
        assert version.is_valid_at(now + timedelta(days=2)) is False

        # Open-ended (no end time)
        open_version = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            valid_time_start=now,
            valid_time_end=None,
        )
        assert open_version.is_valid_at(now + timedelta(days=100)) is True


class TestVersionTracker:
    """Test VersionTracker class with mocked graph backend."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock graph backend with backend attribute for VersionTracker."""
        graph = Mock()
        graph.add_node = Mock(return_value=True)
        graph.update_node = Mock(return_value=True)
        # Backend is the raw storage layer — VersionTracker uses it for get_node, get_neighbors, add_edge
        graph.backend.get_node = Mock(return_value=None)
        graph.backend.get_neighbors = Mock(return_value=[])
        graph.backend.search_nodes = Mock(return_value=[])
        graph.backend.add_edge = Mock(return_value=True)
        return graph

    @pytest.fixture
    def tracker(self, mock_graph):
        """Create version tracker with mock graph."""
        return VersionTracker(mock_graph)

    def test_create_first_version(self, tracker, mock_graph):
        """Test creating the first version of an item."""
        # _query_versions: no neighbors, no search_nodes results
        mock_graph.backend.get_neighbors.return_value = []
        mock_graph.backend.search_nodes.return_value = []
        # _store_version: no existing main node
        mock_graph.backend.get_node.return_value = None

        version = tracker.create_version(
            item_id="test123",
            content="First version",
            metadata={"author": "alice"},
        )

        assert version.item_id == "test123"
        assert version.version_number == 1
        assert version.content == "First version"
        assert version.metadata["author"] == "alice"
        assert version.is_current() is True

        # Verify graph calls — add_node for version + placeholder, add_edge for HAS_VERSION
        assert mock_graph.add_node.call_count == 2  # version node + placeholder
        assert mock_graph.backend.add_edge.called

    def test_create_second_version(self, tracker, mock_graph):
        """Test creating a second version."""
        existing_version = Version(
            item_id="test123",
            version_number=1,
            content="First version",
            transaction_time_start=datetime.now(timezone.utc),
            transaction_time_end=None,
        )

        # _query_versions uses get_neighbors → return existing version as a dict
        version_dict = _version_to_dict(existing_version)
        mock_graph.backend.get_neighbors.return_value = [version_dict]
        # _store_version: main node exists
        mock_graph.backend.get_node.return_value = {"item_id": "test123"}

        version = tracker.create_version(
            item_id="test123",
            content="Second version",
            changed_by="bob",
            change_reason="Update",
        )

        assert version.version_number == 2
        assert version.content == "Second version"
        assert version.changed_by == "bob"
        assert version.change_reason == "Update"
        assert version.previous_version == 1

    def test_get_versions_empty(self, tracker, mock_graph):
        """Test getting versions when none exist."""
        mock_graph.backend.get_neighbors.return_value = []
        mock_graph.backend.search_nodes.return_value = []

        versions = tracker.get_versions("nonexistent")

        assert isinstance(versions, list)
        assert len(versions) == 0

    def test_get_current_version(self, tracker, mock_graph):
        """Test getting current version."""
        current = Version(
            item_id="test123",
            version_number=2,
            content="Current",
            transaction_time_start=datetime.now(timezone.utc),
            transaction_time_end=None,
        )

        mock_graph.backend.get_neighbors.return_value = [_version_to_dict(current)]

        version = tracker.get_current_version("test123")

        assert version is not None
        assert version.version_number == 2
        assert version.is_current() is True

    def test_get_version_at_time(self, tracker, mock_graph):
        """Test getting version at specific time."""
        now = datetime.now(timezone.utc)

        v1 = Version(
            item_id="test123",
            version_number=1,
            content="Version 1",
            transaction_time_start=now - timedelta(hours=2),
            transaction_time_end=now - timedelta(hours=1),
        )

        v2 = Version(
            item_id="test123",
            version_number=2,
            content="Version 2",
            transaction_time_start=now - timedelta(hours=1),
            transaction_time_end=None,
        )

        mock_graph.backend.get_neighbors.return_value = [_version_to_dict(v1), _version_to_dict(v2)]

        # Get version 90 minutes ago (should be v1)
        version = tracker.get_version_at_time("test123", now - timedelta(minutes=90))

        assert version is not None
        assert version.version_number == 1

    def test_compare_versions(self, tracker, mock_graph):
        """Test comparing two versions."""
        now = datetime.now(timezone.utc)

        v1 = Version(
            item_id="test123",
            version_number=1,
            content="Original content",
            metadata={"key1": "value1"},
            transaction_time_start=now - timedelta(hours=1),
        )

        v2 = Version(
            item_id="test123",
            version_number=2,
            content="Updated content",
            metadata={"key1": "value1", "key2": "value2"},
            transaction_time_start=now,
            changed_by="alice",
            change_reason="Update",
        )

        mock_graph.backend.get_neighbors.return_value = [_version_to_dict(v1), _version_to_dict(v2)]

        comparison = tracker.compare_versions("test123", 1, 2)

        assert comparison["item_id"] == "test123"
        assert comparison["version1"] == 1
        assert comparison["version2"] == 2
        assert comparison["content_changed"] is True
        assert "metadata_changes" in comparison
        assert comparison["changed_by"] == "alice"
        assert comparison["change_reason"] == "Update"

    def test_version_caching(self, tracker, mock_graph):
        """Test that versions are cached."""
        v = Version(
            item_id="test123",
            version_number=1,
            content="Test",
            transaction_time_start=datetime.now(timezone.utc),
        )
        mock_graph.backend.get_neighbors.return_value = [_version_to_dict(v)]

        # First call
        versions1 = tracker.get_versions("test123")

        # Second call (should use cache)
        versions2 = tracker.get_versions("test123")

        # Should only query once (get_neighbors called once, not twice)
        assert mock_graph.backend.get_neighbors.call_count == 1
        assert versions1 == versions2
        assert len(versions1) == 1
        assert versions1[0].version_number == 1

    def test_fallback_to_search_nodes(self, tracker, mock_graph):
        """When get_neighbors returns empty, _query_versions falls back to search_nodes."""
        v = Version(
            item_id="test123",
            version_number=1,
            content="Fallback found",
            transaction_time_start=datetime.now(timezone.utc),
        )
        mock_graph.backend.get_neighbors.return_value = []
        mock_graph.backend.search_nodes.return_value = [_version_to_dict(v)]

        versions = tracker.get_versions("test123")

        assert len(versions) == 1
        assert versions[0].content == "Fallback found"
        mock_graph.backend.search_nodes.assert_called_once_with({"ref_item_id": "test123", "memory_type": "Version"})


class TestVersionTrackerSQLite:
    """VersionTracker integration tests with real SQLiteBackend (DIST-LITE-DEGRADE-1a).

    Validates that VersionTracker works end-to-end on SQLite — the core deliverable
    of this feature. No mocks, no Cypher, just real SQLite round-trips.
    """

    @pytest.fixture
    def sqlite_tracker(self):
        """Create a VersionTracker backed by in-memory SQLite."""
        from smartmemory.graph.backends.sqlite import SQLiteBackend
        from smartmemory.graph.smartgraph import SmartGraph

        backend = SQLiteBackend(":memory:")
        graph = SmartGraph(backend=backend)
        return VersionTracker(graph)

    def test_create_and_get_versions(self, sqlite_tracker):
        """Create 2 versions, retrieve, verify order (newest first)."""
        sqlite_tracker.create_version(item_id="item1", content="First content")
        sqlite_tracker.create_version(item_id="item1", content="Second content")

        versions = sqlite_tracker.get_versions("item1")

        assert len(versions) == 2
        assert versions[0].version_number == 2  # newest first
        assert versions[1].version_number == 1
        assert versions[0].content == "Second content"
        assert versions[1].content == "First content"

    def test_version_at_time(self, sqlite_tracker):
        """Create versions at different times, query by time range."""
        import time

        v1 = sqlite_tracker.create_version(item_id="item1", content="Original")
        time.sleep(0.01)
        v2 = sqlite_tracker.create_version(item_id="item1", content="Updated")

        # Query at a time between v1 and v2 creation — should get v1
        midpoint = v1.transaction_time_start + (v2.transaction_time_start - v1.transaction_time_start) / 2
        result = sqlite_tracker.get_version_at_time("item1", midpoint)

        assert result is not None
        assert result.version_number == 1

    def test_compare_versions(self, sqlite_tracker):
        """Create 2 versions with different content, compare metadata."""
        sqlite_tracker.create_version(
            item_id="item1",
            content="Original",
            metadata={"source": "test"},
        )
        sqlite_tracker.create_version(
            item_id="item1",
            content="Modified",
            metadata={"source": "test", "reviewed": True},
            changed_by="alice",
            change_reason="Review",
        )

        comparison = sqlite_tracker.compare_versions("item1", 1, 2)

        assert comparison["content_changed"] is True
        assert comparison["changed_by"] == "alice"
        assert comparison["change_reason"] == "Review"

    def test_placeholder_creation(self, sqlite_tracker):
        """Version for nonexistent item creates placeholder node."""
        # No item1 node exists yet
        sqlite_tracker.create_version(item_id="item1", content="First")

        # Placeholder node should now exist
        node = sqlite_tracker.backend.get_node("item1")
        assert node is not None
        assert node["memory_type"] == "semantic"

    def test_fallback_search(self, sqlite_tracker):
        """search_nodes({"ref_item_id": ...}) fallback works when no edges exist."""
        # Manually create a version node without an edge to test fallback
        from smartmemory.temporal.version_tracker import Version

        now = datetime.now(timezone.utc)
        v = Version(item_id="orphan", version_number=1, content="Orphan version", transaction_time_start=now)
        props = v.to_dict()
        props["ref_item_id"] = props.pop("item_id")
        metadata = props.pop("metadata", {})
        if isinstance(metadata, dict):
            for k, val in metadata.items():
                if k not in props:
                    props[k] = val
        sqlite_tracker.backend.add_node(item_id="version_orphan_1", properties=props, memory_type="Version")

        # Querying via get_neighbors will find nothing (no edge), fallback to search_nodes
        versions = sqlite_tracker.get_versions("orphan")

        assert len(versions) == 1
        assert versions[0].content == "Orphan version"

    def test_cross_workspace_no_overwrite(self, sqlite_tracker):
        """Node belonging to workspace A is not overwritten when workspace B creates a version."""
        # Manually create a node owned by workspace A
        sqlite_tracker.backend.add_node(
            "shared_item",
            {"content": "Workspace A data", "workspace_id": "ws_A", "memory_type": "semantic"},
        )

        # Create a scope_provider that reports workspace B
        scope_mock = Mock()
        scope_mock.get_isolation_filters.return_value = {"workspace_id": "ws_B"}
        scope_mock.get_write_context.return_value = {"workspace_id": "ws_B"}
        sqlite_tracker.scope_provider = scope_mock

        # Create a version — this should NOT overwrite the ws_A node
        sqlite_tracker.create_version(item_id="shared_item", content="Version from ws_B")

        # Verify ws_A's node is untouched
        node = sqlite_tracker.backend.get_node("shared_item")
        assert node["workspace_id"] == "ws_A"  # NOT overwritten to ws_B

        # Version node should still be created
        version_node = sqlite_tracker.backend.get_node("version_shared_item_1")
        assert version_node is not None

    def test_empty_versions(self, sqlite_tracker):
        """get_versions on nonexistent item returns empty list."""
        assert sqlite_tracker.get_versions("nonexistent") == []
