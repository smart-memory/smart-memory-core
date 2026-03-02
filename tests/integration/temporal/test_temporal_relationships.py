"""
Integration tests for temporal relationship queries.

Tests TemporalRelationshipQueries interactions with graph backend.
Uses mocked graph to simulate backend query/response patterns via the
backend-agnostic edge API (get_edges_for_node, get_all_edges).

Updated for DEGRADE-1b: replaced execute_query mocks with edge API mocks.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

import pytest

from smartmemory.temporal.relationships import (
    TemporalRelationshipQueries,
    TemporalRelationship,
)

pytestmark = [pytest.mark.integration]


def _make_edge(source, target, rel_type="RELATED_TO", properties=None, valid_from=None, valid_to=None, created_at=None):
    """Build a normalized edge dict matching the backend-agnostic contract."""
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "source_id": source,
        "target_id": target,
        "edge_type": rel_type,
        "properties": properties or {},
        "valid_from": valid_from or now_iso,
        "valid_to": valid_to,
        "created_at": created_at or now_iso,
    }


class TestTemporalRelationship:
    """Test TemporalRelationship dataclass."""

    def test_relationship_creation(self):
        """Test creating a temporal relationship."""
        now = datetime.now(timezone.utc)
        rel = TemporalRelationship(
            source_id="item1",
            target_id="item2",
            relationship_type="RELATED_TO",
            properties={"strength": 0.9},
            valid_time_start=now,
        )

        assert rel.source_id == "item1"
        assert rel.target_id == "item2"
        assert rel.relationship_type == "RELATED_TO"
        assert rel.properties["strength"] == 0.9

    def test_is_valid_at(self):
        """Test checking if relationship is valid at a time."""
        now = datetime.now(timezone.utc)

        rel = TemporalRelationship(
            source_id="item1",
            target_id="item2",
            relationship_type="RELATED_TO",
            valid_time_start=now,
            valid_time_end=now + timedelta(days=1),
        )

        # Before valid time
        assert rel.is_valid_at(now - timedelta(hours=1)) is False

        # During valid time
        assert rel.is_valid_at(now + timedelta(hours=12)) is True

        # After valid time
        assert rel.is_valid_at(now + timedelta(days=2)) is False

    def test_overlaps_with(self):
        """Test checking if relationships overlap."""
        now = datetime.now(timezone.utc)

        rel1 = TemporalRelationship(
            source_id="item1",
            target_id="item2",
            relationship_type="RELATED_TO",
            valid_time_start=now,
            valid_time_end=now + timedelta(days=5),
        )

        # Overlapping relationship
        rel2 = TemporalRelationship(
            source_id="item1",
            target_id="item3",
            relationship_type="RELATED_TO",
            valid_time_start=now + timedelta(days=3),
            valid_time_end=now + timedelta(days=7),
        )

        assert rel1.overlaps_with(rel2) is True
        assert rel2.overlaps_with(rel1) is True

        # Non-overlapping relationship
        rel3 = TemporalRelationship(
            source_id="item1",
            target_id="item4",
            relationship_type="RELATED_TO",
            valid_time_start=now + timedelta(days=10),
            valid_time_end=now + timedelta(days=15),
        )

        assert rel1.overlaps_with(rel3) is False
        assert rel3.overlaps_with(rel1) is False


class TestTemporalRelationshipQueries:
    """Test TemporalRelationshipQueries class."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock graph backend with edge API methods."""
        graph = Mock()
        graph.get_edges_for_node = Mock(return_value=[])
        graph.get_all_edges = Mock(return_value=[])
        return graph

    @pytest.fixture
    def rel_queries(self, mock_graph):
        """Create temporal relationship queries instance."""
        return TemporalRelationshipQueries(mock_graph)

    def test_initialization(self, rel_queries):
        """Test initialization."""
        assert rel_queries is not None
        assert rel_queries.graph is not None

    def test_get_relationships_at_time(self, rel_queries, mock_graph):
        """Test getting relationships at a specific time."""
        now = datetime.now(timezone.utc)

        # Mock edges via get_edges_for_node (used by _query_relationships)
        mock_graph.get_edges_for_node.return_value = [
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                valid_from=(now - timedelta(days=10)).isoformat(),
                valid_to=(now + timedelta(days=10)).isoformat(),
                created_at=(now - timedelta(days=10)).isoformat(),
            )
        ]

        rels = rel_queries.get_relationships_at_time("item1", now)

        assert isinstance(rels, list)
        assert len(rels) == 1
        assert rels[0].source_id == "item1"
        assert rels[0].target_id == "item2"

    def test_get_relationship_history(self, rel_queries, mock_graph):
        """Test getting relationship history."""
        now = datetime.now(timezone.utc)

        mock_graph.get_edges_for_node.return_value = [
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                properties={"version": 1},
                valid_from=(now - timedelta(days=10)).isoformat(),
                valid_to=(now - timedelta(days=5)).isoformat(),
                created_at=(now - timedelta(days=10)).isoformat(),
            ),
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                properties={"version": 2},
                valid_from=(now - timedelta(days=5)).isoformat(),
                valid_to=None,
                created_at=(now - timedelta(days=5)).isoformat(),
            ),
        ]

        history = rel_queries.get_relationship_history("item1", "item2")

        assert isinstance(history, list)
        assert len(history) == 2
        # Sorted by transaction_time_start descending
        assert history[0].properties["version"] == 2
        assert history[1].properties["version"] == 1

    def test_find_temporal_patterns(self, rel_queries, mock_graph):
        """Test finding temporal patterns."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)
        end = now

        # Mock edges via get_edges_for_node (used by _query_relationships)
        mock_graph.get_edges_for_node.return_value = [
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                valid_from=(start + timedelta(days=5)).isoformat(),
                valid_to=None,
                created_at=(start + timedelta(days=5)).isoformat(),
            ),
            _make_edge(
                source="item1",
                target="item3",
                rel_type="MENTIONS",
                valid_from=(start + timedelta(days=10)).isoformat(),
                valid_to=(start + timedelta(days=20)).isoformat(),
                created_at=(start + timedelta(days=10)).isoformat(),
            ),
        ]

        patterns = rel_queries.find_temporal_patterns("item1", start, end)

        assert isinstance(patterns, dict)
        assert "total_relationships" in patterns
        assert "relationship_types" in patterns
        assert patterns["total_relationships"] == 2

    def test_temporal_join_overlap(self, rel_queries, mock_graph):
        """Test temporal join with overlap."""
        now = datetime.now(timezone.utc)

        # Mock get_edges_for_node to return different edges per item_id
        def mock_edges(item_id):
            if item_id == "item1":
                return [
                    _make_edge(
                        source="item1",
                        target="item2",
                        rel_type="RELATED_TO",
                        valid_from=now.isoformat(),
                        valid_to=(now + timedelta(days=5)).isoformat(),
                        created_at=now.isoformat(),
                    )
                ]
            elif item_id == "item3":
                return [
                    _make_edge(
                        source="item3",
                        target="item4",
                        rel_type="MENTIONS",
                        valid_from=(now + timedelta(days=3)).isoformat(),
                        valid_to=(now + timedelta(days=7)).isoformat(),
                        created_at=now.isoformat(),
                    )
                ]
            return []

        mock_graph.get_edges_for_node.side_effect = mock_edges

        results = rel_queries.temporal_join(["item1", "item3"], now, now + timedelta(days=10), join_type="overlap")

        assert isinstance(results, list)
        # Should find overlapping relationships

    def test_find_co_occurring_relationships(self, rel_queries, mock_graph):
        """Test finding co-occurring relationships."""
        now = datetime.now(timezone.utc)

        # Mock edges created close together
        mock_graph.get_edges_for_node.return_value = [
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                valid_from=now.isoformat(),
                created_at=now.isoformat(),
            ),
            _make_edge(
                source="item1",
                target="item3",
                rel_type="MENTIONS",
                valid_from=(now + timedelta(minutes=30)).isoformat(),
                created_at=(now + timedelta(minutes=30)).isoformat(),
            ),
        ]

        groups = rel_queries.find_co_occurring_relationships("item1", time_window=3600)

        assert isinstance(groups, list)
        # Should find relationships created within 1 hour
        if groups:
            assert groups[0]["count"] == 2


class TestTemporalRelationshipIntegration:
    """Integration tests for temporal relationships."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock graph backend with edge API methods."""
        graph = Mock()
        graph.get_edges_for_node = Mock(return_value=[])
        graph.get_all_edges = Mock(return_value=[])
        return graph

    @pytest.fixture
    def rel_queries(self, mock_graph):
        """Create temporal relationship queries instance."""
        return TemporalRelationshipQueries(mock_graph)

    def test_end_to_end_relationship_tracking(self, rel_queries, mock_graph):
        """Test complete relationship tracking workflow."""
        now = datetime.now(timezone.utc)

        # Mock edges for get_relationship_history (get_edges_for_node)
        history_edges = [
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                properties={"strength": 0.5},
                valid_from=(now - timedelta(days=10)).isoformat(),
                valid_to=(now - timedelta(days=5)).isoformat(),
                created_at=(now - timedelta(days=10)).isoformat(),
            ),
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                properties={"strength": 0.9},
                valid_from=(now - timedelta(days=5)).isoformat(),
                valid_to=None,
                created_at=(now - timedelta(days=5)).isoformat(),
            ),
        ]
        mock_graph.get_edges_for_node.return_value = history_edges

        # Get history
        history = rel_queries.get_relationship_history("item1", "item2")

        assert len(history) == 2
        # Sorted by transaction_time_start descending (newer first)
        assert history[0].properties["strength"] == 0.9
        assert history[1].properties["strength"] == 0.5

        # Get relationships at a time in the first version's range
        # _query_relationships also uses get_edges_for_node
        mock_graph.get_edges_for_node.return_value = [
            _make_edge(
                source="item1",
                target="item2",
                rel_type="RELATED_TO",
                properties={"strength": 0.5},
                valid_from=(now - timedelta(days=10)).isoformat(),
                valid_to=(now - timedelta(days=5)).isoformat(),
                created_at=(now - timedelta(days=10)).isoformat(),
            ),
        ]

        rels_at_old_time = rel_queries.get_relationships_at_time("item1", now - timedelta(days=7))

        # Should find the old relationship (valid from -10d to -5d, querying at -7d)
        assert len(rels_at_old_time) == 1
        assert rels_at_old_time[0].properties["strength"] == 0.5
