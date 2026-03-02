"""Tests for DEGRADE-1b: temporal queries on SQLite (backend-agnostic).

Validates that TemporalQueries.at_time() and TemporalRelationshipQueries
use the backend-agnostic search_nodes/get_edges_for_node/get_all_edges
instead of raw Cypher, and that workspace filtering works correctly.
"""

import inspect
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock


# ---------------------------------------------------------------
# TemporalQueries.at_time() tests
# ---------------------------------------------------------------


class TestTemporalQueriesAtTime:
    """Tests for TemporalQueries.at_time() using search_nodes({})."""

    def _make_temporal_queries(self, nodes=None, scope_provider=None):
        """Build a TemporalQueries with mocked memory and version tracker."""
        from smartmemory.temporal.queries import TemporalQueries

        memory = MagicMock()
        memory.graph.search_nodes.return_value = nodes or []

        tq = TemporalQueries.__new__(TemporalQueries)
        tq.memory = memory
        tq.scope_provider = scope_provider
        tq.version_tracker = MagicMock()
        tq._parse_time = TemporalQueries._parse_time.__get__(tq)
        return tq

    def test_at_time_uses_search_nodes(self):
        """at_time() calls search_nodes({}) and extracts item_id from dicts."""
        now = datetime.now(timezone.utc)
        nodes = [
            {"item_id": "id1", "workspace_id": "ws1"},
            {"item_id": "id2", "workspace_id": "ws1"},
        ]
        tq = self._make_temporal_queries(nodes=nodes)

        mock_version = MagicMock()
        mock_version.version_number = 1
        tq.version_tracker.get_version_at_time.return_value = mock_version

        result = tq.at_time(now.isoformat())

        tq.memory.graph.search_nodes.assert_called_once_with({})
        assert len(result) == 2

    def test_at_time_deduplicates(self):
        """at_time() deduplicates by item_id (replaces Cypher DISTINCT)."""
        now = datetime.now(timezone.utc)
        nodes = [
            {"item_id": "id1", "workspace_id": "ws1"},
            {"item_id": "id1", "workspace_id": "ws1"},  # duplicate
            {"item_id": "id2", "workspace_id": "ws1"},
        ]
        tq = self._make_temporal_queries(nodes=nodes)

        mock_version = MagicMock()
        mock_version.version_number = 1
        tq.version_tracker.get_version_at_time.return_value = mock_version

        tq.at_time(now.isoformat())

        # Only 2 unique item_ids checked
        assert tq.version_tracker.get_version_at_time.call_count == 2

    def test_at_time_filters_workspace(self):
        """at_time() filters nodes by workspace_id when scope_provider is set."""
        now = datetime.now(timezone.utc)
        nodes = [
            {"item_id": "id1", "workspace_id": "ws1"},
            {"item_id": "id2", "workspace_id": "ws2"},  # different workspace
        ]
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws1"}
        tq = self._make_temporal_queries(nodes=nodes, scope_provider=scope_provider)

        mock_version = MagicMock()
        mock_version.version_number = 1
        tq.version_tracker.get_version_at_time.return_value = mock_version

        tq.at_time(now.isoformat())

        # Only id1 checked (id2 is in ws2, filtered out)
        assert tq.version_tracker.get_version_at_time.call_count == 1
        tq.version_tracker.get_version_at_time.assert_called_once()
        call_args = tq.version_tracker.get_version_at_time.call_args
        assert call_args[0][0] == "id1"


# ---------------------------------------------------------------
# TemporalRelationshipQueries tests
# ---------------------------------------------------------------


class TestTemporalRelationshipQueries:
    """Tests for refactored _query_relationships and at_time."""

    def _make_rel_queries(self, edges_for_node=None, all_edges=None, scope_provider=None):
        """Build a TemporalRelationshipQueries with mocked graph."""
        from smartmemory.temporal.relationships import TemporalRelationshipQueries

        graph = MagicMock()
        graph.get_edges_for_node.return_value = edges_for_node or []
        graph.get_all_edges.return_value = all_edges or []

        trq = TemporalRelationshipQueries(graph, scope_provider=scope_provider)
        return trq

    def _make_edge(
        self, source="s1", target="t1", edge_type="related_to", valid_from=None, valid_to=None, created_at=None
    ):
        """Build a normalized edge dict."""
        return {
            "source_id": source,
            "target_id": target,
            "edge_type": edge_type,
            "valid_from": valid_from or datetime.now(timezone.utc).isoformat(),
            "valid_to": valid_to,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "properties": {},
        }

    def test_query_relationships_uses_get_edges(self):
        """_query_relationships() calls get_edges_for_node and maps to TemporalRelationship."""
        edges = [self._make_edge(source="item1", target="item2")]
        trq = self._make_rel_queries(edges_for_node=edges)

        result = trq._query_relationships("item1")

        trq.graph.get_edges_for_node.assert_called_once_with("item1")
        assert len(result) == 1
        assert result[0].source_id == "item1"
        assert result[0].target_id == "item2"
        assert result[0].relationship_type == "related_to"

    def test_query_relationships_direction_filter_outgoing(self):
        """Direction=outgoing filters to edges where source_id == item_id."""
        edges = [
            self._make_edge(source="item1", target="item2"),  # outgoing from item1
            self._make_edge(source="item3", target="item1"),  # incoming to item1
        ]
        trq = self._make_rel_queries(edges_for_node=edges)

        result = trq._query_relationships("item1", direction="outgoing")

        assert len(result) == 1
        assert result[0].target_id == "item2"

    def test_query_relationships_direction_filter_incoming(self):
        """Direction=incoming filters to edges where target_id == item_id."""
        edges = [
            self._make_edge(source="item1", target="item2"),  # outgoing
            self._make_edge(source="item3", target="item1"),  # incoming
        ]
        trq = self._make_rel_queries(edges_for_node=edges)

        result = trq._query_relationships("item1", direction="incoming")

        assert len(result) == 1
        assert result[0].source_id == "item3"

    def test_query_relationships_workspace_filter(self):
        """_query_relationships() filters out edges to nodes in different workspaces."""
        edges = [
            self._make_edge(source="item1", target="item2"),
            self._make_edge(source="item1", target="item3"),  # item3 in different ws
        ]
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws1"}

        graph = MagicMock()
        graph.get_edges_for_node.return_value = edges
        graph.backend.get_node.side_effect = lambda nid: {
            "item2": {"item_id": "item2", "workspace_id": "ws1"},
            "item3": {"item_id": "item3", "workspace_id": "ws2"},
        }.get(nid, {})

        from smartmemory.temporal.relationships import TemporalRelationshipQueries

        trq = TemporalRelationshipQueries(graph, scope_provider=scope_provider)
        result = trq._query_relationships("item1")

        # item3 is in ws2, should be filtered out
        assert len(result) == 1
        assert result[0].target_id == "item2"

    def test_query_relationships_rejects_nodes_without_workspace(self):
        """_query_relationships() rejects partner nodes with missing workspace_id (P1 regression)."""
        edges = [
            self._make_edge(source="item1", target="item2"),
            self._make_edge(source="item1", target="item3"),  # item3 has no workspace_id
        ]
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws1"}

        graph = MagicMock()
        graph.get_edges_for_node.return_value = edges
        graph.backend.get_node.side_effect = lambda nid: {
            "item2": {"item_id": "item2", "workspace_id": "ws1"},
            "item3": {"item_id": "item3"},  # no workspace_id key
        }.get(nid, {})

        from smartmemory.temporal.relationships import TemporalRelationshipQueries

        trq = TemporalRelationshipQueries(graph, scope_provider=scope_provider)
        result = trq._query_relationships("item1")

        # item3 has no workspace_id — must be rejected (matches Cypher WHERE behavior)
        assert len(result) == 1
        assert result[0].target_id == "item2"

    def test_at_time_rejects_nodes_without_workspace(self):
        """at_time() rejects source nodes with missing workspace_id (P1 regression)."""
        now = datetime.now(timezone.utc)
        edges = [
            self._make_edge(source="s1", target="t1", valid_from=(now - timedelta(hours=1)).isoformat()),
            self._make_edge(source="s_unscoped", target="t2", valid_from=(now - timedelta(hours=1)).isoformat()),
        ]
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws1"}

        graph = MagicMock()
        graph.get_all_edges.return_value = edges
        graph.backend.get_node.side_effect = lambda nid: {
            "s1": {"item_id": "s1", "workspace_id": "ws1"},
            "s_unscoped": {"item_id": "s_unscoped"},  # no workspace_id
        }.get(nid, {})

        from smartmemory.temporal.relationships import TemporalRelationshipQueries

        trq = TemporalRelationshipQueries(graph, scope_provider=scope_provider)
        result = trq.at_time(now)

        # s_unscoped has no workspace_id — must be rejected
        assert len(result) == 1
        assert result[0].source_id == "s1"

    def test_get_relationship_history_rejects_nodes_without_workspace(self):
        """get_relationship_history() rejects target nodes with missing workspace_id (P1 regression)."""
        now = datetime.now(timezone.utc)
        edges = [
            self._make_edge(
                source="item1",
                target="item2",
                valid_from=(now - timedelta(days=5)).isoformat(),
                created_at=(now - timedelta(days=5)).isoformat(),
            ),
        ]
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws1"}

        graph = MagicMock()
        graph.get_edges_for_node.return_value = edges
        # item2 has no workspace_id — must be rejected
        graph.backend.get_node.return_value = {"item_id": "item2"}

        from smartmemory.temporal.relationships import TemporalRelationshipQueries

        trq = TemporalRelationshipQueries(graph, scope_provider=scope_provider)
        result = trq.get_relationship_history("item1", "item2")

        assert len(result) == 0

    def test_relationship_at_time_uses_get_all_edges(self):
        """Global at_time() calls get_all_edges() and filters by is_valid_at."""
        now = datetime.now(timezone.utc)
        edges = [
            self._make_edge(source="s1", target="t1", valid_from=(now - timedelta(hours=1)).isoformat()),
            self._make_edge(source="s2", target="t2", valid_from=(now + timedelta(hours=1)).isoformat()),  # future
        ]
        trq = self._make_rel_queries(all_edges=edges)

        result = trq.at_time(now)

        trq.graph.get_all_edges.assert_called_once()
        assert len(result) == 1
        assert result[0].source_id == "s1"

    def test_relationship_at_time_workspace_filter(self):
        """Global at_time() filters edges by source node's workspace."""
        now = datetime.now(timezone.utc)
        edges = [
            self._make_edge(source="s1", target="t1", valid_from=(now - timedelta(hours=1)).isoformat()),
            self._make_edge(source="s2", target="t2", valid_from=(now - timedelta(hours=1)).isoformat()),
        ]
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws1"}

        graph = MagicMock()
        graph.get_all_edges.return_value = edges
        graph.backend.get_node.side_effect = lambda nid: {
            "s1": {"item_id": "s1", "workspace_id": "ws1"},
            "s2": {"item_id": "s2", "workspace_id": "ws2"},
        }.get(nid, {})

        from smartmemory.temporal.relationships import TemporalRelationshipQueries

        trq = TemporalRelationshipQueries(graph, scope_provider=scope_provider)
        result = trq.at_time(now)

        # s2 is in ws2, should be filtered out
        assert len(result) == 1
        assert result[0].source_id == "s1"


class TestNoOpStubRemoved:
    """Verify _NoOpTemporalQueries is fully removed from factory.py."""

    def test_no_op_stub_removed(self):
        """create_lite_memory source should not contain '_NoOpTemporalQueries'."""
        from smartmemory.tools.factory import create_lite_memory

        source = inspect.getsource(create_lite_memory)
        assert "_NoOpTemporalQueries" not in source, "create_lite_memory still references _NoOpTemporalQueries"

    def test_factory_module_no_noop_class(self):
        """The factory module should not define _NoOpTemporalQueries."""
        import smartmemory.tools.factory as factory_module

        assert not hasattr(factory_module, "_NoOpTemporalQueries"), (
            "_NoOpTemporalQueries class still exists in factory module"
        )
