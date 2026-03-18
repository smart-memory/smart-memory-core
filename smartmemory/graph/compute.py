"""GraphComputeLayer — in-memory NetworkX MultiDiGraph synced with a SmartGraphBackend.

Provides the graph data structure that NetworkXAlgos delegates to for
algorithmic operations (centrality, path finding, community detection, etc.).

Uses nx.MultiDiGraph (not DiGraph) because SQLite's edge schema is keyed by
(source_id, target_id, edge_type) — parallel edges between the same node pair
are allowed.  Each edge_type becomes a separate key in the MultiDiGraph.

Sync contract:
- SQLiteBackend mutation methods call the corresponding sync_* method
  AFTER the SQL write succeeds.
- clear() → sync_clear()
- deserialize() → reload()  (re-reads all nodes/edges from SQL)
- add_node/add_edge → sync_add_node/sync_add_edge
- remove_node/remove_edge → sync_remove_node/sync_remove_edge
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict

import networkx as nx

if TYPE_CHECKING:
    from smartmemory.graph.backends.backend import SmartGraphBackend

logger = logging.getLogger(__name__)


class GraphComputeLayer:
    """In-memory MultiDiGraph mirroring a SmartGraphBackend's node/edge state."""

    # Keys that are extracted into positional/explicit args — exclude from
    # **kwargs spread to avoid "got multiple values for keyword argument" errors.
    _NODE_POS_KEYS = frozenset({"item_id", "id"})
    _EDGE_POS_KEYS = frozenset({"source_id", "target_id", "edge_type"})

    def __init__(self, backend: SmartGraphBackend) -> None:
        self._backend = backend
        self._graph = nx.MultiDiGraph()
        self._lock = threading.RLock()
        self._load_from_backend()

    def _load_from_backend(self) -> None:
        """Bulk-load all nodes and edges from the backend into the MultiDiGraph."""
        with self._lock:
            self._graph.clear()

            if hasattr(self._backend, "get_all_nodes"):
                for node in self._backend.get_all_nodes():
                    item_id = node.get("item_id") or node.get("id")
                    if item_id:
                        attrs = {k: v for k, v in node.items() if k not in self._NODE_POS_KEYS}
                        self._graph.add_node(item_id, **attrs)

            if hasattr(self._backend, "get_all_edges"):
                for edge in self._backend.get_all_edges():
                    src = edge.get("source_id")
                    tgt = edge.get("target_id")
                    etype = edge.get("edge_type", "")
                    if src and tgt:
                        attrs = {k: v for k, v in edge.items() if k not in self._EDGE_POS_KEYS}
                        self._graph.add_edge(src, tgt, key=etype, edge_type=etype, **attrs)

            logger.debug(
                "GraphComputeLayer loaded %d nodes, %d edges",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )

    # ── Incremental sync hooks (called by SQLiteBackend after SQL writes) ──

    def sync_add_node(self, item_id: str, properties: Dict[str, Any]) -> None:
        """Mirror a node addition into the MultiDiGraph."""
        attrs = {k: v for k, v in properties.items() if k not in self._NODE_POS_KEYS}
        with self._lock:
            self._graph.add_node(item_id, **attrs)

    def sync_add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Dict[str, Any],
    ) -> None:
        """Mirror an edge addition into the MultiDiGraph.

        Uses edge_type as the MultiDiGraph key so parallel edges between the
        same node pair (different types) coexist — matching SQLite's composite
        PK (source_id, target_id, edge_type).
        """
        attrs = {k: v for k, v in properties.items() if k not in self._EDGE_POS_KEYS}
        with self._lock:
            self._graph.add_edge(source_id, target_id, key=edge_type, edge_type=edge_type, **attrs)

    def sync_remove_node(self, item_id: str) -> None:
        """Mirror a node removal (and its edges) from the MultiDiGraph."""
        with self._lock:
            if self._graph.has_node(item_id):
                self._graph.remove_node(item_id)

    def sync_remove_edge(self, source_id: str, target_id: str, edge_type: str | None = None) -> None:
        """Mirror an edge removal from the MultiDiGraph.

        If edge_type is provided, removes only that specific parallel edge.
        If edge_type is None, removes ALL edges between the pair (matching
        SQLite's DELETE WHERE source_id=? AND target_id=? behaviour).
        """
        with self._lock:
            if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
                return
            if edge_type is not None:
                if self._graph.has_edge(source_id, target_id, key=edge_type):
                    self._graph.remove_edge(source_id, target_id, key=edge_type)
            else:
                while self._graph.has_edge(source_id, target_id):
                    self._graph.remove_edge(source_id, target_id)

    def sync_clear(self) -> None:
        """Wipe the MultiDiGraph (called by backend.clear())."""
        with self._lock:
            self._graph.clear()

    def reload(self) -> None:
        """Re-read all data from the backend (called by backend.deserialize())."""
        self._load_from_backend()

    # ── Thread-safe public access ──────────────────────────────────────

    def snapshot(self) -> nx.MultiDiGraph:
        """Return a frozen copy of the graph for algorithms needing full-graph access.

        The copy is safe to iterate without holding the lock — it won't see
        concurrent mutations, but that's fine for algorithmic passes (centrality,
        community detection) which operate on a point-in-time snapshot.
        """
        with self._lock:
            return self._graph.copy()

    def get_node(self, item_id: str) -> Dict[str, Any] | None:
        """Thread-safe node property lookup."""
        with self._lock:
            if self._graph.has_node(item_id):
                return dict(self._graph.nodes[item_id])
            return None

    def get_neighbors(self, item_id: str) -> list[Dict[str, Any]]:
        """Thread-safe 1-hop neighbor lookup. Returns list of node property dicts."""
        with self._lock:
            if not self._graph.has_node(item_id):
                return []
            return [
                {"item_id": n, **dict(self._graph.nodes[n])}
                for n in self._graph.neighbors(item_id)
            ]

    def has_node(self, item_id: str) -> bool:
        """Thread-safe node existence check."""
        with self._lock:
            return self._graph.has_node(item_id)

    def node_count(self) -> int:
        """Thread-safe node count."""
        with self._lock:
            return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Thread-safe edge count."""
        with self._lock:
            return self._graph.number_of_edges()
