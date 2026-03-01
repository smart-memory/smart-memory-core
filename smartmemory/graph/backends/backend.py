from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class SmartGraphBackend(ABC):
    """Abstract base class for graph storage backends."""

    @abstractmethod
    def add_node(
        self,
        item_id: Optional[str],
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,
        created_at: Optional[Tuple] = None,
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ) -> Dict[str, Any]:
        """Add a node with properties, bi-temporal info, and memory type."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        ...

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,
        created_at: Optional[Tuple] = None,
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ) -> bool:
        """Add an edge with properties, bi-temporal info, and memory type."""
        ...

    @abstractmethod
    def get_node(self, item_id: str, as_of_time: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a node by ID, optionally as of a specific time."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        item_id: str,
        edge_type: Optional[str] = None,
        as_of_time: Optional[str] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes, optionally filtered by edge type, time, and direction.

        Args:
            item_id: The node to find neighbors for.
            edge_type: Optional edge type filter.
            as_of_time: Optional temporal filter.
            direction: Edge traversal direction — ``"both"`` (default, backward-compatible),
                ``"outgoing"`` (edges where item_id is the source), or ``"incoming"``
                (edges where item_id is the target).
        """
        ...

    @abstractmethod
    def remove_node(self, item_id: str) -> bool:
        """Remove a node by ID."""
        ...

    @abstractmethod
    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None) -> bool:
        """Remove an edge by source, target, and optionally edge type."""
        ...

    @abstractmethod
    def search_nodes(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find nodes matching query properties (type, time, etc)."""
        ...

    @abstractmethod
    def serialize(self) -> Any:
        """Serialize the graph (for export, backup, or test snapshot)."""
        ...

    @abstractmethod
    def deserialize(self, data: Any) -> None:
        """Load the graph from a serialized format."""
        ...

    def add_nodes_bulk(self, nodes: List[Dict[str, Any]], batch_size: int = 500, is_global: bool = False) -> int:
        """Bulk upsert nodes. Default: loop over add_node(). Override for performance."""
        count = 0
        for node in nodes:
            item_id = node.get("item_id")
            memory_type = node.get("memory_type")
            self.add_node(item_id, node, memory_type=memory_type, is_global=is_global)
            count += 1
        return count

    def add_edges_bulk(
        self, edges: List[Tuple[str, str, str, Dict[str, Any]]], batch_size: int = 500, is_global: bool = False
    ) -> int:
        """Bulk upsert edges. Default: loop over add_edge(). Override for performance."""
        count = 0
        for src, tgt, etype, props in edges:
            self.add_edge(src, tgt, etype, props, is_global=is_global)
            count += 1
        return count
