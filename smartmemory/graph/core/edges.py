"""
SmartGraph Edge Operations Module

Handles all edge-related operations for the SmartGraph system.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from smartmemory.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class SmartGraphEdges:
    """Handles all edge-related operations for SmartGraph."""

    def __init__(self, backend, nodes_module, enable_caching=True, cache_size=1000):
        self.backend = backend
        self.nodes = nodes_module  # Reference to nodes module for validation
        self.enable_caching = enable_caching
        self.cache_size = cache_size

    def clear_cache(self) -> None:
        """Clear edge cache. No-op for now (edges are not cached)."""
        pass

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,
        transaction_time: Optional[Tuple] = None,
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ):
        """Add an edge to the graph."""
        # Structural validation only (no ontology enforcement)
        try:
            self._validate_edge_structural(source_id, target_id, edge_type, properties)
        except Exception as e:
            # Never fail the operation due to validation; just log and proceed
            logger.debug(f"Structural edge validation encountered an error and was skipped: {e}")

        # Enforce referential integrity: do not create edges to non-existent nodes
        try:
            source_exists = self.nodes.get_node(source_id) is not None
        except Exception as e:
            logger.debug(f"Error checking existence of source node {source_id}: {e}")
            source_exists = False
        try:
            target_exists = self.nodes.get_node(target_id) is not None
        except Exception as e:
            logger.debug(f"Error checking existence of target node {target_id}: {e}")
            target_exists = False

        if not source_exists or not target_exists:
            logger.warning(
                f"Skipping edge creation: missing nodes (source_exists={source_exists}, target_exists={target_exists})"
            )
            return {
                "edge_created": False,
                "error": "missing_node",
                "source_id": source_id,
                "target_id": target_id,
                "source_exists": source_exists,
                "target_exists": target_exists,
                "edge_type": edge_type,
            }

        result = self.backend.add_edge(
            source_id,
            target_id,
            edge_type,
            properties,
            valid_time,
            transaction_time,
            memory_type,
            is_global=is_global,
        )

        # Invalidate neighbor caches
        if self.enable_caching and hasattr(self.nodes, "_node_cache"):
            # Clear any cached neighbors for source and target nodes
            keys_to_remove = [
                key
                for key in self.nodes._node_cache.keys()
                if key.startswith(f"{source_id}:") or key.startswith(f"{target_id}:")
            ]
            for key in keys_to_remove:
                del self.nodes._node_cache[key]

        # Emit graph stats update
        try:
            self._emit_graph_stats(
                "add_edge",
                details={"source_id": source_id, "target_id": target_id, "edge_type": edge_type},
                delta_nodes=0,
                delta_edges=1,
            )
        except Exception:
            pass

        return result

    def add_triple(self, triple: Any, properties: Dict[str, Any] = None, **kwargs):
        """Add a triple (Triple models) to the graph. Properties are attached to the edge."""
        if properties is None:
            properties = {}
        self.nodes.add_node(triple.subject, {})
        self.nodes.add_node(triple.object, {})
        self.add_edge(triple.subject, triple.object, triple.predicate, properties, **kwargs)

    def get_edges_for_node(self, item_id: str):
        """Get all edges (relationships) involving a specific node.

        Delegates to the backend's get_edges_for_node method if available.
        Returns a list of edge dictionaries with 'source', 'target', and 'type' keys.
        """
        if hasattr(self.backend, "get_edges_for_node"):
            return self.backend.get_edges_for_node(item_id)
        else:
            raise NotImplementedError("Triple extraction not supported for this graph backend.")

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None):
        """Remove an edge from the graph."""
        result = self.backend.remove_edge(source_id, target_id, edge_type)
        try:
            self._emit_graph_stats(
                "remove_edge",
                details={"source_id": source_id, "target_id": target_id, "edge_type": edge_type or ""},
                delta_nodes=0,
                delta_edges=-1,
            )
        except Exception:
            pass
        return result

    def _validate_edge_structural(
        self, source_id: str, target_id: str, edge_type: str, properties: Optional[Dict[str, Any]]
    ) -> bool:
        """Lightweight structural validation for edges.

        Checks only the shape and serializability of inputs without enforcing ontology.
        Logs warnings on issues but does not raise, returning a boolean indicating validity.
        """
        is_valid = True

        # Basic ID checks
        if not source_id or not isinstance(source_id, str):
            logger.warning("Edge validation: invalid or missing source_id")
            is_valid = False
        if not target_id or not isinstance(target_id, str):
            logger.warning("Edge validation: invalid or missing target_id")
            is_valid = False

        # Ensure nodes exist (best-effort) via abstracted nodes API (uses retrieval path and caching)
        try:
            if source_id and not self.nodes.get_node(source_id):
                logger.warning(f"Edge validation: source node {source_id} not found")
                is_valid = False
        except Exception as e:
            logger.debug(f"Edge validation: could not verify source node {source_id}: {e}")
        try:
            if target_id and not self.nodes.get_node(target_id):
                logger.warning(f"Edge validation: target node {target_id} not found")
                is_valid = False
        except Exception as e:
            logger.debug(f"Edge validation: could not verify target node {target_id}: {e}")

        # Edge type must be a non-empty string
        if not isinstance(edge_type, str) or not edge_type.strip():
            logger.warning("Edge validation: edge_type must be a non-empty string")
            is_valid = False

        # Properties must be a dict; check keys and JSON-serializability of values
        if properties is None:
            properties = {}
        if not isinstance(properties, dict):
            logger.warning("Edge validation: properties must be a dict")
            is_valid = False
        else:
            for k, v in list(properties.items()):
                if not isinstance(k, str):
                    logger.warning(f"Edge validation: property key {k!r} is not a string")
                    is_valid = False
                # Ensure values are JSON-serializable (avoid unhashable errors elsewhere)
                try:
                    json.dumps(v, default=str)
                except TypeError as e:
                    logger.warning(f"Edge validation: property {k!r} value not JSON-serializable: {e}")
                    is_valid = False

        return is_valid

    def _emit_graph_stats(
        self,
        operation: str,
        details: Dict[str, Any],
        delta_nodes: Optional[int] = None,
        delta_edges: Optional[int] = None,
    ) -> None:
        """Best-effort emission of graph stats update events."""
        try:
            backend_name = type(self.backend).__name__
            node_count: Optional[int] = None
            edge_count: Optional[int] = None

            # Prefer explicit fast counters if backend provides them
            if hasattr(self.backend, "get_counts"):
                try:
                    counts = self.backend.get_counts()  # type: ignore[attr-defined]
                    if isinstance(counts, dict):
                        node_count = counts.get("node_count")
                        edge_count = counts.get("edge_count")
                except Exception:
                    pass
            else:
                # Fallbacks
                try:
                    if hasattr(self.backend, "get_node_count"):
                        node_count = self.backend.get_node_count()  # type: ignore[attr-defined]
                    elif hasattr(self.backend, "get_all_nodes"):
                        nodes = self.backend.get_all_nodes()  # type: ignore[attr-defined]
                        try:
                            node_count = len(nodes) if nodes is not None else None
                        except Exception:
                            node_count = None
                except Exception:
                    node_count = None
                try:
                    if hasattr(self.backend, "get_edge_count"):
                        edge_count = self.backend.get_edge_count()  # type: ignore[attr-defined]
                except Exception:
                    edge_count = None

            data: Dict[str, Any] = {
                "backend": backend_name,
                "node_count": node_count,
                "edge_count": edge_count,
                "delta_nodes": delta_nodes,
                "delta_edges": delta_edges,
                "details": details or {},
            }
            with trace_span("graph.stats_update", {**data, "operation": operation}):
                pass
        except Exception:
            # Observability must never break graph operations
            pass
