import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from smartmemory.graph.core.edges import SmartGraphEdges
from smartmemory.graph.core.nodes import SmartGraphNodes
from smartmemory.graph.core.search import SmartGraphSearch
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.utils import get_config
from smartmemory.interfaces import ScopeProvider


class SmartGraph:
    """
    Unified graph API for agentic memory. Converts backend-native dicts to the canonical models (item_cls) provided at construction.
    Backend is selected based on the 'backend_class' key in config.json['graph_db'] (default: FalkorDBBackend).
    """

    def __init__(
        self,
        item_cls=None,
        enable_caching=True,
        cache_size=1000,
        scope_provider: Optional[ScopeProvider] = None,
        backend=None,
    ):
        if item_cls is None:
            item_cls = MemoryItem
        self.item_cls = item_cls
        if backend is not None:
            self.backend = backend
        else:
            backend_cls = self._get_backend_class()
            # Inject scope provider into backend
            self.backend = backend_cls(scope_provider=scope_provider)

        # Initialize submodules
        self.nodes = SmartGraphNodes(self.backend, item_cls, enable_caching, cache_size)
        self.edges = SmartGraphEdges(self.backend, self.nodes, enable_caching, cache_size)
        self.search = SmartGraphSearch(self.backend, self.nodes, enable_caching, cache_size)

        # Performance caching (for backward compatibility)
        self.enable_caching = enable_caching
        self.cache_size = cache_size

    @staticmethod
    def _get_backend_class():
        """
        Resolve the backend class from config. Import directly from backends module.
        """
        graph_cfg = get_config("graph_db")
        backend_class_name = graph_cfg.get("backend_class", "FalkorDBBackend")

        # Map backend class names to their module paths
        backend_modules = {
            "FalkorDBBackend": "smartmemory.graph.backends.falkordb",
            "Neo4jBackend": "smartmemory.graph.backends.neo4j",
            "SQLiteBackend": "smartmemory.graph.backends.sqlite",
        }

        module_path = backend_modules.get(backend_class_name)
        if not module_path:
            raise ImportError(
                f"Unknown backend class '{backend_class_name}'. Available: {list(backend_modules.keys())}"
            )

        try:
            module = importlib.import_module(module_path)
            return getattr(module, backend_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import backend class '{backend_class_name}' from '{module_path}': {e}") from e

    def _emit_graph_stats(
        self,
        operation: str,
        details: Dict[str, Any],
        delta_nodes: Optional[int] = None,
        delta_edges: Optional[int] = None,
    ) -> None:
        """Best-effort emission of graph stats update events."""
        # Delegate to nodes module for consistency
        self.nodes._emit_graph_stats(operation, details, delta_nodes, delta_edges)

    def clear(self):
        """Remove all nodes and edges from the graph."""
        # Capture pre-clear counts if possible
        pre_nodes: Optional[int] = None
        pre_edges: Optional[int] = None
        try:
            if hasattr(self.backend, "get_counts"):
                counts = self.backend.get_counts()  # type: ignore[attr-defined]
                if isinstance(counts, dict):
                    pre_nodes = counts.get("node_count")
                    pre_edges = counts.get("edge_count")
            else:
                if hasattr(self.backend, "get_node_count"):
                    pre_nodes = self.backend.get_node_count()  # type: ignore[attr-defined]
                elif hasattr(self.backend, "get_all_nodes"):
                    nodes = self.backend.get_all_nodes()  # type: ignore[attr-defined]
                    pre_nodes = len(nodes) if nodes is not None else None
                if hasattr(self.backend, "get_edge_count"):
                    pre_edges = self.backend.get_edge_count()  # type: ignore[attr-defined]
        except Exception:
            pass

        with trace_span("graph.clear_all", {"node_count": pre_nodes, "edge_count": pre_edges}):
            result = self.backend.clear()

        # Clear all submodule caches
        self.nodes.clear_cache()
        self.edges.clear_cache()
        self.search.clear_cache()

        try:
            self._emit_graph_stats(
                "clear",
                details={"pre_counts": {"node_count": pre_nodes, "edge_count": pre_edges}},
                delta_nodes=(-pre_nodes if isinstance(pre_nodes, int) else None),
                delta_edges=(-pre_edges if isinstance(pre_edges, int) else None),
            )
        except Exception:
            pass

        return result

    def add_node(
        self,
        item_id: Optional[str],
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,
        transaction_time: Optional[Tuple] = None,
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ):
        """Add a node to the graph."""
        with trace_span("graph.add_node", {
            "memory_id": item_id,
            "memory_type": memory_type or properties.get("memory_type", ""),
            "label": properties.get("title", properties.get("content", ""))[:40] if properties else "",
            "content": properties.get("content", "")[:200] if properties else "",
        }):
            result = self.nodes.add_node(item_id, properties, valid_time, transaction_time, memory_type, is_global)
        return result

    def add_dual_node(
        self,
        item_id: str,
        memory_properties: Dict[str, Any],
        memory_type: str,
        entity_nodes: List[Dict[str, Any]] = None,
        is_global: bool = False,
    ):
        """Add a dual-node structure through the backend.

        Emits individual observability spans for each entity node and
        structural edge created atomically in the backend Cypher query,
        so that real-time UI consumers (viewer WebSocket) can display
        them progressively via the WebSocket relay.
        """
        with trace_span("graph.add_dual_node", {
            "memory_id": item_id,
            "memory_type": memory_type or "",
            "label": memory_properties.get("title", memory_properties.get("content", ""))[:40]
            if memory_properties else "",
            "content": memory_properties.get("content", "")[:200] if memory_properties else "",
            "entity_count": len(entity_nodes) if entity_nodes else 0,
        }) as span:
            result = self.nodes.add_dual_node(item_id, memory_properties, memory_type, entity_nodes, is_global)

            # Emit memory node event so the viewer shows it during streaming
            memory_label = (
                (memory_properties.get("title") or memory_properties.get("content", "")[:40] or item_id[:12])
                if memory_properties else item_id[:12]
            )
            span.emit_event("graph.add_node", {
                "memory_id": item_id,
                "memory_type": memory_type or "semantic",
                "label": memory_label,
                "content": memory_properties.get("content", "")[:200] if memory_properties else "",
            })

            # Emit per-entity and per-edge events attached to this span so they
            # carry trace_id/span_id and can be correlated to the pipeline stage.
            if entity_nodes:
                try:
                    resolved = result.get("resolved_entities", [])
                    entity_ids = result.get("entity_node_ids", [])
                    resolved_map = {r["index"]: r for r in resolved}

                    for i, entity_node in enumerate(entity_nodes):
                        entity_id = entity_ids[i] if i < len(entity_ids) else None
                        if not entity_id:
                            continue
                        entity_type = entity_node.get("entity_type", "Entity")
                        props = entity_node.get("properties") or {}
                        label = props.get("name") or props.get("content", entity_id)[:40]

                        span.emit_event("graph.add_node", {
                            "memory_id": entity_id,
                            "memory_type": entity_type.lower(),
                            "label": label[:40],
                            "content": props.get("content", label)[:200],
                            "parent_memory_id": item_id,
                            "resolved": resolved_map.get(i, {}).get("resolved", False),
                        })
                        span.emit_event("graph.add_edge", {
                            "source_id": item_id,
                            "target_id": entity_id,
                            "edge_type": "CONTAINS_ENTITY",
                        })
                        span.emit_event("graph.add_edge", {
                            "source_id": entity_id,
                            "target_id": item_id,
                            "edge_type": "MENTIONED_IN",
                        })

                    for i, entity_node in enumerate(entity_nodes):
                        idx = entity_node.get("_index", i)
                        source_id = entity_ids[idx] if idx < len(entity_ids) else None
                        for rel in entity_node.get("relations", []):
                            target_idx = rel.get("target_index")
                            if target_idx is not None and target_idx < len(entity_ids) and source_id:
                                span.emit_event("graph.add_edge", {
                                    "source_id": source_id,
                                    "target_id": entity_ids[target_idx],
                                    "edge_type": rel.get("relation_type", "RELATED"),
                                })
                except Exception as e:
                    logger.warning("add_dual_node entity event emission failed: %s", e, exc_info=True)

        return result

    @staticmethod
    def _to_node_dict(obj):
        """Convert object to node dictionary."""
        return SmartGraphNodes._to_node_dict(obj)

    @staticmethod
    def _from_node_dict(item_cls, node):
        """Convert node dictionary to object."""
        return SmartGraphNodes._from_node_dict(item_cls, node)

    def _validate_edge_structural(
        self, source_id: str, target_id: str, edge_type: str, properties: Optional[Dict[str, Any]]
    ) -> bool:
        """Lightweight structural validation for edges."""
        return self.edges._validate_edge_structural(source_id, target_id, edge_type, properties)

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
        with trace_span("graph.add_edge", {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
        }):
            result = self.edges.add_edge(
                source_id,
                target_id,
                edge_type,
                properties,
                valid_time,
                transaction_time,
                memory_type,
                is_global=is_global,
            )
        return result

    def add_nodes_bulk(self, nodes: List[Dict[str, Any]], batch_size: int = 500, is_global: bool = False) -> int:
        """Bulk upsert nodes using UNWIND Cypher batching.

        Args:
            nodes: List of node dicts, each with at least ``item_id`` and
                optionally ``memory_type`` plus arbitrary properties.
            batch_size: Maximum nodes per UNWIND query chunk.
            is_global: When True, skip workspace scoping — nodes are visible
                across all workspaces (for shared reference data).

        Returns:
            Total number of nodes created or updated.
        """
        with trace_span("graph.add_nodes_bulk", {"node_count": len(nodes)}):
            count = self.backend.add_nodes_bulk(nodes, batch_size=batch_size, is_global=is_global)
        self.nodes.clear_cache()
        return count

    def add_edges_bulk(
        self,
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
        batch_size: int = 500,
        is_global: bool = False,
    ) -> int:
        """Bulk upsert edges using UNWIND Cypher batching.

        Args:
            edges: List of ``(source_id, target_id, edge_type, properties)``
                tuples.
            batch_size: Maximum edges per UNWIND query chunk.
            is_global: When True, skip workspace scoping on both edge
                properties and MATCH clauses (for edges between global nodes).

        Returns:
            Total number of edges created or updated.
        """
        with trace_span("graph.add_edges_bulk", {"edge_count": len(edges)}):
            count = self.backend.add_edges_bulk(edges, batch_size=batch_size, is_global=is_global)
        self.edges.clear_cache()
        return count

    def get_scope_filters(self) -> Dict[str, Any]:
        """Extract isolation filters from the backend's scope_provider.

        Returns workspace_id (and user_id at USER isolation level) for
        injecting into raw Cypher WHERE clauses. Returns empty dict in
        OSS/unscoped mode.
        """
        try:
            if hasattr(self.backend, "scope_provider"):
                return self.backend.scope_provider.get_isolation_filters()
        except Exception:
            pass
        return {}

    def add_triple(self, triple: Any, properties: Dict[str, Any] = None, **kwargs):
        """Add a triple (Triple models) to the graph."""
        return self.edges.add_triple(triple, properties, **kwargs)

    def get_node(self, item_id: str, as_of_time: Optional[str] = None):
        """Get a node by ID."""
        return self.nodes.get_node(item_id, as_of_time)

    def get_neighbors(self, item_id: str, edge_type: Optional[str] = None, as_of_time: Optional[str] = None):
        """Get neighbors of a node."""
        return self.nodes.get_neighbors(item_id, edge_type, as_of_time)

    def get_edges_for_node(self, item_id: str):
        """Get all edges (relationships) involving a specific node."""
        return self.edges.get_edges_for_node(item_id)

    def search_nodes(self, query: Dict[str, Any]):
        """Search for nodes matching the query dict."""
        return self.search.search_nodes(query)

    def search(self, query_str: str, top_k: int = 5, **kwargs):
        """Enhanced search method using vector embeddings as primary method with text-based fallbacks."""
        return self.search.search(query_str, top_k, **kwargs)

    def _search_with_vector_embeddings(self, query_str: str, top_k: int = 5, **kwargs):
        """Primary search method using vector embeddings for semantic similarity."""
        return self.search._search_with_vector_embeddings(query_str, top_k, **kwargs)

    def _search_with_regex(self, query_str: str, top_k: int = 5, **kwargs):
        """Primary search method using FalkorDB/Cypher-compatible text search."""
        return self.search._search_with_regex(query_str, top_k, **kwargs)

    def _search_text_falkordb(self, query_str: str):
        """FalkorDB-compatible text search using CONTAINS operator with multi-word support."""
        return self.search._search_text_falkordb(query_str)

    def _search_with_simple_contains(self, query_str: str, top_k: int = 5, **kwargs):
        """Fallback search using simple contains logic."""
        return self.search._search_with_simple_contains(query_str, top_k, **kwargs)

    def _search_with_keyword_matching(self, query_str: str, top_k: int = 5, **kwargs):
        """Fallback search using keyword matching."""
        return self.search._search_with_keyword_matching(query_str, top_k, **kwargs)

    def _get_all_nodes_fallback(self, query_str: str, top_k: int = 5, **kwargs):
        """Final fallback - just return all available nodes."""
        return self.search._get_all_nodes_fallback(query_str, top_k, **kwargs)

    def get_all_node_ids(self):
        """Return all node IDs for compatibility with memory store iteration."""
        return self.nodes.nodes()

    def nodes(self):
        """Return all node IDs for compatibility with memory store iteration."""
        return self.nodes.nodes()

    def get_all_nodes(self):
        """Get all nodes in the graph (full node data)."""
        if hasattr(self.backend, "get_all_nodes"):
            return self.backend.get_all_nodes()
        else:
            # Fallback: get all node IDs and fetch each node
            node_ids = self.nodes.nodes()
            nodes = []
            for node_id in node_ids:
                node = self.get_node(node_id)
                if node:
                    nodes.append(node)
            return nodes

    def get_incoming_neighbors(self, item_id: str, edge_type: Optional[str] = None):
        """Get incoming neighbors (nodes that link TO this node)."""
        edges = self.get_edges_for_node(item_id)
        incoming_neighbors = []
        for edge in edges:
            # If this node is the target, add the source as incoming neighbor
            if edge.get("target") == item_id:
                if edge_type is None or edge.get("type") == edge_type:
                    source_node = self.get_node(edge.get("source"))
                    if source_node:
                        incoming_neighbors.append(source_node)
        return incoming_neighbors

    def remove_node(self, item_id: str):
        """Remove a node from the graph."""
        with trace_span("graph.delete_node", {"memory_id": item_id}):
            result = self.nodes.remove_node(item_id)
        return result

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None):
        """Remove an edge from the graph."""
        return self.edges.remove_edge(source_id, target_id, edge_type)

    def serialize(self) -> Any:
        return self.backend.serialize()

    def deserialize(self, data: Any):
        return self.backend.deserialize(data)

    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries when cache is full."""
        # Delegate to submodules
        self.nodes._manage_cache_size()
        self.search._manage_cache_size()

    def clear_cache(self):
        """Clear all caches."""
        self.nodes.clear_cache()
        self.search.clear_cache()

    def get_all_edges(self):
        """Get all edges in the graph."""
        if hasattr(self.backend, "get_all_edges"):
            return self.backend.get_all_edges()
        else:
            # Fallback: return empty list if backend doesn't support it
            return []

    def get_cache_stats(self):
        """Get cache performance statistics."""
        if not self.enable_caching:
            return {"caching_enabled": False}

        # Combine stats from all submodules
        node_stats = self.nodes.get_cache_stats()
        search_stats = self.search.get_cache_stats()

        return {"caching_enabled": True, "nodes": node_stats, "search": search_stats, "max_cache_size": self.cache_size}

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Execute a raw query against the backend."""
        return self.nodes.execute_query(query, params)

    def delete_by_run_id(self, run_id: str, workspace_id: Optional[str] = None) -> int:
        """Delete all nodes created by a specific pipeline run.

        Args:
            run_id: Pipeline run identifier
            workspace_id: Optional workspace scope (uses scope_provider if not set)

        Returns:
            Number of nodes deleted
        """
        params: Dict[str, Any] = {"run_id": run_id}
        where_clauses = ["n.run_id = $run_id"]

        # Add scope filtering
        if workspace_id:
            params["workspace_id"] = workspace_id
            where_clauses.append("n.workspace_id = $workspace_id")
        elif self.backend.scope_provider:
            try:
                ctx = self.backend.scope_provider.get_read_context()
                if ctx.get("workspace_id"):
                    params["workspace_id"] = ctx["workspace_id"]
                    where_clauses.append("n.workspace_id = $workspace_id")
            except Exception:
                pass  # Continue without workspace filter

        query = f"""
            MATCH (n)
            WHERE {" AND ".join(where_clauses)}
            DETACH DELETE n
            RETURN count(n) as deleted_count
        """
        result = self.execute_query(query, params)
        deleted = result[0]["deleted_count"] if result and isinstance(result[0], dict) else 0

        # Emit observability event
        self._emit_graph_stats("delete_by_run", {"run_id": run_id, "deleted": deleted}, delta_nodes=-deleted)
        return deleted

    def rename_entity_type(self, old_type: str, new_type: str, workspace_id: Optional[str] = None) -> int:
        """Rename an entity type across all matching nodes.

        Args:
            old_type: Current entity type name
            new_type: New entity type name
            workspace_id: Optional workspace scope

        Returns:
            Number of nodes updated
        """
        params: Dict[str, Any] = {"old_type": old_type, "new_type": new_type}
        where_clauses = ["n.entity_type = $old_type"]

        # Add scope filtering
        if workspace_id:
            params["workspace_id"] = workspace_id
            where_clauses.append("n.workspace_id = $workspace_id")
        elif self.backend.scope_provider:
            try:
                ctx = self.backend.scope_provider.get_read_context()
                if ctx.get("workspace_id"):
                    params["workspace_id"] = ctx["workspace_id"]
                    where_clauses.append("n.workspace_id = $workspace_id")
            except Exception:
                pass  # Continue without workspace filter

        query = f"""
            MATCH (n)
            WHERE {" AND ".join(where_clauses)}
            SET n.entity_type = $new_type
            RETURN count(n) as updated_count
        """
        result = self.execute_query(query, params)
        updated = result[0]["updated_count"] if result and isinstance(result[0], dict) else 0

        self._emit_graph_stats(
            "rename_entity_type",
            {"old_type": old_type, "new_type": new_type, "updated": updated},
        )
        return updated

    def merge_entity_types(self, source_types: List[str], target_type: str, workspace_id: Optional[str] = None) -> int:
        """Merge multiple entity types into a single target type.

        Args:
            source_types: List of entity types to rename
            target_type: The target entity type
            workspace_id: Optional workspace scope

        Returns:
            Total number of nodes updated
        """
        total = 0
        for source_type in source_types:
            if source_type != target_type:
                total += self.rename_entity_type(source_type, target_type, workspace_id)
        return total
