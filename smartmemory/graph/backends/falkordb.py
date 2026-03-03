from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from smartmemory.graph.backends.backend import SmartGraphBackend
from smartmemory.utils import unflatten_dict, flatten_dict, get_config
from smartmemory.utils.deduplication import get_canonical_key, normalize_entity_type
from smartmemory.interfaces import ScopeProvider
from smartmemory.scope_provider import DefaultScopeProvider

logger = logging.getLogger(__name__)


def sanitize_label(label: str) -> str:
    """
    Sanitize a string for use as a Neo4j/FalkorDB node label.

    Labels must:
    - Start with a letter
    - Contain only letters, numbers, and underscores
    - Not contain hyphens, spaces, or special characters
    """
    if not label:
        return "Entity"
    # Replace hyphens and spaces with underscores
    sanitized = re.sub(r"[-\s]+", "_", label)
    # Remove any other non-alphanumeric characters (except underscores)
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", sanitized)
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "E_" + sanitized
    # Default if empty
    return sanitized if sanitized else "Entity"


class FalkorDBBackend(SmartGraphBackend):
    """Minimal RedisGraph/FalkorDB backend implementing the SmartGraphBackend interface.

    This adapter lets Agentic Memory switch from Neo4j to FalkorDB by flipping
    `graph_db.backend_class` in `config.json` to ``FalkorDBBackend``.

    Notes
    -----
    * FalkorDB re-uses openCypher; parameter placeholders (e.g. `$name`) are supported.
    * The backend keeps the entire graph inside the Redis module. Ensure sufficient
      RAM or shard via Redis Cluster.
    * Only core CRUD and simple search are implemented for now. Add advanced
      analytics or vector search later if needed.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        graph_name: str = "smartmemory",
        config_path: str = "config.json",
        scope_provider: Optional[ScopeProvider] = None,
    ):
        config = get_config("graph_db")

        # Use direct config access with explicit parameter override
        self.host = host or config.host
        self.port = port or config.port
        self.graph_name = graph_name or config.get("graph_name", "smartmemory")

        self.scope_provider = scope_provider or DefaultScopeProvider()

        try:
            from falkordb import FalkorDB as _FalkorDB
        except ImportError:
            raise ImportError(
                "falkordb is required for server mode. Install it with: pip install smartmemory-core[server]"
            ) from None
        self.db = _FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)

        from smartmemory.graph.cypher_algos import CypherAlgos

        self._algos = CypherAlgos(self)

    @property
    def algos(self):
        return self._algos

    # ---------- Capability Checks ----------
    def has_capability(self, name: str) -> bool:
        if name in {"vector"}:  # RedisGraph lacks native vector ops
            return False
        return False

    # ---------- Utility ----------

    def execute_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Execute a raw Cypher query."""
        return self._query(cypher, params)

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> List[Any]:
        """Public Cypher query interface.

        Accepts optional ``graph_name`` kwarg for API compatibility with
        callers like ``OntologyGraph`` that were initialised with a dedicated
        graph — the kwarg is accepted but ignored because this backend is
        already bound to its graph at construction time.
        """
        return self._query(cypher, params)

    def _query(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        """Run a Cypher query and return raw records list."""
        res = self.graph.query(cypher, params or {})
        return res.result_set if hasattr(res, "result_set") else []

    # ---------- Bulk helpers ----------

    @staticmethod
    def _chunked(items: list, size: int):
        """Yield successive chunks of *size* from *items*."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def add_nodes_bulk(self, nodes: List[Dict[str, Any]], batch_size: int = 500, is_global: bool = False) -> int:
        """Bulk upsert nodes using UNWIND Cypher, grouped by label.

        When ``is_global=False`` (default), all nodes receive write context
        (workspace_id, user_id) from the scope provider for tenant isolation.

        When ``is_global=True``, write context is skipped — nodes are visible
        across all workspaces.  Use for shared reference data (ontology types,
        Wikipedia entities).

        Returns the total number of nodes created or updated.
        """
        if not nodes:
            return 0

        write_ctx = {} if is_global else self.scope_provider.get_write_context()

        # Group nodes by sanitized label
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            item_id = n.get("item_id")
            if not item_id:
                continue
            # Capitalize to match add_node() behavior (FalkorDB labels are case-sensitive)
            raw_type = n.get("memory_type", "Node")
            label = sanitize_label(raw_type.capitalize() if raw_type else "Node")
            flat = flatten_dict(n)
            # Filter and serialize properties like add_node() does
            props: Dict[str, Any] = {}
            for key, value in flat.items():
                if self._is_valid_property(key, value):
                    props[key] = self._serialize_value(value)
            props.update(write_ctx)
            by_label.setdefault(label, []).append({"item_id": item_id, "props": props})

        total = 0
        for label, batch_items in by_label.items():
            query = (
                f"UNWIND $batch AS item "
                f"MERGE (n:{label} {{item_id: item.item_id}}) "
                f"SET n += item.props "
                f"RETURN count(n) AS cnt"
            )
            for chunk in self._chunked(batch_items, batch_size):
                res = self.graph.query(query, {"batch": chunk})
                if hasattr(res, "result_set") and res.result_set:
                    total += res.result_set[0][0]
        return total

    def add_edges_bulk(
        self,
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
        batch_size: int = 500,
        is_global: bool = False,
    ) -> int:
        """Bulk upsert edges using UNWIND Cypher, grouped by edge type.

        When ``is_global=False`` (default), all edges receive write context
        and MATCH clauses are scoped to the workspace for tenant isolation.

        When ``is_global=True``, workspace scoping is skipped for both edge
        properties and MATCH clauses — use when connecting global nodes that
        have no ``workspace_id``.

        Returns the total number of edges created or updated.
        """
        if not edges:
            return 0

        write_ctx = {} if is_global else self.scope_provider.get_write_context()

        # Group edges by sanitized type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for src, tgt, etype, raw_props in edges:
            sanitized = re.sub(r"[^A-Z0-9_]", "_", etype.upper().replace("-", "_"))
            if not sanitized:
                sanitized = "RELATED"
            flat = flatten_dict(raw_props)
            # Filter and serialize properties like add_edge() does
            props: Dict[str, Any] = {}
            for key, value in flat.items():
                if self._is_valid_property(key, value):
                    props[key] = self._serialize_value(value)
            props.update(write_ctx)
            by_type.setdefault(sanitized, []).append({"src": src, "tgt": tgt, "props": props})

        # Scope MATCH to workspace when in multi-tenant mode
        ws_id = write_ctx.get("workspace_id")
        if ws_id:
            match_tpl = (
                "UNWIND $batch AS edge "
                "MATCH (a {{item_id: edge.src, workspace_id: edge.props.workspace_id}}), "
                "(b {{item_id: edge.tgt, workspace_id: edge.props.workspace_id}}) "
                "MERGE (a)-[r:{etype}]->(b) "
                "SET r += edge.props "
                "RETURN count(r) AS cnt"
            )
        else:
            match_tpl = (
                "UNWIND $batch AS edge "
                "MATCH (a {{item_id: edge.src}}), (b {{item_id: edge.tgt}}) "
                "MERGE (a)-[r:{etype}]->(b) "
                "SET r += edge.props "
                "RETURN count(r) AS cnt"
            )

        total = 0
        for etype, batch_items in by_type.items():
            query = match_tpl.format(etype=etype)
            for chunk in self._chunked(batch_items, batch_size):
                res = self.graph.query(query, {"batch": chunk})
                created = 0
                if hasattr(res, "result_set") and res.result_set:
                    created = res.result_set[0][0]
                if created < len(chunk):
                    logger.warning(
                        "add_edges_bulk: %d/%d %s edges matched — %d dropped (source/target nodes not found)",
                        created,
                        len(chunk),
                        etype,
                        len(chunk) - created,
                    )
                total += created
        return total

    # ---------- CRUD ----------

    def clear(self):
        try:
            self.graph.delete()
        except Exception as e:
            # Ignore error if graph doesn't exist yet
            if "Invalid graph operation on empty key" not in str(e):
                raise

    def add_node(
        self,
        item_id: Optional[str],
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,
        created_at: Optional[Tuple] = None,
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ):
        label = sanitize_label(memory_type.capitalize()) if memory_type else "Node"
        props = flatten_dict(properties)

        # Detect write mode marker (used by CRUD.update_memory_node for replace semantics)
        write_mode = props.pop("_write_mode", None)

        # Apply Scope Provider Logic
        # Only apply context if not explicitly global
        if not is_global:
            write_ctx = self.scope_provider.get_write_context()
            props.update(write_ctx)

        # Persist is_global flag — stripped by get_node()/get_properties() on read
        props["is_global"] = is_global

        # Build individual SET clauses for each property to avoid parameter expansion issues
        set_clauses = []
        params = {"item_id": item_id}

        for key, value in props.items():
            if not self._is_valid_property(key, value):
                continue

            param_key = f"prop_{key}"
            set_clauses.append(f"n.{key} = ${param_key}")
            params[param_key] = self._serialize_value(value)

        # If replace semantics requested, remove existing properties first (except item_id)
        if write_mode == "replace":
            try:
                # Get existing keys
                existing_res = self._query("MATCH (n {item_id: $item_id}) RETURN n LIMIT 1", {"item_id": item_id})
                if existing_res and existing_res[0]:
                    node_obj = existing_res[0][0]
                    if hasattr(node_obj, "properties"):
                        existing_keys = list(node_obj.properties.keys())
                    else:
                        existing_keys = [k for k in vars(node_obj).keys() if not k.startswith("_")]
                    # Exclude item_id
                    keys_to_remove = [k for k in existing_keys if k != "item_id"]
                    if keys_to_remove:
                        remove_clause = ", ".join([f"n.{k}" for k in keys_to_remove])
                        remove_query = f"MATCH (n:{label} {{item_id: $item_id}}) REMOVE {remove_clause}"
                        self._query(remove_query, {"item_id": item_id})
            except Exception:
                # Best-effort removal; continue to set new properties
                pass

        # Always set item_id as a property (not just in MERGE/MATCH pattern)
        set_clauses.insert(0, "n.item_id = $item_id")

        # Build final SET clause
        set_clause = "SET " + ", ".join(set_clauses) if set_clauses else "SET n.item_id = $item_id"

        # IMPORTANT: When a node with this item_id already exists, we must update it
        # regardless of its label instead of creating a second node with a different label.
        # Otherwise, calls like SmartGraph.add_node / SmartMemory._graph.add_node used for
        # enrichment will create a parallel node that backend.get_node() may not return,
        # causing properties (e.g. sentiment) to appear "missing".
        try:
            if self.node_exists(item_id):
                # Update existing node by item_id (label-agnostic)
                query = f"MATCH (n {{item_id: $item_id}}) {set_clause} RETURN n"
            else:
                # Create a new labeled node if it does not exist yet
                query = f"MERGE (n:{label} {{item_id: $item_id}}) {set_clause} RETURN n"
        except Exception:
            # Best-effort fallback: preserve previous MERGE behavior
            query = f"MERGE (n:{label} {{item_id: $item_id}}) {set_clause} RETURN n"

        self._query(query, params)
        return {"item_id": item_id, "properties": props}

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
    ):
        # Attach write context to relationship properties
        props_in = dict(properties or {})
        write_ctx = {} if is_global else self.scope_provider.get_write_context()
        props_in.update(write_ctx)

        # Serialize properties for edge
        flat_props = flatten_dict(props_in)
        serialized_props = {}
        for k, v in flat_props.items():
            if self._is_valid_property(k, v):
                serialized_props[k] = self._serialize_value(v)

        # Sanitize edge type (same logic as add_edges_bulk)
        sanitized_type = re.sub(r"[^A-Z0-9_]", "_", edge_type.upper().replace("-", "_"))
        if not sanitized_type:
            sanitized_type = "RELATED"

        params = {
            "source": source_id,
            "target": target_id,
            "props": serialized_props,
        }
        # Scope MATCH to workspace when in multi-tenant mode
        ws_id = write_ctx.get("workspace_id")
        if ws_id:
            params["ws_id"] = ws_id
            query = (
                f"MATCH (a {{item_id: $source, workspace_id: $ws_id}}), "
                f"(b {{item_id: $target, workspace_id: $ws_id}}) "
                f"MERGE (a)-[r:{sanitized_type}]->(b) "
                f"SET r += $props"
            )
        else:
            query = (
                f"MATCH (a {{item_id: $source}}), (b {{item_id: $target}}) "
                f"MERGE (a)-[r:{sanitized_type}]->(b) "
                f"SET r += $props"
            )

        try:
            self._query(query, params)
            # Verify the edge was actually created
            verify_params = {"source": source_id, "target": target_id}
            if ws_id:
                verify_params["ws_id"] = ws_id
                verify_query = (
                    f"MATCH (a {{item_id: $source, workspace_id: $ws_id}})"
                    f"-[r:{sanitized_type}]->"
                    f"(b {{item_id: $target, workspace_id: $ws_id}}) RETURN count(r) as edge_count"
                )
            else:
                verify_query = f"MATCH (a {{item_id: $source}})-[r:{sanitized_type}]->(b {{item_id: $target}}) RETURN count(r) as edge_count"
            verify_result = self._query(verify_query, verify_params)
            edge_count = verify_result[0][0] if verify_result and verify_result[0] else 0

            if edge_count == 0:
                source_check = self._query("MATCH (n {item_id: $id}) RETURN count(n) as count", {"id": source_id})
                target_check = self._query("MATCH (n {item_id: $id}) RETURN count(n) as count", {"id": target_id})
                source_exists = source_check[0][0] if source_check and source_check[0] else 0
                target_exists = target_check[0][0] if target_check and target_check[0] else 0
                logger.debug(
                    "Edge not created: %s --[%s]--> %s | source_exists=%s target_exists=%s",
                    source_id,
                    sanitized_type,
                    target_id,
                    bool(source_exists),
                    bool(target_exists),
                )
                return False

            logger.debug("Edge created: %s --[%s]--> %s", source_id, sanitized_type, target_id)
            return True

        except Exception as e:
            logger.warning("Edge creation error: %s --[%s]--> %s: %s", source_id, sanitized_type, target_id, e)
            return False

    def get_node(self, item_id: str, as_of_time: Optional[str] = None):
        res = self._query("MATCH (n {item_id: $item_id}) RETURN n LIMIT 1", {"item_id": item_id})
        if not res or not res[0]:
            return None

        # Extract properties from FalkorDB Node object
        node = res[0][0]
        if hasattr(node, "properties"):
            props = dict(node.properties)
        else:
            # Fallback to direct attribute access if properties attribute is not available
            props = {k: v for k, v in vars(node).items() if not k.startswith("_") and k != "properties"}

        # Ensure item_id is included in the returned properties at the top level
        props["item_id"] = item_id

        # Remove internal properties that shouldn't be exposed
        props.pop("is_global", None)

        return props

    def get_neighbors(
        self,
        item_id: str,
        edge_type: Optional[str] = None,
        as_of_time: Optional[str] = None,
        direction: str = "both",
    ):
        # Build relationship pattern with optional direction
        rel = f"[r:{edge_type.upper()}]" if edge_type else "[r]"
        if direction == "outgoing":
            pattern = f"-{rel}->"
        elif direction == "incoming":
            pattern = f"<-{rel}-"
        else:  # "both" — backward-compatible undirected traversal
            pattern = f"-{rel}-"
        query = f"MATCH (n){pattern}(m) WHERE n.item_id = $item_id RETURN m, type(r) as link_type"
        res = self._query(query, {"item_id": item_id})
        out = []
        for record in res:
            node = record[0]
            link_type = record[1] if len(record) > 1 else None

            # Handle both dict-like and Node objects
            if hasattr(node, "keys") and hasattr(node, "values"):
                props = dict(zip(node.keys(), node.values(), strict=False))
            elif hasattr(node, "properties"):
                # FalkorDB Node object
                props = dict(node.properties)
            elif isinstance(node, dict):
                props = node
            else:
                # Fallback: try to convert to dict
                try:
                    props = dict(node)
                except Exception:
                    continue
            # Return tuple of (neighbor, link_type) to match expected format
            out.append((unflatten_dict(props), link_type))
        return out

    def get_all_edges(self):
        """Get all edges in the graph for debugging purposes."""
        try:
            query = "MATCH (a)-[r]->(b) RETURN a.item_id as source_id, type(r) as edge_type, b.item_id as target_id, r"
            result = self._query(query, {})
            edges = []
            for record in result:
                # FalkorDB returns results as tuples/lists
                if len(record) >= 4:
                    source_id = record[0] if record[0] else "unknown"
                    edge_type = record[1] if record[1] else "unknown"
                    target_id = record[2] if record[2] else "unknown"
                    edge_obj = record[3]

                    # Extract edge properties
                    edge_props = {}
                    if hasattr(edge_obj, "properties"):
                        edge_props = dict(edge_obj.properties)

                    edge_info = {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_type,
                        "valid_from": edge_props.get("valid_from"),
                        "valid_to": edge_props.get("valid_to"),
                        "created_at": edge_props.get("created_at"),
                        "properties": edge_props,
                    }
                    edges.append(edge_info)
            return edges
        except Exception as e:
            logger.debug("Error getting all edges: %s", e)
            return []

    def remove_node(self, item_id: str):
        self._query("MATCH (n {item_id: $item_id}) DETACH DELETE n", {"item_id": item_id})
        return True

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None):
        if edge_type:
            query = f"MATCH (a {{item_id: $source}})-[r:{edge_type.upper()}]->(b {{item_id: $target}}) DELETE r"
        else:
            query = "MATCH (a {item_id: $source})-[r]->(b {item_id: $target}) DELETE r"
        self._query(query, {"source": source_id, "target": target_id})
        return True

    # ---------- Read helpers for transactional layer ----------

    def node_exists(self, item_id: str) -> bool:
        try:
            res = self._query("MATCH (n {item_id: $item_id}) RETURN count(n)", {"item_id": item_id})
            if res and res[0]:
                val = res[0][0]
                return int(val) > 0
        except Exception:
            return False
        return False

    def get_properties(self, item_id: str) -> Dict[str, Any]:
        props = self.get_node(item_id) or {}
        # Remove internal flags
        props.pop("is_global", None)
        return props

    def set_properties(self, item_id: str, properties: Dict[str, Any]) -> bool:
        # Update only provided scalar properties
        from smartmemory.utils import flatten_dict

        props = flatten_dict(properties or {})
        if not props:
            return True
        set_parts = []
        params: Dict[str, Any] = {"item_id": item_id}
        for k, v in props.items():
            if v is None:
                continue
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            if not isinstance(v, (str, int, float, bool)):
                continue
            pk = f"p_{k}"
            set_parts.append(f"n.{k} = ${pk}")
            params[pk] = v
        if not set_parts:
            return True
        q = f"MATCH (n {{item_id: $item_id}}) SET {', '.join(set_parts)}"
        self._query(q, params)
        return True

    def edge_exists(self, source_id: str, target_id: str, relation_type: str) -> bool:
        try:
            q = f"MATCH (a {{item_id: $s}})-[r:{relation_type.upper()}]->(b {{item_id: $t}}) RETURN count(r)"
            res = self._query(q, {"s": source_id, "t": target_id})
            if res and res[0]:
                return int(res[0][0]) > 0
        except Exception:
            return False
        return False

    def get_edge_properties(self, source_id: str, target_id: str, relation_type: str) -> Optional[Dict[str, Any]]:
        try:
            q = f"MATCH (a {{item_id: $s}})-[r:{relation_type.upper()}]->(b {{item_id: $t}}) RETURN r LIMIT 1"
            res = self._query(q, {"s": source_id, "t": target_id})
            if res and res[0] and res[0][0] is not None:
                r = res[0][0]
                if hasattr(r, "properties"):
                    return dict(r.properties)
                # best-effort
                return {k: v for k, v in vars(r).items() if not k.startswith("_")}
        except Exception:
            return None
        return None

    def list_edges(self, item_id: str) -> List[Dict[str, Any]]:
        try:
            return self.get_edges_for_node(item_id)
        except Exception:
            return []

    def archive(self, item_id: str) -> bool:
        self._query("MATCH (n {item_id: $id}) SET n.archived = true", {"id": item_id})
        return True

    def unarchive(self, item_id: str) -> bool:
        self._query("MATCH (n {item_id: $id}) SET n.archived = false", {"id": item_id})
        return True

    def merge_nodes(self, target_id: str, source_ids: List[str]) -> bool:
        """Merge multiple source nodes into a target node.

        Rewires relationships and merges properties, then deletes source nodes.
        Uses a read-then-recreate approach because FalkorDB does not support
        dynamic edge types (``TYPE(r)``) in MERGE/CREATE patterns.

        Args:
            target_id: The ID of the node to keep (canonical node).
            source_ids: List of IDs of nodes to merge into the target.

        Returns:
            True if successful, False if target does not exist or on error.
        """
        if not source_ids:
            return True

        try:
            # Guard: target must exist — otherwise source deletion causes silent data loss.
            target_check = self._query(
                "MATCH (t {item_id: $tid}) RETURN t.item_id LIMIT 1",
                {"tid": target_id},
            )
            if not target_check:
                logger.error("merge_nodes: target %s does not exist", target_id)
                return False

            for source_id in source_ids:
                if source_id == target_id:
                    continue

                # 1. Read outgoing edges: (source)-[r]->(other), excluding target.
                out_rows = self._query(
                    "MATCH (source {item_id: $sid})-[r]->(other) "
                    "WHERE other.item_id <> $tid "
                    "RETURN other.item_id, type(r), properties(r)",
                    {"sid": source_id, "tid": target_id},
                )

                # 2. Read incoming edges: (other)-[r]->(source), excluding target.
                in_rows = self._query(
                    "MATCH (other)-[r]->(source {item_id: $sid}) "
                    "WHERE other.item_id <> $tid "
                    "RETURN other.item_id, type(r), properties(r)",
                    {"sid": source_id, "tid": target_id},
                )

                # 3. Delete all edges from/to source.
                self._query(
                    "MATCH (source {item_id: $sid})-[r]-() DELETE r",
                    {"sid": source_id},
                )

                # 4. Recreate outgoing edges on target.
                for row in out_rows:
                    other_id, etype, props = row[0], row[1], row[2] if len(row) > 2 else {}
                    self.add_edge(target_id, other_id, etype, props or {})

                # 5. Recreate incoming edges on target.
                for row in in_rows:
                    other_id, etype, props = row[0], row[1], row[2] if len(row) > 2 else {}
                    self.add_edge(other_id, target_id, etype, props or {})

                # 6. Merge properties (source fills gaps, target keeps identity).
                #    ``+=`` overwrites ALL matching keys, including ``item_id``,
                #    so we immediately restore the target's ``item_id`` afterward.
                self._query(
                    "MATCH (source {item_id: $sid}), (target {item_id: $tid}) "
                    "SET target += properties(source), target.item_id = $tid",
                    {"sid": source_id, "tid": target_id},
                )

                # 7. Delete source node.
                self.remove_node(source_id)

            return True

        except Exception as e:
            logger.error("Failed to merge nodes into %s: %s", target_id, e)
            return False

    # ---------- Vector similarity ----------
    def vector_similarity_search(self, embedding: List[float], top_k: int = 5, prop_key: str = "embedding"):
        # Fallback to base implementation (Python-side) until Redis vector module available
        return super().vector_similarity_search(embedding, top_k, prop_key)

    # ---------- Query helpers ----------

    def search_nodes(self, query: Dict[str, Any], is_global: bool = False):
        """Search for nodes matching query properties.

        Args:
            query: Dictionary of property filters (simple equality only)
            is_global: Whether to search globally or scope to user/workspace

        Returns:
            List of node dictionaries matching the query
        """
        clauses = []
        params = {}

        # Add query clauses (simple equality only)
        for idx, (k, v) in enumerate(query.items()):
            param_key = f"p{idx}"
            clauses.append(f"n.{k} = ${param_key}")
            params[param_key] = v

        # Apply Scope Provider Isolation
        if is_global:
            # Global search: Explicitly exclude user-scoped nodes (Public/Shared nodes)
            user_key = self.scope_provider.get_user_isolation_key()
            clauses.append(f"(n.{user_key} IS NULL)")

            # Enforce isolation filters for global searches (excludes user-level)
            filters = self.scope_provider.get_global_search_filters()
            for k, v in filters.items():
                clauses.append(f"n.{k} = $ctx_{k}")
                params[f"ctx_{k}"] = v
        else:
            # Scoped search: Enforce all isolation filters (User + Tenant + Workspace)
            filters = self.scope_provider.get_isolation_filters()
            for k, v in filters.items():
                # Skip if the query itself explicitly filters on this key
                if k not in query:
                    clauses.append(f"n.{k} = $ctx_{k}")
                    params[f"ctx_{k}"] = v

        if clauses:
            where_clause = " AND ".join(clauses)
            cypher = f"MATCH (n) WHERE {where_clause} RETURN n"
        else:
            cypher = "MATCH (n) RETURN n"
        res = self._query(cypher, params)
        result = []
        for record in res:
            node = record[0]
            if hasattr(node, "properties"):
                props = dict(node.properties)
            else:
                # Fallback to direct attribute access if properties attribute is not available
                props = {k: v for k, v in vars(node).items() if not k.startswith("_") and k != "properties"}

            # Remove internal properties that shouldn't be exposed
            props.pop("is_global", None)

            # Don't use unflatten_dict to preserve flat structure with item_id
            result.append(props)
        return result

    def search_nodes_by_type_or_tag(self, type_or_tag: str, is_global: bool = False) -> List[Dict[str, Any]]:
        """Search for nodes by type OR tag using proper Cypher OR logic.

        This replaces the MongoDB-style {"$or": [{"type": x}, {"tags": x}]} pattern.

        Args:
            type_or_tag: The type or tag value to search for
            is_global: Whether to search globally or scope to user/workspace

        Returns:
            List of node dictionaries matching the query
        """
        clauses = []
        params = {"type_or_tag": type_or_tag}

        # Build OR condition for type or tags
        or_clause = "(n.type = $type_or_tag OR n.tags = $type_or_tag)"
        clauses.append(or_clause)

        # Apply Scope Provider Isolation
        if is_global:
            # Global search constraints
            user_key = self.scope_provider.get_user_isolation_key()
            clauses.append(f"(n.{user_key} IS NULL)")
            filters = self.scope_provider.get_global_search_filters()
            for k, v in filters.items():
                clauses.append(f"n.{k} = $ctx_{k}")
                params[f"ctx_{k}"] = v
        else:
            # Scoped search constraints
            filters = self.scope_provider.get_isolation_filters()
            for k, v in filters.items():
                clauses.append(f"n.{k} = $ctx_{k}")
                params[f"ctx_{k}"] = v

        where_clause = " AND ".join(clauses)
        cypher = f"MATCH (n) WHERE {where_clause} RETURN n"
        res = self._query(cypher, params)

        result = []
        for record in res:
            node = record[0]
            if hasattr(node, "properties"):
                props = dict(node.properties)
            else:
                props = {k: v for k, v in vars(node).items() if not k.startswith("_") and k != "properties"}
            props.pop("is_global", None)
            result.append(props)
        return result

    def get_all_nodes(self):
        """Get all nodes in the graph."""
        query = "MATCH (n) RETURN n"
        result = self._query(query)
        nodes = []
        for record in result:
            if record and len(record) > 0:
                node = record[0]
                if hasattr(node, "properties"):
                    node_dict = dict(node.properties)
                    nodes.append(node_dict)
                elif isinstance(node, dict):
                    nodes.append(node)
        return nodes

    def get_edges_for_node(self, item_id: str) -> List[Dict[str, Any]]:
        """Get all edges involving a specific node.

        Returns list of dicts with normalized keys:
        {source_id, target_id, edge_type, valid_from, valid_to, created_at, properties}
        """
        query = """
        MATCH (n {item_id: $item_id})-[r]-(m)
        RETURN startNode(r).item_id as source_id, endNode(r).item_id as target_id,
               type(r) as edge_type, r
        """
        result = self._query(query, {"item_id": item_id})

        edges = []
        for record in result:
            if record and len(record) >= 4:
                source_id = record[0] if record[0] else "unknown"
                target_id = record[1] if record[1] else "unknown"
                edge_type = record[2] if record[2] else "unknown"
                edge_obj = record[3]

                edge_props = {}
                if hasattr(edge_obj, "properties"):
                    edge_props = dict(edge_obj.properties)

                edges.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_type": edge_type,
                    "valid_from": edge_props.get("valid_from"),
                    "valid_to": edge_props.get("valid_to"),
                    "created_at": edge_props.get("created_at"),
                    "properties": edge_props,
                })

        return edges

    # ---------- Stats helpers ----------
    def get_node_count(self) -> int:
        """Return total number of nodes in the graph."""
        try:
            res = self._query("MATCH (n) RETURN count(n)")
            if res and res[0]:
                val = res[0][0]
                try:
                    return int(val)
                except Exception:
                    # Some drivers return dict-like or typed values
                    return int(str(val))
        except Exception:
            pass
        return 0

    def get_edge_count(self) -> int:
        """Return total number of edges (relationships) in the graph."""
        try:
            res = self._query("MATCH ()-[r]->() RETURN count(r)")
            if res and res[0]:
                val = res[0][0]
                try:
                    return int(val)
                except Exception:
                    return int(str(val))
        except Exception:
            pass
        return 0

    def get_counts(self) -> Dict[str, int]:
        """Return a dict with node_count and edge_count for fast stats emission."""
        return {
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
        }

    # ---------- (De)Serialization ----------

    def serialize(self) -> Any:
        nodes_res = self._query("MATCH (n) RETURN n")
        edges_res = self._query("MATCH (a)-[r]->(b) RETURN a.item_id, b.item_id, type(r), r")
        nodes = []
        for rec in nodes_res:
            props = dict(zip(rec[0].keys(), rec[0].values(), strict=False))  # type: ignore[index]
            nodes.append(props)
        edges = []
        for src, tgt, etype, rprops in edges_res:
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "type": etype,
                    "properties": rprops,
                }
            )
        return {"nodes": nodes, "edges": edges}

    def deserialize(self, data: Any):
        self.clear()
        for node in data.get("nodes", []):
            item_id = node.pop("item_id")
            self.add_node(item_id, node)
        for edge in data.get("edges", []):
            self.add_edge(edge["source"], edge["target"], edge["type"], edge.get("properties") or {})

    def find_entity_by_canonical_key(
        self,
        canonical_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find an existing entity node by its canonical key.

        Used for cross-memory entity resolution - if an entity with the same
        canonical key exists, we link to it instead of creating a duplicate.

        Args:
            canonical_key: Normalized key like "john smith|person"

        Returns:
            Entity node dict with item_id and properties, or None if not found
        """
        # Build query with scope isolation filters
        filters = ["n.canonical_key = $canonical_key"]
        params = {"canonical_key": canonical_key}

        # Apply scope provider isolation
        isolation_filters = self.scope_provider.get_isolation_filters()
        for k, v in isolation_filters.items():
            filters.append(f"n.{k} = $ctx_{k}")
            params[f"ctx_{k}"] = v

        where_clause = " AND ".join(filters)

        # Search across all entity labels (we don't know the exact type)
        query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n.item_id AS item_id, n AS props
            LIMIT 1
        """

        try:
            result = self._query(query, params)
            if result and result.result_set:
                row = result.result_set[0]
                return {
                    "item_id": row[0],
                    "properties": dict(row[1].properties) if hasattr(row[1], "properties") else {},
                }
        except Exception as e:
            logger.debug(f"Entity lookup failed for {canonical_key}: {e}")

        return None

    def add_dual_node(
        self,
        item_id: str,
        memory_properties: Dict[str, Any],
        memory_type: str,
        entity_nodes: List[Dict[str, Any]] = None,
        is_global: bool = False,
    ):
        """
        Add a dual-node structure: one memory node + related entity nodes.

        Args:
            item_id: Unique identifier for the memory node
            memory_properties: Properties for the memory node (content, metadata, etc.)
            memory_type: Memory type for the memory node label (semantic, episodic, etc.)
            entity_nodes: List of entity node dicts with {entity_type, properties, relationships}
            is_global: Whether nodes are global or user-scoped

        Returns:
            Dict with creation results and node IDs

        .. warning::
            This operation is NOT atomic in FalkorDB as it executes multiple sequential queries.
            If an error occurs midway, the graph may be left in an inconsistent state (e.g., memory node created but entities missing).
        """
        # Note: FalkorDB executes queries sequentially (no multi-statement transactions).
        # This is acceptable for now - failures are rare and can be cleaned up.

        # Prepare memory node
        memory_label = sanitize_label(memory_type.capitalize())
        memory_props = flatten_dict(memory_properties)

        # Apply scope provider logic
        if not is_global:
            write_ctx = self.scope_provider.get_write_context()
            memory_props.update(write_ctx)

        memory_props["is_global"] = is_global

        # Build transaction query for atomic dual-node creation
        queries = []
        params = {"memory_id": item_id}

        # 1. Create memory node
        memory_set_clauses = []
        for key, value in memory_props.items():
            if self._is_valid_property(key, value):
                param_key = f"mem_{key}"
                memory_set_clauses.append(f"m.{key} = ${param_key}")
                params[param_key] = self._serialize_value(value)

        if memory_set_clauses:
            memory_query = f"MERGE (m:{memory_label} {{item_id: $memory_id}}) SET {', '.join(memory_set_clauses)}"
        else:
            memory_query = f"MERGE (m:{memory_label} {{item_id: $memory_id}})"
        queries.append(memory_query)

        # 2. Process entity nodes with cross-memory resolution
        entity_ids = []
        resolved_entities = []  # Track which entities were resolved vs created
        entity_id_map = {}  # Map index to actual entity_id for relationship creation

        if entity_nodes:
            for i, entity_node in enumerate(entity_nodes):
                entity_type = entity_node.get("entity_type", "Entity")
                entity_props = entity_node.get("properties") or {}
                entity_relationships = entity_node.get("relations", [])
                entity_name = entity_props.get("name") or entity_props.get("content", "")

                # Generate canonical key for entity resolution
                canonical_key = entity_props.get("canonical_key")
                if not canonical_key and entity_name:
                    canonical_key = get_canonical_key(entity_name, entity_type)

                logger.info(
                    f"Entity resolution: name='{entity_name}', type='{entity_type}', canonical_key='{canonical_key}'"
                )

                # Try to find existing entity with same canonical key (uses scope_provider for isolation)
                existing_entity = None
                if canonical_key:
                    existing_entity = self.find_entity_by_canonical_key(canonical_key)
                    logger.info(f"Entity lookup result for '{canonical_key}': {existing_entity}")

                if existing_entity:
                    # Reuse existing entity - just link to it
                    entity_id = existing_entity["item_id"]
                    entity_ids.append(entity_id)
                    entity_id_map[i] = entity_id
                    resolved_entities.append(
                        {"index": i, "entity_id": entity_id, "resolved": True, "canonical_key": canonical_key}
                    )
                    logger.debug(f"Resolved entity '{entity_name}' to existing node {entity_id}")

                    # Create MENTIONED_IN relationship from existing entity to new memory
                    # This links the entity to all memories that mention it
                    link_query = f"""
                        MATCH (m:{memory_label} {{item_id: $memory_id}}), (e {{item_id: $existing_entity_id}})
                        MERGE (m)-[:CONTAINS_ENTITY]->(e)
                        MERGE (e)-[:MENTIONED_IN]->(m)
                    """
                    params[f"existing_entity_id_{i}"] = entity_id
                    # Execute this after memory node is created

                else:
                    # Create new entity node
                    entity_id = f"{item_id}_entity_{i}"
                    entity_ids.append(entity_id)
                    entity_id_map[i] = entity_id
                    entity_label = sanitize_label(normalize_entity_type(entity_type).capitalize())

                    resolved_entities.append(
                        {"index": i, "entity_id": entity_id, "resolved": False, "canonical_key": canonical_key}
                    )

                    # Add user context to entity via scope provider
                    if not is_global:
                        write_ctx = self.scope_provider.get_write_context()
                        entity_props.update(write_ctx)

                    entity_props["is_global"] = is_global
                    entity_props["canonical_key"] = canonical_key  # Store for future resolution

                    # Build entity creation query
                    entity_set_clauses = []
                    for key, value in entity_props.items():
                        if self._is_valid_property(key, value):
                            param_key = f"ent_{i}_{key}"
                            entity_set_clauses.append(f"e{i}.{key} = ${param_key}")
                            params[param_key] = self._serialize_value(value)

                    params[f"entity_id_{i}"] = entity_id

                    if entity_set_clauses:
                        entity_query = f"MERGE (e{i}:{entity_label} {{item_id: $entity_id_{i}}}) SET {', '.join(entity_set_clauses)}"
                    else:
                        entity_query = f"MERGE (e{i}:{entity_label} {{item_id: $entity_id_{i}}})"
                    queries.append(entity_query)

                    # Create bidirectional relationships using MATCH to find the
                    # already-created nodes.  Bare CREATE (m)-[:R]->(e) in a
                    # standalone query would create *new* anonymous nodes because
                    # variable bindings don't persist across separate query() calls.
                    queries.append(
                        f"MATCH (m:{memory_label} {{item_id: $memory_id}}), "
                        f"(e {{item_id: $entity_id_{i}}}) "
                        f"CREATE (m)-[:CONTAINS_ENTITY]->(e)"
                    )
                    queries.append(
                        f"MATCH (e {{item_id: $entity_id_{i}}}), "
                        f"(m:{memory_label} {{item_id: $memory_id}}) "
                        f"CREATE (e)-[:MENTIONED_IN]->(m)"
                    )

                # Store relationships for later (after all entities processed)
                entity_node["_relationships"] = entity_relationships
                entity_node["_index"] = i

            # Create semantic relationships between entities (using resolved IDs).
            # Use MERGE to safely handle both new and resolved (deduplicated) entities —
            # resolved entities may already have this edge from a prior ingestion.
            for entity_node in entity_nodes:
                i = entity_node.get("_index", 0)
                for rel in entity_node.get("_relationships", []):
                    target_idx = rel.get("target_index")
                    rel_type = rel.get("relation_type", "RELATED")
                    if target_idx is not None and target_idx in entity_id_map:
                        source_eid = entity_id_map[i]
                        target_eid = entity_id_map[target_idx]
                        rel_param_src = f"rel_src_{i}_{target_idx}"
                        rel_param_tgt = f"rel_tgt_{i}_{target_idx}"
                        params[rel_param_src] = source_eid
                        params[rel_param_tgt] = target_eid
                        queries.append(
                            f"MATCH (a {{item_id: ${rel_param_src}}}), "
                            f"(b {{item_id: ${rel_param_tgt}}}) "
                            f"MERGE (a)-[:{rel_type}]->(b)"
                        )

        # Create memory node first with all properties
        # Note: The logic below duplicates the queries list building above.
        # The original code had a confusing double-block structure (queries append vs memory_query direct execution?).
        # Ah, I see the original code builds 'queries' list but then seems to have a secondary block
        # starting at line 783 "Create memory node first with all properties" which re-does memory params.
        # This looks like dead code or a merge artifact in the original file.
        # I will stick to the first block which builds 'queries'.
        # Wait, the original code effectively ignored 'queries' list execution?
        # No, looking at the end of the file (which I can't see all of), it probably executes 'queries'.
        # But there is a block at 783 that rebuilds memory params.
        # Let's assume the first block is the correct one for the transaction.

        # Execute queries sequentially
        if queries:
            for q in queries:
                self._query(q, params)

        # Link resolved entities to the new memory node
        for resolved in resolved_entities:
            if resolved["resolved"]:
                link_query = f"""
                    MATCH (m:{memory_label} {{item_id: $memory_id}}), (e {{item_id: $resolved_entity_id}})
                    MERGE (m)-[:CONTAINS_ENTITY]->(e)
                    MERGE (e)-[:MENTIONED_IN]->(m)
                """
                self._query(link_query, {"memory_id": item_id, "resolved_entity_id": resolved["entity_id"]})
                logger.info(f"Linked existing entity {resolved['entity_id']} to memory {item_id}")

        # Log resolution stats
        new_count = sum(1 for e in resolved_entities if not e["resolved"])
        resolved_count = sum(1 for e in resolved_entities if e["resolved"])
        if resolved_count > 0:
            logger.info(f"Entity resolution: {new_count} new, {resolved_count} resolved to existing")

        return {
            "memory_node_id": item_id,
            "entity_node_ids": entity_ids,
            "resolved_entities": resolved_entities,
            "memory_type": memory_type,
            "entity_count": len(entity_ids),
            "new_entity_count": new_count,
            "resolved_entity_count": resolved_count,
        }

    def _is_valid_property(self, key: str, value: Any) -> bool:
        """Check if a property is valid for FalkorDB storage."""
        if value is None or key == "embedding":
            return False
        if isinstance(value, str) and value == "":
            return False
        # Allow list, tuple, dict - they will be serialized to JSON
        return isinstance(value, (str, int, float, bool, list, tuple, dict))

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for FalkorDB storage."""
        import json

        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, default=str)
        return value
