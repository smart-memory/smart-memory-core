"""Async FalkorDB backend — spike implementation for CORE-ASYNC-1.

Uses ``falkordb.asyncio`` to provide non-blocking graph I/O on top of
``redis.asyncio.ConnectionPool``.  The implementation mirrors
``FalkorDBBackend`` exactly but every method is an ``async def``.

Pure utility functions (``sanitize_label``, ``_is_valid_property``,
``_serialize_value``) are imported from the sync module to avoid duplication.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB as AsyncFalkorDB

from smartmemory.graph.backends.async_backend import AsyncSmartGraphBackend
from smartmemory.graph.backends.falkordb import sanitize_label, FalkorDBBackend
from smartmemory.interfaces import ScopeProvider
from smartmemory.scope_provider import DefaultScopeProvider
from smartmemory.utils import flatten_dict, unflatten_dict, get_config

logger = logging.getLogger(__name__)


def _is_valid_property(key: str, value: Any) -> bool:
    """Check if a property is valid for FalkorDB storage (shared with sync backend)."""
    return FalkorDBBackend._is_valid_property(None, key, value)  # type: ignore[arg-type]


def _serialize_value(value: Any) -> Any:
    """Serialize a value for FalkorDB storage (shared with sync backend)."""
    return FalkorDBBackend._serialize_value(None, value)  # type: ignore[arg-type]


def _chunked(items: list, size: int):
    """Yield successive chunks of *size* from *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class AsyncFalkorDBBackend(AsyncSmartGraphBackend):
    """Async FalkorDB backend implementing AsyncSmartGraphBackend.

    Connection is lazy — call ``await connect()`` (or use ``async with``)
    before issuing any queries.

    Args:
        host: FalkorDB/Redis host (default from config or ``localhost``).
        port: FalkorDB/Redis port (default from config or ``9010``).
        graph_name: Graph identifier (default ``smartmemory``).
        scope_provider: Tenant isolation provider.
        max_connections: Cap on the redis.asyncio connection pool (``None`` =
            unlimited).  Set to a small number to test backpressure behaviour.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        graph_name: str = "smartmemory",
        scope_provider: Optional[ScopeProvider] = None,
        max_connections: Optional[int] = None,
    ):
        config = get_config("graph_db")
        self.host = host or config.host
        self.port = port or config.port
        self.graph_name = graph_name
        self.scope_provider = scope_provider or DefaultScopeProvider()
        self.max_connections = max_connections

        # Set after connect()
        self._db: Optional[AsyncFalkorDB] = None
        self._graph = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the async connection and select the graph.

        Must be awaited before any query method is called.
        """
        try:
            from falkordb.asyncio import FalkorDB as _AsyncFalkorDB
        except ImportError:
            raise ImportError(
                "falkordb is required for server mode. "
                "Install it with: pip install smartmemory[server]"
            ) from None
        kwargs: Dict[str, Any] = {"host": self.host, "port": self.port}
        if self.max_connections is not None:
            kwargs["max_connections"] = self.max_connections
        self._db = _AsyncFalkorDB(**kwargs)
        self._graph = self._db.select_graph(self.graph_name)

    async def close(self) -> None:
        """Release the underlying redis.asyncio connection pool.

        Safe to call multiple times (idempotent).
        """
        if self._db is not None:
            try:
                await self._db.connection.aclose()
            except Exception:
                pass
            self._db = None
            self._graph = None

    async def __aenter__(self) -> "AsyncFalkorDBBackend":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal query helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._graph is None:
            raise RuntimeError("AsyncFalkorDBBackend: call connect() before querying")

    async def _query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Run a mutating Cypher query and return raw result_set."""
        self._ensure_connected()
        res = await self._graph.query(cypher, params or {})
        return res.result_set if hasattr(res, "result_set") else []

    async def _ro_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Run a read-only Cypher query and return raw result_set.

        FalkorDB raises ``ResponseError: Invalid graph operation on empty key``
        when querying a graph that has never been written to.  We treat that as
        an empty result set rather than propagating the error — it's equivalent
        to a MATCH returning zero rows.
        """
        self._ensure_connected()
        try:
            res = await self._graph.ro_query(cypher, params or {})
            return res.result_set if hasattr(res, "result_set") else []
        except Exception as e:
            if "Invalid graph operation on empty key" in str(e):
                return []
            raise

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    async def execute_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Execute a raw Cypher query."""
        return await self._query(cypher, params)

    async def query(self, cypher: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> List[Any]:
        """Public Cypher query interface (accepts graph_name kwarg for compat)."""
        return await self._query(cypher, params)

    # ------------------------------------------------------------------
    # AsyncSmartGraphBackend — CRUD
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._ensure_connected()
        try:
            await self._graph.delete()
        except Exception as e:
            if "Invalid graph operation on empty key" not in str(e):
                raise

    async def add_node(
        self,
        item_id: Optional[str],
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,
        created_at: Optional[Tuple] = None,
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ) -> Dict[str, Any]:
        label = sanitize_label(memory_type.capitalize()) if memory_type else "Node"
        props = flatten_dict(properties)
        write_mode = props.pop("_write_mode", None)

        if not is_global:
            write_ctx = self.scope_provider.get_write_context()
            props.update(write_ctx)

        props["is_global"] = is_global

        set_clauses = []
        params: Dict[str, Any] = {"item_id": item_id}

        for key, value in props.items():
            if not _is_valid_property(key, value):
                continue
            param_key = f"prop_{key}"
            set_clauses.append(f"n.{key} = ${param_key}")
            params[param_key] = _serialize_value(value)

        # Replace semantics: remove old properties before setting new ones
        if write_mode == "replace":
            try:
                existing_res = await self._ro_query(
                    "MATCH (n {item_id: $item_id}) RETURN n LIMIT 1", {"item_id": item_id}
                )
                if existing_res and existing_res[0]:
                    node_obj = existing_res[0][0]
                    existing_keys = (
                        list(node_obj.properties.keys())
                        if hasattr(node_obj, "properties")
                        else [k for k in vars(node_obj).keys() if not k.startswith("_")]
                    )
                    keys_to_remove = [k for k in existing_keys if k != "item_id"]
                    if keys_to_remove:
                        remove_clause = ", ".join([f"n.{k}" for k in keys_to_remove])
                        await self._query(
                            f"MATCH (n:{label} {{item_id: $item_id}}) REMOVE {remove_clause}",
                            {"item_id": item_id},
                        )
            except Exception:
                pass

        set_clauses.insert(0, "n.item_id = $item_id")
        set_clause = "SET " + ", ".join(set_clauses) if set_clauses else "SET n.item_id = $item_id"

        try:
            exists = await self.node_exists(item_id)
            if exists:
                query_str = f"MATCH (n {{item_id: $item_id}}) {set_clause} RETURN n"
            else:
                query_str = f"MERGE (n:{label} {{item_id: $item_id}}) {set_clause} RETURN n"
        except Exception:
            query_str = f"MERGE (n:{label} {{item_id: $item_id}}) {set_clause} RETURN n"

        await self._query(query_str, params)
        return {"item_id": item_id, "properties": props}

    async def add_edge(
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
        props_in = dict(properties or {})
        write_ctx = {} if is_global else self.scope_provider.get_write_context()
        props_in.update(write_ctx)

        flat_props = flatten_dict(props_in)
        serialized_props = {
            k: _serialize_value(v) for k, v in flat_props.items() if _is_valid_property(k, v)
        }

        sanitized_type = re.sub(r"[^A-Z0-9_]", "_", edge_type.upper().replace("-", "_"))
        if not sanitized_type:
            sanitized_type = "RELATED"

        params: Dict[str, Any] = {"source": source_id, "target": target_id, "props": serialized_props}
        ws_id = write_ctx.get("workspace_id")
        if ws_id:
            params["ws_id"] = ws_id
            query_str = (
                f"MATCH (a {{item_id: $source, workspace_id: $ws_id}}), "
                f"(b {{item_id: $target, workspace_id: $ws_id}}) "
                f"MERGE (a)-[r:{sanitized_type}]->(b) "
                f"SET r += $props"
            )
        else:
            query_str = (
                f"MATCH (a {{item_id: $source}}), (b {{item_id: $target}}) "
                f"MERGE (a)-[r:{sanitized_type}]->(b) "
                f"SET r += $props"
            )

        try:
            await self._query(query_str, params)
            verify_params: Dict[str, Any] = {"source": source_id, "target": target_id}
            if ws_id:
                verify_params["ws_id"] = ws_id
                verify_query = (
                    f"MATCH (a {{item_id: $source, workspace_id: $ws_id}})"
                    f"-[r:{sanitized_type}]->"
                    f"(b {{item_id: $target, workspace_id: $ws_id}}) RETURN count(r) as edge_count"
                )
            else:
                verify_query = (
                    f"MATCH (a {{item_id: $source}})-[r:{sanitized_type}]->"
                    f"(b {{item_id: $target}}) RETURN count(r) as edge_count"
                )
            verify_result = await self._ro_query(verify_query, verify_params)
            edge_count = verify_result[0][0] if verify_result and verify_result[0] else 0

            if edge_count == 0:
                source_check = await self._ro_query(
                    "MATCH (n {item_id: $id}) RETURN count(n) as count", {"id": source_id}
                )
                target_check = await self._ro_query(
                    "MATCH (n {item_id: $id}) RETURN count(n) as count", {"id": target_id}
                )
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

    async def get_node(self, item_id: str, as_of_time: Optional[str] = None) -> Optional[Dict[str, Any]]:
        res = await self._ro_query("MATCH (n {item_id: $item_id}) RETURN n LIMIT 1", {"item_id": item_id})
        if not res or not res[0]:
            return None
        node = res[0][0]
        props = dict(node.properties) if hasattr(node, "properties") else {
            k: v for k, v in vars(node).items() if not k.startswith("_") and k != "properties"
        }
        props["item_id"] = item_id
        props.pop("is_global", None)
        return props

    async def get_neighbors(
        self,
        item_id: str,
        edge_type: Optional[str] = None,
        as_of_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if edge_type:
            query_str = (
                f"MATCH (n)-[r:{edge_type.upper()}]-(m) WHERE n.item_id = $item_id "
                f"RETURN m, type(r) as link_type"
            )
        else:
            query_str = "MATCH (n)-[r]-(m) WHERE n.item_id = $item_id RETURN m, type(r) as link_type"
        res = await self._ro_query(query_str, {"item_id": item_id})
        out = []
        for record in res:
            node = record[0]
            link_type = record[1] if len(record) > 1 else None
            if hasattr(node, "keys") and hasattr(node, "values"):
                props = dict(zip(node.keys(), node.values(), strict=False))
            elif hasattr(node, "properties"):
                props = dict(node.properties)
            elif isinstance(node, dict):
                props = node
            else:
                try:
                    props = dict(node)
                except Exception:
                    continue
            out.append((unflatten_dict(props), link_type))
        return out

    async def remove_node(self, item_id: str) -> bool:
        await self._query("MATCH (n {item_id: $item_id}) DETACH DELETE n", {"item_id": item_id})
        return True

    async def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None) -> bool:
        if edge_type:
            query_str = (
                f"MATCH (a {{item_id: $source}})-[r:{edge_type.upper()}]->(b {{item_id: $target}}) DELETE r"
            )
        else:
            query_str = "MATCH (a {item_id: $source})-[r]->(b {item_id: $target}) DELETE r"
        await self._query(query_str, {"source": source_id, "target": target_id})
        return True

    async def search_nodes(self, query: Dict[str, Any], is_global: bool = False) -> List[Dict[str, Any]]:
        clauses = []
        params: Dict[str, Any] = {}

        for idx, (k, v) in enumerate(query.items()):
            param_key = f"p{idx}"
            clauses.append(f"n.{k} = ${param_key}")
            params[param_key] = v

        if is_global:
            user_key = self.scope_provider.get_user_isolation_key()
            clauses.append(f"(n.{user_key} IS NULL)")
            filters = self.scope_provider.get_global_search_filters()
            for k, v in filters.items():
                clauses.append(f"n.{k} = $ctx_{k}")
                params[f"ctx_{k}"] = v
        else:
            filters = self.scope_provider.get_isolation_filters()
            for k, v in filters.items():
                if k not in query:
                    clauses.append(f"n.{k} = $ctx_{k}")
                    params[f"ctx_{k}"] = v

        if clauses:
            where_clause = " AND ".join(clauses)
            cypher = f"MATCH (n) WHERE {where_clause} RETURN n"
        else:
            cypher = "MATCH (n) RETURN n"

        res = await self._ro_query(cypher, params)
        result = []
        for record in res:
            node = record[0]
            props = dict(node.properties) if hasattr(node, "properties") else {
                k: v for k, v in vars(node).items() if not k.startswith("_") and k != "properties"
            }
            props.pop("is_global", None)
            result.append(props)
        return result

    async def serialize(self) -> Any:
        nodes_res = await self._ro_query("MATCH (n) RETURN n")
        edges_res = await self._ro_query("MATCH (a)-[r]->(b) RETURN a.item_id, b.item_id, type(r), r")
        nodes = []
        for rec in nodes_res:
            node = rec[0]
            if hasattr(node, "properties"):
                props = dict(node.properties)
            else:
                props = dict(zip(node.keys(), node.values(), strict=False))
            nodes.append(props)
        edges = []
        for src, tgt, etype, rprops in edges_res:
            edges.append({"source": src, "target": tgt, "type": etype, "properties": rprops})
        return {"nodes": nodes, "edges": edges}

    async def deserialize(self, data: Any) -> None:
        await self.clear()
        for node in data.get("nodes", []):
            item_id = node.pop("item_id")
            await self.add_node(item_id, node)
        for edge in data.get("edges", []):
            await self.add_edge(edge["source"], edge["target"], edge["type"], edge.get("properties") or {})

    # ------------------------------------------------------------------
    # Bulk methods (UNWIND-based, chunked)
    # ------------------------------------------------------------------

    async def add_nodes_bulk(
        self, nodes: List[Dict[str, Any]], batch_size: int = 500, is_global: bool = False
    ) -> int:
        """Bulk upsert nodes using UNWIND Cypher, grouped by label."""
        if not nodes:
            return 0

        write_ctx = {} if is_global else self.scope_provider.get_write_context()

        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            item_id = n.get("item_id")
            if not item_id:
                continue
            raw_type = n.get("memory_type", "Node")
            label = sanitize_label(raw_type.capitalize() if raw_type else "Node")
            flat = flatten_dict(n)
            props: Dict[str, Any] = {}
            for key, value in flat.items():
                if _is_valid_property(key, value):
                    props[key] = _serialize_value(value)
            props.update(write_ctx)
            by_label.setdefault(label, []).append({"item_id": item_id, "props": props})

        total = 0
        for label, batch_items in by_label.items():
            query_str = (
                f"UNWIND $batch AS item "
                f"MERGE (n:{label} {{item_id: item.item_id}}) "
                f"SET n += item.props "
                f"RETURN count(n) AS cnt"
            )
            for chunk in _chunked(batch_items, batch_size):
                res = await self._graph.query(query_str, {"batch": chunk})
                if hasattr(res, "result_set") and res.result_set:
                    total += res.result_set[0][0]
        return total

    async def add_edges_bulk(
        self,
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
        batch_size: int = 500,
        is_global: bool = False,
    ) -> int:
        """Bulk upsert edges using UNWIND Cypher, grouped by edge type."""
        if not edges:
            return 0

        write_ctx = {} if is_global else self.scope_provider.get_write_context()

        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for src, tgt, etype, raw_props in edges:
            sanitized = re.sub(r"[^A-Z0-9_]", "_", etype.upper().replace("-", "_"))
            if not sanitized:
                sanitized = "RELATED"
            flat = flatten_dict(raw_props)
            props: Dict[str, Any] = {}
            for key, value in flat.items():
                if _is_valid_property(key, value):
                    props[key] = _serialize_value(value)
            props.update(write_ctx)
            by_type.setdefault(sanitized, []).append({"src": src, "tgt": tgt, "props": props})

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
            query_str = match_tpl.format(etype=etype)
            for chunk in _chunked(batch_items, batch_size):
                res = await self._graph.query(query_str, {"batch": chunk})
                created = 0
                if hasattr(res, "result_set") and res.result_set:
                    created = res.result_set[0][0]
                if created < len(chunk):
                    logger.warning(
                        "add_edges_bulk: %d/%d %s edges matched — %d dropped",
                        created,
                        len(chunk),
                        etype,
                        len(chunk) - created,
                    )
                total += created
        return total

    # ------------------------------------------------------------------
    # Helper methods (parity with sync backend)
    # ------------------------------------------------------------------

    async def node_exists(self, item_id: str) -> bool:
        try:
            res = await self._ro_query(
                "MATCH (n {item_id: $item_id}) RETURN count(n)", {"item_id": item_id}
            )
            if res and res[0]:
                return int(res[0][0]) > 0
        except Exception:
            return False
        return False

    async def get_properties(self, item_id: str) -> Dict[str, Any]:
        props = await self.get_node(item_id) or {}
        props.pop("is_global", None)
        return props

    async def set_properties(self, item_id: str, properties: Dict[str, Any]) -> bool:
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
        await self._query(q, params)
        return True

    async def get_node_count(self) -> int:
        try:
            res = await self._ro_query("MATCH (n) RETURN count(n)")
            if res and res[0]:
                return int(res[0][0])
        except Exception:
            pass
        return 0

    async def get_edge_count(self) -> int:
        try:
            res = await self._ro_query("MATCH ()-[r]->() RETURN count(r)")
            if res and res[0]:
                return int(res[0][0])
        except Exception:
            pass
        return 0

    async def get_counts(self) -> Dict[str, int]:
        return {
            "node_count": await self.get_node_count(),
            "edge_count": await self.get_edge_count(),
        }
