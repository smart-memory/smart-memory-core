"""SQLiteBackend — zero-infra graph backend using stdlib sqlite3.

Implements SmartGraphBackend using an adjacency-list model:
- nodes table: item_id, properties (JSON), memory_type, valid_from, valid_to, created_at
- edges table: source_id, target_id, edge_type, properties (JSON), with ON DELETE CASCADE

Intended for SmartMemory Lite (DIST-LITE-1). Does not support Cypher queries,
temporal time-travel, or graph analytics. Use FalkorDB for those features.
"""
import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smartmemory.graph.backends.backend import SmartGraphBackend
from smartmemory.interfaces import ScopeProvider

logger = logging.getLogger(__name__)

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    item_id      TEXT PRIMARY KEY,
    memory_type  TEXT,
    valid_from   TEXT,
    valid_to     TEXT,
    created_at   TEXT,
    properties   TEXT NOT NULL DEFAULT '{}'
)
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL REFERENCES nodes(item_id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES nodes(item_id) ON DELETE CASCADE,
    edge_type   TEXT NOT NULL,
    memory_type TEXT,
    valid_from  TEXT,
    valid_to    TEXT,
    created_at  TEXT,
    properties  TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, edge_type)
)
"""

# Migration: columns added after initial release. ALTER TABLE ADD COLUMN is idempotent via try/except.
_EDGE_MIGRATIONS = [
    "ALTER TABLE edges ADD COLUMN valid_from TEXT",
    "ALTER TABLE edges ADD COLUMN valid_to   TEXT",
    "ALTER TABLE edges ADD COLUMN created_at TEXT",
]

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_nodes_memory_type ON nodes(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
]


class SQLiteBackend(SmartGraphBackend):
    """Graph backend backed by SQLite. Adjacency-list model, WAL mode, no network required."""

    def __init__(self, db_path: str = ":memory:", scope_provider: Optional[ScopeProvider] = None):
        self._is_memory_db = db_path == ":memory:"
        if not self._is_memory_db:
            path = Path(db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(path)
        # SQLiteBackend is single-tenant by design (SmartMemory Lite).
        # If a non-None scope_provider is supplied the caller expects isolation this backend
        # cannot provide. Raise immediately so the gap is impossible to miss or suppress.
        if scope_provider is not None:
            raise ValueError(
                "SQLiteBackend does not enforce tenant/workspace isolation and cannot accept a "
                "scope_provider. All data is stored in a single-tenant SQLite database. "
                "Use FalkorDB-backed SmartMemory for multi-tenant deployments."
            )
        self._scope_provider = scope_provider
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        result = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        self._journal_mode = result[0] if result else "unknown"
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(_CREATE_NODES)
                self._conn.execute(_CREATE_EDGES)
                for idx in _CREATE_INDEXES:
                    self._conn.execute(idx)
                # Apply edge schema migrations for databases created before temporal columns existed.
                # SQLite does not support "ADD COLUMN IF NOT EXISTS"; try/except is the correct pattern.
                for migration in _EDGE_MIGRATIONS:
                    try:
                        self._conn.execute(migration)
                    except sqlite3.OperationalError:
                        pass  # Column already exists — safe to ignore

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_time(t: Optional[Tuple]) -> Optional[str]:  # type: ignore[type-arg]
        if t is None:
            return None
        return str(t[0]) if isinstance(t, (tuple, list)) else str(t)

    @staticmethod
    def _unpack_valid_to(t: Optional[Tuple]) -> Optional[str]:  # type: ignore[type-arg]
        if t is None:
            return None
        if isinstance(t, (tuple, list)) and len(t) > 1:
            return str(t[1])
        return None

    def _row_to_node(self, row: Any) -> Dict[str, Any]:
        item_id, memory_type, valid_from, valid_to, created_at, properties_json = row
        props: Dict[str, Any] = json.loads(properties_json)
        props["item_id"] = item_id
        props["memory_type"] = memory_type
        props["valid_from"] = valid_from
        props["valid_to"] = valid_to
        props["created_at"] = created_at
        return props

    def close(self) -> None:
        """Close the SQLite connection, flushing WAL to the main database file."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    # ── SmartGraphBackend abstract methods ───────────────────────────────────

    def add_node(
        self,
        item_id: Optional[str],
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,  # type: ignore[type-arg]
        created_at: Optional[Tuple] = None,  # type: ignore[type-arg]
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ) -> Dict[str, Any]:
        if item_id is None:
            item_id = str(uuid.uuid4())
        mem_type = memory_type or properties.get("memory_type")
        props_to_store = {k: v for k, v in properties.items() if k not in ("item_id", "memory_type")}
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO nodes "
                    "(item_id, memory_type, valid_from, valid_to, created_at, properties) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(item_id) DO UPDATE SET "
                    "    memory_type = excluded.memory_type, "
                    "    valid_from   = excluded.valid_from, "
                    "    valid_to     = excluded.valid_to, "
                    "    created_at   = excluded.created_at, "
                    "    properties   = excluded.properties",
                    (
                        item_id,
                        mem_type,
                        self._unpack_time(valid_time),
                        self._unpack_valid_to(valid_time),
                        self._unpack_time(created_at),
                        json.dumps(props_to_store),
                    ),
                )
        result = dict(properties)
        result["item_id"] = item_id
        return result

    def clear(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM edges")
                self._conn.execute("DELETE FROM nodes")

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Dict[str, Any],
        valid_time: Optional[Tuple] = None,  # type: ignore[type-arg]
        created_at: Optional[Tuple] = None,  # type: ignore[type-arg]
        memory_type: Optional[str] = None,
        is_global: bool = False,
    ) -> bool:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO edges "
                    "(source_id, target_id, edge_type, memory_type, valid_from, valid_to, created_at, properties) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET "
                    "    memory_type = COALESCE(excluded.memory_type, edges.memory_type), "
                    "    valid_from  = COALESCE(excluded.valid_from,  edges.valid_from), "
                    "    valid_to    = COALESCE(excluded.valid_to,    edges.valid_to), "
                    "    created_at  = COALESCE(excluded.created_at,  edges.created_at), "
                    "    properties  = excluded.properties",
                    (
                        source_id,
                        target_id,
                        edge_type,
                        memory_type,
                        self._unpack_time(valid_time),
                        self._unpack_valid_to(valid_time),
                        self._unpack_time(created_at),
                        json.dumps(properties),
                    ),
                )
        return True

    def get_node(self, item_id: str, as_of_time: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT item_id, memory_type, valid_from, valid_to, created_at, properties "
                "FROM nodes WHERE item_id=?",
                (item_id,),
            ).fetchone()
        return self._row_to_node(row) if row else None

    def get_neighbors(
        self, item_id: str, edge_type: Optional[str] = None, as_of_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all neighboring nodes in either direction, matching FalkorDB's undirected semantics.

        FalkorDB uses MATCH (n)-[r]-(m) (no arrow) which traverses both outgoing and incoming
        edges. SQLite stores directed edges, so bidirectionality is implemented via UNION of
        outgoing (source_id=?) and incoming (target_id=?) legs.
        """
        _select = (
            "SELECT n.item_id, n.memory_type, n.valid_from, n.valid_to, n.created_at, n.properties "
            "FROM nodes n"
        )
        with self._lock:
            if edge_type:
                sql = (
                    f"{_select} JOIN edges e ON n.item_id = e.target_id "
                    "WHERE e.source_id=? AND e.edge_type=? "
                    "UNION "
                    f"{_select} JOIN edges e ON n.item_id = e.source_id "
                    "WHERE e.target_id=? AND e.edge_type=?"
                )
                rows = self._conn.execute(sql, (item_id, edge_type, item_id, edge_type)).fetchall()
            else:
                sql = (
                    f"{_select} JOIN edges e ON n.item_id = e.target_id WHERE e.source_id=? "
                    "UNION "
                    f"{_select} JOIN edges e ON n.item_id = e.source_id WHERE e.target_id=?"
                )
                rows = self._conn.execute(sql, (item_id, item_id)).fetchall()
        return [self._row_to_node(r) for r in rows]

    def remove_node(self, item_id: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute("DELETE FROM nodes WHERE item_id=?", (item_id,))
        return cursor.rowcount > 0

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None) -> bool:
        with self._lock:
            with self._conn:
                if edge_type:
                    cursor = self._conn.execute(
                        "DELETE FROM edges WHERE source_id=? AND target_id=? AND edge_type=?",
                        (source_id, target_id, edge_type),
                    )
                else:
                    cursor = self._conn.execute(
                        "DELETE FROM edges WHERE source_id=? AND target_id=?",
                        (source_id, target_id),
                    )
        return cursor.rowcount > 0

    # Top-level columns that can be filtered directly without going through properties JSON.
    _NODE_COLUMNS = frozenset({"memory_type", "valid_from", "valid_to", "created_at"})

    def search_nodes(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter nodes by an equality query dict.

        Keys matching top-level node columns (memory_type, valid_from, valid_to, created_at)
        are filtered via direct SQL equality. The special key 'content' uses a LIKE substring
        match. All other keys are matched against the JSON properties blob via json_extract(),
        replicating FalkorDB's equality-filter semantics so callers get consistent results
        regardless of which backend is in use.
        """
        conditions: List[str] = []
        params: List[Any] = []
        for key, value in query.items():
            if key in self._NODE_COLUMNS:
                conditions.append(f"{key} = ?")
                params.append(value)
            elif key == "content":
                # content uses substring match (LIKE) to match existing behaviour
                conditions.append("json_extract(properties, '$.content') LIKE ?")
                params.append(f"%{value}%")
            else:
                # Generic property equality — json path is a bound parameter (SQL-injection safe)
                conditions.append("json_extract(properties, ?) = ?")
                params.append(f"$.{key}")
                params.append(value)
        sql = "SELECT item_id, memory_type, valid_from, valid_to, created_at, properties FROM nodes"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def serialize(self) -> Any:
        with self._lock:
            nodes = self._conn.execute(
                "SELECT item_id, memory_type, valid_from, valid_to, created_at, properties FROM nodes"
            ).fetchall()
            edges = self._conn.execute(
                "SELECT source_id, target_id, edge_type, memory_type, valid_from, valid_to, created_at, properties "
                "FROM edges"
            ).fetchall()
        return {
            "nodes": [
                {
                    "item_id": r[0],
                    "memory_type": r[1],
                    "valid_from": r[2],
                    "valid_to": r[3],
                    "created_at": r[4],
                    "properties": json.loads(r[5]),
                }
                for r in nodes
            ],
            "edges": [
                {
                    "source_id": r[0],
                    "target_id": r[1],
                    "edge_type": r[2],
                    "memory_type": r[3],
                    "valid_from": r[4],
                    "valid_to": r[5],
                    "created_at": r[6],
                    "properties": json.loads(r[7]),
                }
                for r in edges
            ],
        }

    def get_all_edges(self) -> list[dict]:
        """Return all edges in the same shape serialize() produces for edges."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, target_id, edge_type, memory_type, "
                "valid_from, valid_to, created_at, properties FROM edges"
            ).fetchall()
        return [
            {
                "source_id": r[0], "target_id": r[1], "edge_type": r[2],
                "memory_type": r[3], "valid_from": r[4], "valid_to": r[5],
                "created_at": r[6], "properties": json.loads(r[7]),
            }
            for r in rows
        ]

    def get_edges_for_node(self, node_id: str) -> list[dict]:
        """Return all edges where node_id appears as source or target."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, target_id, edge_type, memory_type, "
                "valid_from, valid_to, created_at, properties "
                "FROM edges WHERE source_id=? OR target_id=?",
                (node_id, node_id),
            ).fetchall()
        return [
            {
                "source_id": r[0], "target_id": r[1], "edge_type": r[2],
                "memory_type": r[3], "valid_from": r[4], "valid_to": r[5],
                "created_at": r[6], "properties": json.loads(r[7]),
            }
            for r in rows
        ]

    def get_counts(self) -> dict[str, int]:
        """Return node and edge counts without loading full graph data."""
        with self._lock:
            node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {"node_count": node_count, "edge_count": edge_count}

    def deserialize(self, data: Any) -> None:
        """Load the graph from a serialized format produced by serialize().

        Uses a two-phase approach: nodes are committed first, then edges.
        If an edge references a node not in the data, it is logged and skipped
        rather than rolling back the entire restore. This means there is a brief
        window during restore where nodes exist but their edges are not yet committed.
        Callers should not read the graph concurrently during deserialize().
        """
        # Phase 1: nodes — committed as one transaction
        with self._lock:
            with self._conn:
                for node in data.get("nodes", []):
                    self._conn.execute(
                        "INSERT INTO nodes "
                        "(item_id, memory_type, valid_from, valid_to, created_at, properties) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(item_id) DO UPDATE SET "
                        "    memory_type = excluded.memory_type, "
                        "    valid_from   = excluded.valid_from, "
                        "    valid_to     = excluded.valid_to, "
                        "    created_at   = excluded.created_at, "
                        "    properties   = excluded.properties",
                        (
                            node["item_id"],
                            node.get("memory_type"),
                            node.get("valid_from"),
                            node.get("valid_to"),
                            node.get("created_at"),
                            json.dumps(node.get("properties", {})),
                        ),
                    )
        # Phase 2: edges — separate transaction so FK violations don't roll back nodes
        with self._lock:
            with self._conn:
                for edge in data.get("edges", []):
                    try:
                        self._conn.execute(
                            "INSERT INTO edges "
                            "(source_id, target_id, edge_type, memory_type, valid_from, valid_to, created_at, properties) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                            "ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET "
                            "    memory_type = COALESCE(excluded.memory_type, edges.memory_type), "
                            "    valid_from  = COALESCE(excluded.valid_from,  edges.valid_from), "
                            "    valid_to    = COALESCE(excluded.valid_to,    edges.valid_to), "
                            "    created_at  = COALESCE(excluded.created_at,  edges.created_at), "
                            "    properties  = excluded.properties",
                            (
                                edge["source_id"],
                                edge["target_id"],
                                edge["edge_type"],
                                edge.get("memory_type"),
                                edge.get("valid_from"),
                                edge.get("valid_to"),
                                edge.get("created_at"),
                                json.dumps(edge.get("properties", {})),
                            ),
                        )
                    except sqlite3.IntegrityError as e:
                        logger.warning(
                            "Skipping edge %s→%s: FK violation: %s",
                            edge.get("source_id"),
                            edge.get("target_id"),
                            e,
                        )

    def execute_query(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Cypher queries are not supported in SQLiteBackend. "
            "Use SmartMemory with FalkorDB for temporal queries and graph analytics."
        )
