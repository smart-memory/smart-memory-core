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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from smartmemory.graph.algos import GraphAlgos
from smartmemory.graph.backends.backend import SmartGraphBackend
from smartmemory.graph.compute import GraphComputeLayer
from smartmemory.graph.networkx_algos import NetworkXAlgos
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

_UPSERT_NODE_SQL = (
    "INSERT INTO nodes "
    "(item_id, memory_type, valid_from, valid_to, created_at, properties) "
    "VALUES (?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(item_id) DO UPDATE SET "
    "    memory_type = excluded.memory_type, "
    "    valid_from   = excluded.valid_from, "
    "    valid_to     = excluded.valid_to, "
    "    created_at   = excluded.created_at, "
    "    properties   = excluded.properties"
)

_UPSERT_EDGE_SQL = (
    "INSERT INTO edges "
    "(source_id, target_id, edge_type, memory_type, valid_from, valid_to, created_at, properties) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET "
    "    memory_type = COALESCE(excluded.memory_type, edges.memory_type), "
    "    valid_from  = COALESCE(excluded.valid_from,  edges.valid_from), "
    "    valid_to    = COALESCE(excluded.valid_to,    edges.valid_to), "
    "    created_at  = COALESCE(excluded.created_at,  edges.created_at), "
    "    properties  = excluded.properties"
)


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
        self._lock = threading.RLock()  # RLock: transaction_context() + nested mutators
        self._in_transaction = False  # CORE-EVO-LIVE-1: suppress nested auto-commit
        self._deferred_syncs: list = []  # CORE-EVO-LIVE-1: compute syncs deferred until commit
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        result = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        self._journal_mode = result[0] if result else "unknown"
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        # GraphComputeLayer mirrors SQL state into an in-memory nx.DiGraph.
        # NetworkXAlgos delegates algorithmic operations to that DiGraph.
        self._compute = GraphComputeLayer(self)
        self._algos: GraphAlgos = NetworkXAlgos(self._compute)

    @property
    def algos(self) -> GraphAlgos:
        return self._algos

    def _create_tables(self) -> None:
        with self._lock:
            with self._auto_commit():
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

    @staticmethod
    def _row_to_edge(row: Any) -> Dict[str, Any]:
        return {
            "source_id": row[0],
            "target_id": row[1],
            "edge_type": row[2],
            "memory_type": row[3],
            "valid_from": row[4],
            "valid_to": row[5],
            "created_at": row[6],
            "properties": json.loads(row[7]),
        }

    def close(self) -> None:
        """Close the SQLite connection, flushing WAL to the main database file."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    # ── CORE-EVO-LIVE-1: property merge + transaction support ────────────────

    def set_properties(self, item_id: str, properties: Dict[str, Any]) -> bool:
        """Merge properties into an existing node (partial update).

        Reads the current JSON blob, merges the new keys, writes back.
        Syncs the compute layer after the write.
        """
        if not properties:
            return True
        with self._lock:
            row = self._conn.execute(
                "SELECT memory_type, properties FROM nodes WHERE item_id=?", (item_id,)
            ).fetchone()
            if row is None:
                return False
            memory_type, props_json = row
            existing = json.loads(props_json)
            existing.update(properties)
            with self._auto_commit():
                self._conn.execute(
                    "UPDATE nodes SET properties=? WHERE item_id=?",
                    (json.dumps(existing), item_id),
                )
        # Sync compute layer with the full merged property set
        full_props = dict(existing)
        full_props["item_id"] = item_id
        full_props["memory_type"] = memory_type
        self._sync_compute("sync_add_node", item_id, full_props)
        return True

    def _sync_compute(self, fn_name: str, *args, **kwargs) -> None:
        """Execute a compute layer sync immediately, or defer if inside a transaction."""
        if self._in_transaction:
            self._deferred_syncs.append((fn_name, args, kwargs))
        else:
            getattr(self._compute, fn_name)(*args, **kwargs)

    def _flush_deferred_syncs(self) -> None:
        """Replay all deferred compute syncs (called after COMMIT)."""
        for fn_name, args, kwargs in self._deferred_syncs:
            try:
                getattr(self._compute, fn_name)(*args, **kwargs)
            except Exception as exc:
                logger.warning("SQLiteBackend: deferred sync %s failed: %s", fn_name, exc)
        self._deferred_syncs.clear()

    @contextmanager
    def _auto_commit(self) -> Generator[None, None, None]:
        """Context manager that wraps in ``with self._conn:`` only when no outer transaction is active.

        When ``self._in_transaction`` is True (inside ``transaction_context()``),
        this yields without any commit/rollback — the outer transaction handles it.
        """
        if self._in_transaction:
            yield
        else:
            with self._conn:
                yield

    @contextmanager
    def transaction_context(self) -> Generator[None, None, None]:
        """Wrap multiple operations in a single SQLite transaction.

        Uses BEGIN IMMEDIATE for write-ahead locking. Rolls back on exception.
        Mutators called inside this context use ``_auto_commit()`` which
        becomes a no-op, preventing premature commits.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_transaction = True
            self._deferred_syncs.clear()
            try:
                yield
                self._conn.execute("COMMIT")
                # Replay compute syncs only after successful commit
                self._flush_deferred_syncs()
            except Exception:
                self._conn.execute("ROLLBACK")
                self._deferred_syncs.clear()  # Discard phantom syncs
                raise
            finally:
                self._in_transaction = False

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
            with self._auto_commit():
                self._conn.execute(
                    _UPSERT_NODE_SQL,
                    (
                        item_id,
                        mem_type,
                        self._unpack_time(valid_time),
                        self._unpack_valid_to(valid_time),
                        self._unpack_time(created_at),
                        json.dumps(props_to_store),
                    ),
                )
        self._sync_compute("sync_add_node", item_id, properties)
        result = dict(properties)
        result["item_id"] = item_id
        return result

    def clear(self) -> None:
        with self._lock:
            with self._auto_commit():
                self._conn.execute("DELETE FROM edges")
                self._conn.execute("DELETE FROM nodes")
        self._sync_compute("sync_clear")

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
            with self._auto_commit():
                self._conn.execute(
                    _UPSERT_EDGE_SQL,
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
        self._sync_compute("sync_add_edge", source_id, target_id, edge_type, properties)
        return True

    def get_node(self, item_id: str, as_of_time: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT item_id, memory_type, valid_from, valid_to, created_at, properties FROM nodes WHERE item_id=?",
                (item_id,),
            ).fetchone()
        return self._row_to_node(row) if row else None

    def get_neighbors(
        self,
        item_id: str,
        edge_type: Optional[str] = None,
        as_of_time: Optional[str] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """Return neighboring nodes, optionally filtered by edge direction.

        Args:
            direction: ``"both"`` (default) — UNION of outgoing and incoming legs,
                ``"outgoing"`` — only targets of edges where item_id is source,
                ``"incoming"`` — only sources of edges where item_id is target.
        """
        _select = "SELECT n.item_id, n.memory_type, n.valid_from, n.valid_to, n.created_at, n.properties FROM nodes n"
        # Build directional legs
        if edge_type:
            outgoing = f"{_select} JOIN edges e ON n.item_id = e.target_id WHERE e.source_id=? AND e.edge_type=?"
            incoming = f"{_select} JOIN edges e ON n.item_id = e.source_id WHERE e.target_id=? AND e.edge_type=?"
            out_params: tuple = (item_id, edge_type)
            in_params: tuple = (item_id, edge_type)
        else:
            outgoing = f"{_select} JOIN edges e ON n.item_id = e.target_id WHERE e.source_id=?"
            incoming = f"{_select} JOIN edges e ON n.item_id = e.source_id WHERE e.target_id=?"
            out_params = (item_id,)
            in_params = (item_id,)

        with self._lock:
            if direction == "outgoing":
                rows = self._conn.execute(outgoing, out_params).fetchall()
            elif direction == "incoming":
                rows = self._conn.execute(incoming, in_params).fetchall()
            else:  # "both" — backward-compatible default
                sql = f"{outgoing} UNION {incoming}"
                rows = self._conn.execute(sql, out_params + in_params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def remove_node(self, item_id: str) -> bool:
        with self._lock:
            with self._auto_commit():
                cursor = self._conn.execute("DELETE FROM nodes WHERE item_id=?", (item_id,))
        if cursor.rowcount > 0:
            self._sync_compute("sync_remove_node", item_id)
        return cursor.rowcount > 0

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None) -> bool:
        with self._lock:
            with self._auto_commit():
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
        if cursor.rowcount > 0:
            self._sync_compute("sync_remove_edge", source_id, target_id, edge_type)
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
        # NOTE: serialize() intentionally does NOT use _row_to_node() because
        # deserialize() expects a nested "properties" key.  _row_to_node()
        # flattens properties into top-level keys for API consumers — different
        # contract than the round-trip format used here.
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
            "edges": [self._row_to_edge(r) for r in edges],
        }

    def get_all_edges(self) -> list[dict]:
        """Return all edges in the same shape serialize() produces for edges."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, target_id, edge_type, memory_type, "
                "valid_from, valid_to, created_at, properties FROM edges"
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_for_node(self, node_id: str) -> list[dict]:
        """Return all edges where node_id appears as source or target."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, target_id, edge_type, memory_type, "
                "valid_from, valid_to, created_at, properties "
                "FROM edges WHERE source_id=? OR target_id=?",
                (node_id, node_id),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

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
            with self._auto_commit():
                for node in data.get("nodes", []):
                    self._conn.execute(
                        _UPSERT_NODE_SQL,
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
            with self._auto_commit():
                for edge in data.get("edges", []):
                    try:
                        self._conn.execute(
                            _UPSERT_EDGE_SQL,
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
        # Re-read all nodes/edges from SQL into the in-memory DiGraph.
        self._compute.reload()

    # ── DIST-LITE-PARITY-1 Phase 1: missing query methods ─────────────────────

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Return all nodes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT item_id, memory_type, valid_from, valid_to, created_at, properties FROM nodes"
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_nodes(self) -> int:
        """Return total number of nodes."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def count_edges(self) -> int:
        """Return total number of edges."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def get_node_types(self) -> List[str]:
        """Return distinct memory_type values across all nodes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT memory_type FROM nodes WHERE memory_type IS NOT NULL"
            ).fetchall()
        return [r[0] for r in rows]

    def get_edge_types(self) -> List[str]:
        """Return distinct edge_type values across all edges."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT edge_type FROM edges WHERE edge_type IS NOT NULL"
            ).fetchall()
        return [r[0] for r in rows]

    def search_nodes_by_type_or_tag(self, type_or_tag: str, is_global: bool = False) -> List[Dict[str, Any]]:
        """Return nodes whose memory_type matches OR whose tags property contains the value.

        Uses json_each() for exact element matching within the tags JSON array,
        avoiding substring false positives (e.g. "note" matching "notebook").
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT item_id, memory_type, valid_from, valid_to, created_at, properties "
                "FROM nodes "
                "WHERE memory_type = ? "
                "   OR EXISTS ("
                "       SELECT 1 FROM json_each(json_extract(properties, '$.tags')) "
                "       WHERE value = ?"
                "   )",
                (type_or_tag, type_or_tag),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def merge_nodes(self, target_id: str, source_ids: List[str]) -> bool:
        """Merge multiple source nodes into a target node.

        Matches FalkorDB merge_nodes semantics:
        1. Rewire outgoing edges: (source→other) becomes (target→other), skip self-loops
        2. Rewire incoming edges: (other→source) becomes (other→target), skip self-loops
        3. Merge properties: source fills missing keys, target wins on collision
        4. Delete source nodes (CASCADE removes remaining edges like source↔target)
        5. Reload the in-memory compute graph

        Uses UPSERT (INSERT OR REPLACE) for rewired edges to handle PK conflicts
        when the target already has an edge to the same neighbor with the same type.
        """
        if not source_ids:
            return True
        try:
            with self._lock:
                # Guard: target must exist — otherwise source deletion causes silent data loss.
                target_exists = self._conn.execute(
                    "SELECT 1 FROM nodes WHERE item_id=?", (target_id,)
                ).fetchone()
                if not target_exists:
                    logger.error("merge_nodes: target %s does not exist", target_id)
                    return False
                with self._auto_commit():
                    for source_id in source_ids:
                        if source_id == target_id:
                            continue

                        # 1. Rewire outgoing: (source)-[r]->(other) → (target)-[r]->(other)
                        #    Skip edges pointing at target (would create self-loop).
                        outgoing = self._conn.execute(
                            "SELECT source_id, target_id, edge_type, memory_type, "
                            "valid_from, valid_to, created_at, properties "
                            "FROM edges WHERE source_id=? AND target_id!=?",
                            (source_id, target_id),
                        ).fetchall()
                        for row in outgoing:
                            self._conn.execute(
                                _UPSERT_EDGE_SQL,
                                (target_id, row[1], row[2], row[3], row[4], row[5], row[6], row[7]),
                            )

                        # 2. Rewire incoming: (other)-[r]->(source) → (other)-[r]->(target)
                        #    Skip edges coming from target (would create self-loop).
                        incoming = self._conn.execute(
                            "SELECT source_id, target_id, edge_type, memory_type, "
                            "valid_from, valid_to, created_at, properties "
                            "FROM edges WHERE target_id=? AND source_id!=?",
                            (source_id, target_id),
                        ).fetchall()
                        for row in incoming:
                            self._conn.execute(
                                _UPSERT_EDGE_SQL,
                                (row[0], target_id, row[2], row[3], row[4], row[5], row[6], row[7]),
                            )

                        # 3. Merge properties (target wins on collision)
                        source_row = self._conn.execute(
                            "SELECT properties FROM nodes WHERE item_id=?", (source_id,)
                        ).fetchone()
                        target_row = self._conn.execute(
                            "SELECT properties FROM nodes WHERE item_id=?", (target_id,)
                        ).fetchone()
                        if source_row and target_row:
                            source_props = json.loads(source_row[0])
                            target_props = json.loads(target_row[0])
                            merged = {**source_props, **target_props}
                            self._conn.execute(
                                "UPDATE nodes SET properties=? WHERE item_id=?",
                                (json.dumps(merged), target_id),
                            )

                        # 4. Delete source node (CASCADE removes source↔target edges)
                        self._conn.execute("DELETE FROM nodes WHERE item_id=?", (source_id,))

            # 5. Rebuild the in-memory graph from SQL state
            self._compute.reload()
            return True
        except Exception as e:
            logger.error("Failed to merge nodes into %s: %s", target_id, e)
            return False

    # ── DIST-LITE-PARITY-1 Phase 4: optimizations ──────────────────────────────

    def add_nodes_bulk(
        self, nodes: List[Dict[str, Any]], batch_size: int = 500, is_global: bool = False
    ) -> int:
        """Bulk upsert nodes using executemany() — bypasses per-item sync hooks.

        Calls _compute.reload() once at the end to sync the in-memory graph.
        """
        rows = []
        for node in nodes:
            item_id = node.get("item_id")
            if item_id is None:
                item_id = str(uuid.uuid4())
            mem_type = node.get("memory_type")
            props = {k: v for k, v in node.items() if k not in ("item_id", "memory_type")}
            rows.append((
                item_id,
                mem_type,
                node.get("valid_from"),
                node.get("valid_to"),
                node.get("created_at"),
                json.dumps(props),
            ))
        with self._lock:
            with self._auto_commit():
                self._conn.executemany(_UPSERT_NODE_SQL, rows)
        self._compute.reload()
        return len(rows)

    def add_edges_bulk(
        self, edges: List[Tuple[str, str, str, Dict[str, Any]]], batch_size: int = 500, is_global: bool = False
    ) -> int:
        """Bulk upsert edges — skips edges referencing missing nodes.

        FK constraints cause ``executemany`` to abort the whole batch on one bad
        reference.  To match FalkorDB's per-edge graceful degradation, edges are
        inserted individually so valid edges still land when some reference
        non-existent nodes.

        Calls ``_compute.reload()`` once at the end to sync the in-memory graph.
        """
        created = 0
        with self._lock:
            for src, tgt, etype, props in edges:
                row = (
                    src,
                    tgt,
                    etype,
                    props.get("memory_type"),
                    props.get("valid_from"),
                    props.get("valid_to"),
                    props.get("created_at"),
                    json.dumps(props),
                )
                try:
                    with self._auto_commit():
                        self._conn.execute(_UPSERT_EDGE_SQL, row)
                    created += 1
                except sqlite3.IntegrityError:
                    logger.debug("add_edges_bulk: skipped edge %s->%s (FK miss)", src, tgt)
        self._compute.reload()
        return created

    def health_check(self) -> Dict[str, Any]:
        """Run SQLite PRAGMA integrity_check and return diagnostic info."""
        with self._lock:
            result = self._conn.execute("PRAGMA integrity_check").fetchone()
        ok = result is not None and result[0] == "ok"
        return {
            "status": "ok" if ok else "error",
            "integrity_check": result[0] if result else "unknown",
            "backend": "sqlite",
        }

    def backend_info(self) -> Dict[str, Any]:
        """Return SQLite backend metadata for diagnostics."""
        info: Dict[str, Any] = {
            "backend": "sqlite",
            "journal_mode": self._journal_mode,
            "node_count": self.count_nodes(),
            "edge_count": self.count_edges(),
            "in_memory": self._is_memory_db,
        }
        if not self._is_memory_db:
            try:
                db_path = self._conn.execute("PRAGMA database_list").fetchone()
                if db_path and db_path[2]:
                    info["file_size_bytes"] = Path(db_path[2]).stat().st_size
            except Exception:
                pass
        return info

    def execute_query(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Cypher queries are not supported in SQLiteBackend. "
            "Use SmartMemory with FalkorDB for temporal queries and graph analytics."
        )
