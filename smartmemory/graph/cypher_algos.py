"""CypherAlgos — GraphAlgos implementation backed by FalkorDB Cypher queries.

Used by FalkorDBBackend.  Each protocol method translates to one or two
Cypher queries executed via backend.execute_cypher().
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from smartmemory.graph.backends.falkordb import FalkorDBBackend

logger = logging.getLogger(__name__)


class CypherAlgos:
    """Graph algorithms implemented via Cypher on FalkorDB."""

    def __init__(self, backend: FalkorDBBackend) -> None:
        self._backend = backend

    def _q(self, cypher: str, params: dict[str, Any] | None = None) -> list:
        return self._backend.execute_cypher(cypher, params)

    # ── Aggregate queries ─────────────────────────────────────────────

    def orphan_nodes(self) -> List[str]:
        rows = self._q("MATCH (n) WHERE NOT (n)-[]-() RETURN n.item_id")
        return [r[0] for r in rows if r[0] is not None]

    def edge_type_counts(self) -> Dict[str, int]:
        rows = self._q("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c")
        return {r[0]: int(r[1]) for r in rows}

    def degree_map(self) -> Dict[str, int]:
        # FalkorDB doesn't have a built-in degree() function, so count both legs.
        rows = self._q(
            "MATCH (n) "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN n.item_id, count(r)"
        )
        return {r[0]: int(r[1]) for r in rows if r[0] is not None}

    # ── Path finding ──────────────────────────────────────────────────

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> List[List[str]]:
        rows = self._q(
            "MATCH (s {item_id: $src}), (t {item_id: $tgt}), "
            f"path = (s)-[*1..{max_depth}]-(t) "
            "RETURN [n IN nodes(path) | n.item_id]",
            {"src": source_id, "tgt": target_id},
        )
        return [list(r[0]) for r in rows]

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        rows = self._q(
            "MATCH (s {item_id: $src}), (t {item_id: $tgt}), "
            f"path = shortestPath((s)-[*..{max_depth}]-(t)) "
            "RETURN [n IN nodes(path) | n.item_id]",
            {"src": source_id, "tgt": target_id},
        )
        if rows:
            return list(rows[0][0])
        return None

    # ── Transitive closure & pattern matching ─────────────────────────

    def transitive_closure(self, edge_type: str) -> List[Tuple[str, str]]:
        rows = self._q(
            f"MATCH (a)-[:{edge_type}*2..]->(c) "
            f"WHERE a <> c AND NOT (a)-[:{edge_type}]->(c) "
            "RETURN DISTINCT a.item_id, c.item_id",
        )
        return [(r[0], r[1]) for r in rows]

    def pattern_match_2hop(
        self,
        edge_type_1: str,
        edge_type_2: str,
    ) -> List[Tuple[str, str, str]]:
        rows = self._q(
            f"MATCH (a)-[:{edge_type_1}]->(b)-[:{edge_type_2}]->(c) "
            "WHERE a <> c "
            "RETURN a.item_id, b.item_id, c.item_id",
        )
        return [(r[0], r[1], r[2]) for r in rows]

    # ── Centrality & community ────────────────────────────────────────

    def betweenness_centrality(self) -> Dict[str, float]:
        # FalkorDB doesn't have a native betweenness_centrality function.
        # Fall back to loading nodes/edges into NetworkX for this operation.
        try:
            import networkx as nx_
        except ImportError:
            logger.warning("networkx not installed; betweenness_centrality unavailable on FalkorDB")
            return {}

        g = nx_.DiGraph()
        for row in self._q("MATCH (n) RETURN n.item_id"):
            if row[0]:
                g.add_node(row[0])
        for row in self._q("MATCH (a)-[r]->(b) RETURN a.item_id, b.item_id"):
            if row[0] and row[1]:
                g.add_edge(row[0], row[1])
        if g.number_of_nodes() == 0:
            return {}
        return nx_.betweenness_centrality(g)

    def connected_components(self) -> List[Set[str]]:
        # FalkorDB doesn't have native connected_components.
        # Fall back to NetworkX.
        try:
            import networkx as nx_
        except ImportError:
            logger.warning("networkx not installed; connected_components unavailable on FalkorDB")
            return []

        g = nx_.Graph()
        for row in self._q("MATCH (n) RETURN n.item_id"):
            if row[0]:
                g.add_node(row[0])
        for row in self._q("MATCH (a)-[r]-(b) RETURN DISTINCT a.item_id, b.item_id"):
            if row[0] and row[1]:
                g.add_edge(row[0], row[1])
        return [set(c) for c in nx_.connected_components(g)]
