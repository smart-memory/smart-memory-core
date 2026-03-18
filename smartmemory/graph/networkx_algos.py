"""NetworkXAlgos — GraphAlgos implementation backed by GraphComputeLayer.

Used by SQLiteBackend.  All algorithms run against the in-memory nx.MultiDiGraph
maintained by GraphComputeLayer.

Each method captures ONE snapshot at the start and operates on it throughout.
This avoids repeated graph copies and ensures consistency within a single call.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from smartmemory.graph.compute import GraphComputeLayer


class NetworkXAlgos:
    """Graph algorithms implemented via NetworkX, powered by GraphComputeLayer."""

    def __init__(self, compute: GraphComputeLayer) -> None:
        self._compute = compute

    # ── Aggregate queries ─────────────────────────────────────────────

    def orphan_nodes(self) -> List[str]:
        g = self._compute.snapshot()
        return [n for n in g.nodes if g.degree(n) == 0]

    def edge_type_counts(self) -> Dict[str, int]:
        g = self._compute.snapshot()
        counts: Dict[str, int] = {}
        for _, _, data in g.edges(data=True):
            etype = data.get("edge_type", "")
            counts[etype] = counts.get(etype, 0) + 1
        return counts

    def degree_map(self) -> Dict[str, int]:
        g = self._compute.snapshot()
        return dict(g.degree())

    # ── Path finding ──────────────────────────────────────────────────

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> List[List[str]]:
        g = self._compute.snapshot()
        if source_id not in g or target_id not in g:
            return []
        undirected = g.to_undirected(as_view=True)
        try:
            return list(nx.all_simple_paths(undirected, source_id, target_id, cutoff=max_depth))
        except nx.NetworkXError:
            return []

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        g = self._compute.snapshot()
        if source_id not in g or target_id not in g:
            return None
        undirected = g.to_undirected(as_view=True)
        try:
            path = nx.shortest_path(undirected, source_id, target_id)
            return path if len(path) - 1 <= max_depth else None
        except nx.NetworkXNoPath:
            return None

    # ── Transitive closure & pattern matching ─────────────────────────

    def transitive_closure(self, edge_type: str) -> List[Tuple[str, str]]:
        g = self._compute.snapshot()
        sub = nx.DiGraph()
        for u, v, data in g.edges(data=True):
            if data.get("edge_type") == edge_type:
                sub.add_edge(u, v)
        result: List[Tuple[str, str]] = []
        for source in sub.nodes:
            for target in nx.descendants(sub, source):
                if source != target and not sub.has_edge(source, target):
                    result.append((source, target))
        return result

    def pattern_match_2hop(
        self,
        edge_type_1: str,
        edge_type_2: str,
    ) -> List[Tuple[str, str, str]]:
        g = self._compute.snapshot()
        result: List[Tuple[str, str, str]] = []
        for a, b, d1 in g.edges(data=True):
            if d1.get("edge_type") != edge_type_1:
                continue
            for _, c, d2 in g.out_edges(b, data=True):
                if d2.get("edge_type") == edge_type_2 and a != c:
                    result.append((a, b, c))
        return result

    # ── Centrality & community ────────────────────────────────────────

    def betweenness_centrality(self) -> Dict[str, float]:
        g = self._compute.snapshot()
        if g.number_of_nodes() == 0:
            return {}
        return nx.betweenness_centrality(g)

    def connected_components(self) -> List[Set[str]]:
        g = self._compute.snapshot()
        undirected = g.to_undirected(as_view=True)
        return [set(c) for c in nx.connected_components(undirected)]
