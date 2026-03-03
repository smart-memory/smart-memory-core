"""GraphAlgos — backend-agnostic protocol for graph-algorithmic operations.

Two implementations:
- NetworkXAlgos (smartmemory.graph.networkx_algos) — backed by GraphComputeLayer, for SQLiteBackend
- CypherAlgos  (smartmemory.graph.cypher_algos)   — backed by execute_cypher, for FalkorDBBackend

Callers import this protocol and never depend on NetworkX or Cypher directly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable


@runtime_checkable
class GraphAlgos(Protocol):
    """Backend-agnostic interface for graph-algorithmic operations.

    All methods operate on the full graph visible to the current backend
    (tenant scoping is the backend's responsibility, not the protocol's).
    """

    # ── Aggregate queries ─────────────────────────────────────────────

    def orphan_nodes(self) -> List[str]:
        """Return item_ids of nodes with zero edges (in or out)."""
        ...

    def edge_type_counts(self) -> Dict[str, int]:
        """Return {edge_type: count} for every distinct edge type."""
        ...

    def degree_map(self) -> Dict[str, int]:
        """Return {item_id: degree} where degree = in_degree + out_degree."""
        ...

    # ── Path finding ──────────────────────────────────────────────────

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> List[List[str]]:
        """Return all simple paths (as lists of item_ids) up to max_depth hops."""
        ...

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        """Return the shortest path as a list of item_ids, or None if unreachable."""
        ...

    # ── Transitive closure & pattern matching ─────────────────────────

    def transitive_closure(self, edge_type: str) -> List[Tuple[str, str]]:
        """Return all (source, target) pairs reachable via chains of edge_type.

        Only returns pairs where no direct edge of that type already exists.
        Used by inference rules (e.g. causal transitivity).
        """
        ...

    def pattern_match_2hop(
        self,
        edge_type_1: str,
        edge_type_2: str,
    ) -> List[Tuple[str, str, str]]:
        """Return (a, b, c) triples where a-[edge_type_1]->b-[edge_type_2]->c."""
        ...

    # ── Centrality & community ────────────────────────────────────────

    def betweenness_centrality(self) -> Dict[str, float]:
        """Return {item_id: centrality_score} for all nodes."""
        ...

    def connected_components(self) -> List[Set[str]]:
        """Return list of connected components (weakly connected, ignoring direction)."""
        ...
