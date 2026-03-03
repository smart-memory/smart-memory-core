"""Graph health metrics collection.

Measures graph quality: orphan nodes, edge distribution, provenance coverage.
Backend-agnostic: uses SmartGraphBackend query methods and the GraphAlgos protocol
so the same code works for both SQLiteBackend and FalkorDBBackend.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from smartmemory.models.base import MemoryBaseModel

logger = logging.getLogger(__name__)

# Edge types that count as "provenance" for decision nodes.
_PROVENANCE_EDGE_TYPES = frozenset({"DERIVED_FROM", "CAUSED_BY", "PRODUCED"})

ORPHAN_THRESHOLD = 0.2
PROVENANCE_THRESHOLD = 0.5


@dataclass
class HealthReport(MemoryBaseModel):
    """Graph health metrics snapshot."""

    total_nodes: int = 0
    total_edges: int = 0
    orphan_count: int = 0
    type_distribution: dict[str, int] = field(default_factory=dict)
    edge_distribution: dict[str, int] = field(default_factory=dict)
    provenance_coverage: float = 0.0

    @property
    def orphan_ratio(self) -> float:
        """Fraction of nodes with no connections."""
        if self.total_nodes == 0:
            return 0.0
        return self.orphan_count / self.total_nodes

    @property
    def is_healthy(self) -> bool:
        """Whether the graph meets health thresholds."""
        return self.orphan_ratio < ORPHAN_THRESHOLD and self.provenance_coverage > PROVENANCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "orphan_count": self.orphan_count,
            "orphan_ratio": round(self.orphan_ratio, 4),
            "type_distribution": self.type_distribution,
            "edge_distribution": self.edge_distribution,
            "provenance_coverage": round(self.provenance_coverage, 4),
            "is_healthy": self.is_healthy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthReport":
        return cls(
            total_nodes=data.get("total_nodes", 0),
            total_edges=data.get("total_edges", 0),
            orphan_count=data.get("orphan_count", 0),
            type_distribution=data.get("type_distribution", {}),
            edge_distribution=data.get("edge_distribution", {}),
            provenance_coverage=data.get("provenance_coverage", 0.0),
        )


class GraphHealthChecker:
    """Collects graph health metrics using backend-agnostic queries.

    Works on both SQLiteBackend (via GraphAlgos / get_all_nodes) and
    FalkorDBBackend (same interface).

    Usage:
        checker = GraphHealthChecker(smart_memory)
        report = checker.collect_health()
        if not report.is_healthy:
            logger.warning("Graph unhealthy: %s", report.to_dict())
    """

    def __init__(self, memory: Any, graph: Any = None):
        self.memory = memory
        self.graph = graph or getattr(memory, "_graph", None)

    @property
    def _backend(self) -> Any:
        if self.graph is None:
            return None
        return getattr(self.graph, "backend", self.graph)

    def collect_health(self) -> HealthReport:
        """Collect all health metrics into a single report."""
        report = HealthReport()
        try:
            report.total_nodes = self._count_nodes()
            report.total_edges = self._count_edges()
            report.type_distribution = self._type_distribution()
            report.edge_distribution = self._edge_distribution()
            report.orphan_count = self._count_orphans()
            report.provenance_coverage = self._provenance_coverage()
        except Exception as e:
            logger.warning("Health check failed: %s", e)
        return report

    def _count_nodes(self) -> int:
        backend = self._backend
        if backend is None:
            return 0
        if hasattr(backend, "count_nodes"):
            return backend.count_nodes()
        return len(backend.get_all_nodes())

    def _count_edges(self) -> int:
        backend = self._backend
        if backend is None:
            return 0
        if hasattr(backend, "count_edges"):
            return backend.count_edges()
        return sum(backend.algos.edge_type_counts().values())

    def _type_distribution(self) -> dict[str, int]:
        backend = self._backend
        if backend is None:
            return {}
        dist: dict[str, int] = {}
        for node in backend.get_all_nodes():
            mt = node.get("memory_type")
            if mt:
                dist[mt] = dist.get(mt, 0) + 1
        return dist

    def _edge_distribution(self) -> dict[str, int]:
        backend = self._backend
        if backend is None:
            return {}
        return backend.algos.edge_type_counts()

    def _count_orphans(self) -> int:
        backend = self._backend
        if backend is None:
            return 0
        return len(backend.algos.orphan_nodes())

    def _provenance_coverage(self) -> float:
        """Fraction of decision nodes that have at least one provenance edge."""
        backend = self._backend
        if backend is None:
            return 1.0
        decisions = [
            n for n in backend.get_all_nodes()
            if n.get("memory_type") == "decision"
        ]
        if not decisions:
            return 1.0
        with_prov = 0
        for d in decisions:
            item_id = d.get("item_id")
            if not item_id:
                continue
            edges = backend.get_edges_for_node(item_id)
            if any(e.get("edge_type") in _PROVENANCE_EDGE_TYPES for e in edges):
                with_prov += 1
        return with_prov / len(decisions)
