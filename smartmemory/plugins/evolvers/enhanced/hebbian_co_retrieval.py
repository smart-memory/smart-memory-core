"""Hebbian Co-Retrieval Edge Reinforcement Evolver — CORE-EVO-ENH-3.

Boosts ``metadata["retention_score"]`` on memory nodes that are strongly
co-activated: i.e. whose ``CO_RETRIEVED`` edge weight (``co_retrieval_count``)
exceeds the configured threshold.

Formula:
    boost = min(max_boost, (co_retrieval_count - weight_threshold) * retention_boost_per_unit)
    new_retention = min(1.0, existing_retention + boost)

Nodes appearing in multiple qualifying edges are deduped by ``item_id`` —
each node receives exactly one boost per cycle, computed from the first
qualifying edge encountered (ordered by count DESC).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

_logger = logging.getLogger(__name__)


@dataclass
class HebbianCoRetrievalConfig(MemoryBaseModel):
    """Configuration for HebbianCoRetrievalEvolver.

    Attributes:
        weight_threshold: Minimum ``co_retrieval_count`` required to apply a boost.
        max_edges_per_cycle: LIMIT applied to the Cypher query — caps work per cycle.
        retention_boost_per_unit: Boost per count unit above ``weight_threshold``.
        max_boost: Ceiling for this evolver's single-cycle contribution to retention.
        memory_types: Memory types eligible for boost (working excluded by design).
    """

    weight_threshold: float = 3.0
    max_edges_per_cycle: int = 500
    retention_boost_per_unit: float = 0.02
    max_boost: float = 0.3
    memory_types: list = field(default_factory=lambda: ["episodic", "semantic", "procedural"])


class HebbianCoRetrievalEvolver(EvolverPlugin):
    """Boost retention on nodes that are frequently co-retrieved together.

    Reads ``CO_RETRIEVED`` edges written by ``RetrievalFlushConsumer`` and boosts
    ``retention_score`` on both endpoint nodes proportionally to edge strength.
    Drives from edge strength, not node frequency (contrast with ENH-2).
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="hebbian_co_retrieval",
            version="1.0.0",
            author="SmartMemory Team",
            description="Boost retention for nodes strongly co-activated in search results",
            plugin_type="evolver",
            tags=["hebbian", "co-retrieval", "retention", "graph"],
        )

    def __init__(self, config: HebbianCoRetrievalConfig | None = None) -> None:
        self.config = config or HebbianCoRetrievalConfig()

    def evolve(self, memory: Any, logger: Any = None) -> None:
        """Apply Hebbian co-retrieval retention boost.

        Args:
            memory: Workspace-scoped SmartMemory instance.
            logger: Optional logger override.
        """
        with trace_span("pipeline.evolve.hebbian_co_retrieval", {}):
            self._evolve_impl(memory, logger or _logger)

    def _evolve_impl(self, memory: Any, log: Any) -> None:
        cfg = self.config

        query = """
            MATCH (a)-[r:CO_RETRIEVED]-(b)
            WHERE r.co_retrieval_count >= $threshold
            RETURN a.item_id AS id_a, b.item_id AS id_b, r.co_retrieval_count AS cnt
            ORDER BY r.co_retrieval_count DESC
            LIMIT $limit
        """
        params = {
            "threshold": cfg.weight_threshold,
            "limit": cfg.max_edges_per_cycle,
        }

        try:
            rows = memory.execute_cypher(query, params)
        except Exception as exc:
            log.warning("HebbianCoRetrievalEvolver: query failed — %s", exc)
            return

        if not rows:
            log.debug("HebbianCoRetrievalEvolver: no qualifying CO_RETRIEVED edges found")
            return

        seen: set[str] = set()
        boosted = 0

        for row in rows:
            for item_id in (row[0], row[1]):  # id_a, id_b
                if not item_id or item_id in seen:
                    continue
                seen.add(item_id)
                try:
                    item = memory.get(item_id)
                    if item is None:
                        continue
                    base = float(item.metadata.get("retention_score", 1.0))
                    cnt = float(row[2])
                    boost = min(
                        cfg.max_boost,
                        (cnt - cfg.weight_threshold) * cfg.retention_boost_per_unit,
                    )
                    new_score = min(1.0, base + boost)
                    memory.update_properties(item_id, {"retention_score": new_score})
                    boosted += 1
                except Exception as exc:
                    log.warning("HebbianCoRetrievalEvolver: skipping %s — %s", item_id, exc)

        log.info("HebbianCoRetrievalEvolver: boosted %d item(s)", boosted)
