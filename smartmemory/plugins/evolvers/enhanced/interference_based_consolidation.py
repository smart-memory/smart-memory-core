"""Interference-Based Consolidation Evolver.

Models retroactive and proactive interference between similar memories.
When two memories are highly similar (cosine similarity > threshold), each
reduces the other's ``retention_score`` proportionally to their similarity:

    new_score = current_score * (1 - similarity * interference_weight)

Pairs are processed at most once (via a ``frozenset`` deduplication set) to
avoid double-counting, and items without embeddings are skipped because cosine
similarity cannot be computed without a vector.
"""

import logging
from dataclasses import dataclass
from typing import Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

_logger = logging.getLogger(__name__)

_MEMORY_TYPES = ["episodic", "semantic"]


@dataclass
class InterferenceBasedConsolidationConfig(MemoryBaseModel):
    """Configuration for InterferenceBasedConsolidationEvolver.

    Attributes:
        similarity_threshold: Minimum cosine similarity for interference to trigger.
        interference_weight: Fraction of retention score lost per interfering neighbor.
        top_k_neighbors: Number of nearest neighbors to search per memory.
        max_memories: Cap on memories processed per cycle (O(n*k) safety valve).
    """

    similarity_threshold: float = 0.85
    interference_weight: float = 0.05
    top_k_neighbors: int = 5
    max_memories: int = 200


class InterferenceBasedConsolidationEvolver(EvolverPlugin):
    """Penalise similar memories for competing in recall (interference theory).

    For each memory with an embedding, searches for semantically close neighbors.
    Pairs that exceed *similarity_threshold* receive a proportional penalty on
    their ``metadata["retention_score"]``.

    Pair deduplication ensures each (A, B) pair is penalised exactly once.
    Items without embeddings are silently skipped.

    Follows Pattern B: reads via ``memory.search()`` and writes via
    ``memory.update_properties()``.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="interference_based_consolidation",
            version="1.0.0",
            author="SmartMemory Team",
            description="Interference-theory-based retention penalty for competing similar memories",
            plugin_type="evolver",
            tags=["interference", "consolidation", "retention", "similarity"],
        )

    def __init__(self, config: InterferenceBasedConsolidationConfig | None = None) -> None:
        self.config = config or InterferenceBasedConsolidationConfig()

    def evolve(self, memory: Any, logger: Any = None) -> None:
        """Apply interference-based retention penalties to episodic and semantic memories.

        Args:
            memory: SmartMemory instance (workspace-scoped).
            logger: Optional logger override.
        """
        memory_id = getattr(memory, "item_id", None)
        with trace_span(
            "pipeline.evolve.interference_based_consolidation",
            {"memory_id": memory_id, "similarity_threshold": self.config.similarity_threshold},
        ):
            self._evolve_impl(memory, logger or _logger)

    def _evolve_impl(self, memory: Any, log: Any) -> None:
        cfg = self.config

        try:
            items = memory.search(query="*", memory_types=_MEMORY_TYPES, limit=cfg.max_memories)
        except Exception as exc:
            log.warning(f"InterferenceBasedConsolidationEvolver: initial search failed — {exc}")
            return

        if not items:
            log.debug("InterferenceBasedConsolidationEvolver: no memories found")
            return

        # Filter to items that have embeddings (cosine_sim requires a vector).
        embedded_items = [it for it in items if it.embedding]
        log.info(f"InterferenceBasedConsolidationEvolver: {len(embedded_items)}/{len(items)} items have embeddings")
        if not embedded_items:
            return

        processed_pairs: set[frozenset] = set()
        total_penalised = 0

        for item in embedded_items:
            current_score: float = float(item.metadata.get("retention_score", 1.0))
            interference_count: int = int(item.metadata.get("interference_count", 0))

            try:
                neighbors = memory.search(
                    query=item.content,
                    memory_types=_MEMORY_TYPES,
                    limit=cfg.top_k_neighbors,
                )
            except Exception as exc:
                log.debug(f"InterferenceBasedConsolidationEvolver: neighbor search failed for {item.item_id} — {exc}")
                continue

            if not neighbors:
                continue

            for neighbor in neighbors:
                # Skip self-comparisons.
                if neighbor.item_id == item.item_id:
                    continue

                pair = frozenset({item.item_id, neighbor.item_id})
                if pair in processed_pairs:
                    continue

                if not neighbor.embedding:
                    continue

                sim = MemoryItem.cosine_similarity(item.embedding, neighbor.embedding)
                if sim <= cfg.similarity_threshold:
                    continue

                # Apply interference penalty to both memories.
                penalty_factor = sim * cfg.interference_weight
                new_item_score = max(0.0, current_score * (1.0 - penalty_factor))
                neighbor_score = max(
                    0.0,
                    float(neighbor.metadata.get("retention_score", 1.0)) * (1.0 - penalty_factor),
                )
                neighbor_count = int(neighbor.metadata.get("interference_count", 0))

                try:
                    memory.update_properties(
                        item.item_id,
                        {"retention_score": new_item_score, "interference_count": interference_count + 1},
                    )
                    memory.update_properties(
                        neighbor.item_id,
                        {"retention_score": neighbor_score, "interference_count": neighbor_count + 1},
                    )
                except Exception as exc:
                    log.warning(
                        f"InterferenceBasedConsolidationEvolver: failed to update pair "
                        f"({item.item_id}, {neighbor.item_id}) — {exc}"
                    )

                processed_pairs.add(pair)
                # Update local score so subsequent neighbors see the already-penalised value.
                current_score = new_item_score
                interference_count += 1
                total_penalised += 1

        log.info(f"InterferenceBasedConsolidationEvolver: {total_penalised} interference penalties applied")
