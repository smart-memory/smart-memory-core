"""Retrieval-Based Strengthening Evolver — CORE-EVO-ENH-2.

Adjusts ``metadata["retention_score"]`` based on retrieval history stored in
``metadata["retrieval__profile"]`` by the ``RetrievalFlushConsumer``.

Formula (all confirmed in design doc):

    rank_weight    = 1 / (1 + avg_search_rank)
    recency_weight = velocity_7d / (velocity_30d + 1e-6)
    boost          = retrieval_weight × log(1 + total_count) × rank_weight × recency_weight
    retention      = min(1.0, base_retention + boost)

For memories with zero retrievals in the lookback window, a gentle unretrieved
decay penalty is applied instead:

    retention *= (1 - unretrieved_decay_rate × elapsed_days_in_window)

This evolver is *additive* on top of ``ExponentialDecayEvolver``; it is designed
not to heavily punish unretrieved memories (rate=0.005/day by default).
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

_logger = logging.getLogger(__name__)

_EPSILON = 1e-6


@dataclass
class RetrievalBasedStrengtheningConfig(MemoryBaseModel):
    """Configuration for RetrievalBasedStrengtheningEvolver.

    Attributes:
        retrieval_weight: Scaling factor applied to the log-count term.
            Higher values mean retrievals matter more for retention.
        unretrieved_decay_rate: Per-day decay applied to memories with no
            retrievals in the lookback window. Kept gentle (0.005/day) because
            ExponentialDecayEvolver already handles primary forgetting.
        lookback_days: Window in days for the unretrieved decay penalty.
            Should match the RetrievalFlushConsumer's aggregation window.
        max_memories: Maximum number of memories processed per evolver cycle.
        memory_types: Memory types to process. Working memory is excluded
            (it has its own short lifecycle managed by WorkingToEpisodicEvolver).
    """

    retrieval_weight: float = 0.1
    unretrieved_decay_rate: float = 0.005
    lookback_days: int = 30
    max_memories: int = 500
    memory_types: list = field(default_factory=lambda: ["episodic", "semantic", "procedural"])


class RetrievalBasedStrengtheningEvolver(EvolverPlugin):
    """Strengthen frequently-retrieved memories; gently penalise unretrieved ones.

    Reads ``metadata["retrieval__profile"]`` (written by ``RetrievalFlushConsumer``)
    to compute a retrieval-based boost and applies it to ``retention_score``.

    Follows Pattern B: reads via ``memory.search()`` and writes via
    ``memory.update_properties()``.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="retrieval_based_strengthening",
            version="1.0.0",
            author="SmartMemory Team",
            description="Boost retention for frequently-retrieved memories; penalise unretrieved ones",
            plugin_type="evolver",
            tags=["retrieval", "retention", "strengthening", "forgetting-curve"],
        )

    def __init__(self, config: RetrievalBasedStrengtheningConfig | None = None) -> None:
        self.config = config or RetrievalBasedStrengtheningConfig()

    def evolve(self, memory: Any, logger: Any = None) -> None:
        """Apply retrieval-based retention adjustment to all configured memory types.

        Args:
            memory: SmartMemory instance (workspace-scoped).
            logger: Optional logger override.
        """
        memory_id = getattr(memory, "item_id", None)
        with trace_span(
            "pipeline.evolve.retrieval_based_strengthening",
            {"memory_id": memory_id, "lookback_days": self.config.lookback_days},
        ):
            self._evolve_impl(memory, logger or _logger)

    def _evolve_impl(self, memory: Any, log: Any) -> None:
        cfg = self.config

        try:
            items = memory.search(query="*", memory_types=cfg.memory_types, limit=cfg.max_memories)
        except Exception as exc:
            log.warning("RetrievalBasedStrengtheningEvolver: search failed — %s", exc)
            return

        if not items:
            log.debug("RetrievalBasedStrengtheningEvolver: no memories found")
            return

        log.info("RetrievalBasedStrengtheningEvolver: processing %d memories", len(items))
        boosted = 0
        penalised = 0

        for item in items:
            try:
                with trace_span(
                    "pipeline.evolve.retrieval_based_strengthening",
                    {"memory_id": getattr(item, "item_id", None)},
                ):
                    new_score = self._compute_score(item, cfg, log)
                    memory.update_properties(item.item_id, {"retention_score": new_score})

                    profile = self._deserialise_profile(item)
                    if profile and profile.get("total_count", 0) > 0:
                        boosted += 1
                    else:
                        penalised += 1

            except Exception as exc:
                log.warning(
                    "RetrievalBasedStrengtheningEvolver: skipping item %s — %s",
                    getattr(item, "item_id", "?"),
                    exc,
                )

        log.info(
            "RetrievalBasedStrengtheningEvolver: %d boosted, %d penalised",
            boosted,
            penalised,
        )

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        item: Any,
        cfg: RetrievalBasedStrengtheningConfig,
        log: Any,
    ) -> float:
        """Compute the new retention_score for a single memory item.

        Args:
            item: MemoryItem to evaluate.
            cfg: Evolver configuration.
            log: Logger instance.

        Returns:
            New retention_score clamped to [0.0, 1.0].
        """
        base_retention = float(item.metadata.get("retention_score", 1.0))
        profile = self._deserialise_profile(item)

        if not profile or profile.get("total_count", 0) == 0:
            # No retrieval history — apply gentle unretrieved decay
            penalty = cfg.unretrieved_decay_rate * cfg.lookback_days
            new_score = base_retention * max(0.0, 1.0 - penalty)
            return max(0.0, min(1.0, new_score))

        total_count: int = profile.get("total_count", 0)
        avg_search_rank: float = float(profile.get("avg_search_rank", 1.0))
        velocity_7d: float = float(profile.get("velocity_7d", 0.0))
        velocity_30d: float = float(profile.get("velocity_30d", 0.0))

        rank_weight = 1.0 / (1.0 + avg_search_rank)
        recency_weight = velocity_7d / (velocity_30d + _EPSILON)
        boost = cfg.retrieval_weight * math.log1p(total_count) * rank_weight * recency_weight

        new_score = min(1.0, base_retention + boost)
        return max(0.0, new_score)

    # ------------------------------------------------------------------
    # Deserialisation helpers
    # ------------------------------------------------------------------

    def _deserialise_profile(self, item: Any) -> dict | None:
        """Read and deserialise ``metadata["retrieval__profile"]``.

        FalkorDB stores flat scalar properties. When a dict was written via
        ``update_properties()``, ``_serialize_value()`` in ``falkordb.py``
        converts it to a JSON string automatically. This guard handles both
        the raw-dict case (in-memory/test paths) and the JSON-string case
        (production FalkorDB path).

        Args:
            item: MemoryItem whose ``metadata`` dict may contain the profile.

        Returns:
            Parsed dict or None if the key is absent or unparseable.
        """
        try:
            raw = item.metadata.get("retrieval__profile")
            if raw is None:
                return None
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
            return None
        except Exception:
            return None
