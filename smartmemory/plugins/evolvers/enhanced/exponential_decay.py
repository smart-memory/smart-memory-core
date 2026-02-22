"""Exponential Decay Evolver — Ebbinghaus forgetting curve for episodic memories.

Models the forgetting curve as:

    retention = exp(-elapsed_days / stability)

Where *stability* is a per-memory parameter (stored in ``metadata["stability"]``)
that defaults to :attr:`ExponentialDecayConfig.default_stability`.  Memories whose
retention drops below :attr:`ExponentialDecayConfig.archive_threshold` are flagged
as archived.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

_logger = logging.getLogger(__name__)


@dataclass
class ExponentialDecayConfig(MemoryBaseModel):
    """Configuration for ExponentialDecayEvolver.

    Attributes:
        default_stability: Half-life in days (higher = slower forgetting).
            Used when a memory's ``metadata["stability"]`` is absent.
        archive_threshold: Retention score below which a memory is archived.
        max_memories: Maximum number of episodic memories to process per cycle.
    """

    default_stability: float = 30.0
    archive_threshold: float = 0.1
    max_memories: int = 500


class ExponentialDecayEvolver(EvolverPlugin):
    """Apply Ebbinghaus exponential decay to episodic memories.

    Computes a per-memory retention score using:

        retention = exp(-elapsed_days / stability)

    and persists the score in ``metadata["retention_score"]``.  Memories that fall
    below the archive threshold are additionally flagged with
    ``metadata["archived"] = True`` and ``metadata["archive_reason"] = "exponential_decay"``.

    Follows Pattern B: reads via ``memory.search()`` and writes via
    ``memory.update_properties()``.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="exponential_decay",
            version="1.0.0",
            author="SmartMemory Team",
            description="Ebbinghaus forgetting-curve decay for episodic memories",
            plugin_type="evolver",
            tags=["decay", "episodic", "retention", "forgetting-curve"],
        )

    def __init__(self, config: ExponentialDecayConfig | None = None) -> None:
        self.config = config or ExponentialDecayConfig()

    def evolve(self, memory: Any, logger: Any = None) -> None:
        """Apply exponential decay to all episodic memories.

        Args:
            memory: SmartMemory instance (workspace-scoped).
            logger: Optional logger override.
        """
        memory_id = getattr(memory, "item_id", None)
        with trace_span(
            "pipeline.evolve.exponential_decay",
            {"memory_id": memory_id, "stability": self.config.default_stability},
        ):
            self._evolve_impl(memory, logger or _logger)

    def _evolve_impl(self, memory: Any, log: Any) -> None:
        cfg = self.config
        now = datetime.now(timezone.utc)

        try:
            items = memory.search(query="*", memory_types=["episodic"], limit=cfg.max_memories)
        except Exception as exc:
            log.warning(f"ExponentialDecayEvolver: search failed — {exc}")
            return

        if not items:
            log.debug("ExponentialDecayEvolver: no episodic memories found")
            return

        log.info(f"ExponentialDecayEvolver: processing {len(items)} episodic memories")
        archived = 0

        for item in items:
            try:
                tx = item.transaction_time
                if tx is None:
                    tx = now
                elif tx.tzinfo is None:
                    tx = tx.replace(tzinfo=timezone.utc)

                elapsed_days = (now - tx).total_seconds() / 86400.0
                stability = float(item.metadata.get("stability", cfg.default_stability))
                # Guard against zero/negative stability to avoid math domain errors.
                if stability <= 0:
                    stability = cfg.default_stability

                retention = math.exp(-elapsed_days / stability)

                props: dict[str, Any] = {"retention_score": retention}
                if retention < cfg.archive_threshold:
                    props["archived"] = True
                    props["archive_reason"] = "exponential_decay"
                    archived += 1

                memory.update_properties(item.item_id, props)
            except Exception as exc:
                log.warning(f"ExponentialDecayEvolver: skipping item {getattr(item, 'item_id', '?')} — {exc}")

        log.info(f"ExponentialDecayEvolver: {archived}/{len(items)} memories archived")
