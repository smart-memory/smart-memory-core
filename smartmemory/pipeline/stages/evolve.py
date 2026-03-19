"""Evolve stage — wraps EvolutionOrchestrator.run_evolution_cycle()."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class EvolveStage:
    """Run evolution and clustering after ingestion."""

    def __init__(self, memory):
        """Args: memory — a SmartMemory instance (has _evolution, _clustering)."""
        self._memory = memory

    @property
    def name(self) -> str:
        return "evolve"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.mode == "preview":
            return state

        results = {}

        if config.evolve.run_evolution:
            # CORE-EVO-LIVE-1: skip batch evolution when incremental worker is active
            worker = getattr(self._memory, "_evolution_worker", None)
            if worker is not None and getattr(worker, "is_active", False):
                results["evolution"] = "skipped (incremental worker active)"
                logger.debug("EvolveStage: skipping batch evolution — incremental worker is active")
            else:
                try:
                    self._memory._evolution.run_evolution_cycle()
                    results["evolution"] = "completed"
                except Exception as e:
                    logger.warning("Evolution cycle failed (non-fatal): %s", e)
                    results["evolution"] = f"failed: {e}"

        if config.evolve.run_clustering:
            try:
                self._memory._clustering.run(use_semhash=True)
                results["clustering"] = "completed"
            except Exception as e:
                logger.debug("Clustering failed (non-fatal): %s", e)
                results["clustering"] = f"failed: {e}"

        return replace(state, evolutions=results)

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, evolutions={})
