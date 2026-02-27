"""Classify stage — wraps MemoryIngestionFlow.classify_item()."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

# Deterministic priority order for memory_type selection from a classify result set.
# classify_item() returns a set-derived list with non-deterministic ordering; picking
# [0] from that list is hash-seed-dependent. This tuple defines the canonical order:
# specific types (decision, reasoning) override generic defaults (semantic).
# "zettel" is excluded separately before this lookup (it is a processing route, not
# a storage type). Any type not listed here falls to the end of priority.
_TYPE_PRIORITY: tuple = ("decision", "reasoning", "episodic", "procedural", "semantic")


class ClassifyStage:
    """Determine memory types for the incoming text."""

    def __init__(self, ingestion_flow):
        """Args: ingestion_flow — a MemoryIngestionFlow instance (has classify_item)."""
        self._flow = ingestion_flow

    @property
    def name(self) -> str:
        return "classify"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        from smartmemory.models.memory_item import MemoryItem
        from smartmemory.memory.pipeline.config import ClassificationConfig

        # Build a MemoryItem so classify_item can inspect metadata
        item = MemoryItem(
            content=state.text,
            memory_type=state.memory_type or "semantic",
            metadata=dict(state.raw_metadata),
        )

        # Map v2 config → legacy ClassificationConfig
        legacy_conf = ClassificationConfig(
            content_analysis_enabled=config.classify.content_analysis_enabled,
            default_confidence=config.classify.default_confidence,
            inferred_confidence=config.classify.inferred_confidence,
        )

        types = self._flow.classify_item(item, legacy_conf)

        # Determine memory_type for storage.  classified_types is a routing set
        # (e.g. ["semantic", "zettel"]) — "zettel" is a universal processing route,
        # not a meaningful storage label, so exclude it when picking memory_type.
        if state.memory_type:
            mt = state.memory_type
        else:
            # Use deterministic priority rather than set iteration order.
            # classify_item() returns list(set(...)) which is hash-seed-dependent;
            # picking [0] is nondeterministic across processes.
            non_zettel = {t for t in types if t != "zettel"}
            mt = next(
                (t for t in _TYPE_PRIORITY if t in non_zettel),
                next(iter(non_zettel), types[0] if types else "semantic"),
            )

        return replace(
            state,
            classified_types=types,
            memory_type=mt,
        )

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, classified_types=[], memory_type=None)
