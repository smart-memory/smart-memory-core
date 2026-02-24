from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class EpisodicToSemanticConfig(MemoryBaseModel):
    confidence: float = 0.9
    days: int = 3


@dataclass
class EpisodicToSemanticRequest(StageRequest):
    confidence: float = 0.9
    days: int = 3
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class EpisodicToSemanticEvolver(EvolverPlugin):
    """
    Promotes stable facts/events from episodic to semantic memory based on confidence/frequency.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="episodic_to_semantic",
            version="1.0.0",
            author="SmartMemory Team",
            description="Promotes stable episodic events to semantic memory",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        cfg = self.config
        if not (hasattr(cfg, "confidence") and hasattr(cfg, "days")):
            raise TypeError(
                "EpisodicToSemanticEvolver requires a typed config with 'confidence' and 'days' attributes. "
                "Provide EpisodicToSemanticConfig or a compatible typed config."
            )
        confidence = float(cfg.confidence)
        min_days = int(cfg.days)
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.episodic_to_semantic", {"memory_id": memory_id, "confidence": confidence, "min_days": min_days}):
            stable_events = memory.episodic.get_stable_events(confidence=confidence, min_days=min_days)
            for event in stable_events:
                memory.semantic.add(event)
                memory.episodic.archive(event)
                if logger:
                    logger.info(f"Promoted episodic event to semantic: {event}")
