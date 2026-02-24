from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class EpisodicDecayConfig(MemoryBaseModel):
    half_life: int = 30  # days


@dataclass
class EpisodicDecayRequest(StageRequest):
    half_life: int = 30
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class EpisodicDecayEvolver(EvolverPlugin):
    """
    Archives or deletes stale episodic events based on age or relevance.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="episodic_decay",
            version="1.0.0",
            author="SmartMemory Team",
            description="Archives stale episodic events based on age",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        cfg = self.config
        if not hasattr(cfg, "half_life"):
            raise TypeError(
                "EpisodicDecayEvolver requires a typed config with 'half_life'. "
                "Provide EpisodicDecayConfig or a compatible typed config."
            )
        half_life = int(cfg.half_life)
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.episodic_decay", {"memory_id": memory_id, "half_life": half_life}):
            stale_events = memory.episodic.get_stale_events(half_life=half_life)
            for event in stale_events:
                memory.episodic.archive(event)
                if logger:
                    logger.info(f"Archived stale episodic event: {event}")
