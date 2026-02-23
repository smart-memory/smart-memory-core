from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class EpisodicToZettelConfig(MemoryBaseModel):
    period: int = 1  # days


@dataclass
class EpisodicToZettelRequest(StageRequest):
    period: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class EpisodicToZettelEvolver(EvolverPlugin):
    """
    Rolls up episodic events into zettels (notes) on a periodic basis.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="episodic_to_zettel",
            version="1.0.0",
            author="SmartMemory Team",
            description="Rolls up episodic events into zettel notes",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        cfg = getattr(self, "config")
        if not hasattr(cfg, "period"):
            raise TypeError(
                "EpisodicToZettelEvolver requires a typed config with 'period'. "
                "Provide EpisodicToZettelConfig or a compatible typed config."
            )
        period = int(getattr(cfg, "period"))
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.episodic_to_zettel", {"memory_id": memory_id, "period_days": period}):
            events = memory.episodic.get_events_since(days=period)
            for event in events:
                zettel = memory.zettel.create_note_from_event(event)
                memory.zettel.add(zettel)
                if logger:
                    logger.info(f"Rolled up episodic event into zettel: {event}")
