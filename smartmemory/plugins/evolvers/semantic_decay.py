from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class SemanticDecayConfig(MemoryBaseModel):
    threshold: float = 0.2


@dataclass
class SemanticDecayRequest(StageRequest):
    threshold: float = 0.2
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class SemanticDecayEvolver(EvolverPlugin):
    """
    Prunes/archives semantic facts based on low relevance, age, or feedback.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="semantic_decay",
            version="1.0.0",
            author="SmartMemory Team",
            description="Archives low-relevance semantic facts",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        cfg = getattr(self, "config")
        if not hasattr(cfg, "threshold"):
            raise TypeError(
                "SemanticDecayEvolver requires a typed config with 'threshold'. "
                "Provide SemanticDecayConfig or a compatible typed config."
            )
        threshold = float(getattr(cfg, "threshold"))
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.semantic_decay", {"memory_id": memory_id, "threshold": threshold}):
            old_facts = memory.semantic.get_low_relevance(threshold=threshold)
            for fact in old_facts:
                memory.semantic.archive(fact)
                if logger:
                    logger.info(f"Archived low-relevance semantic fact: {fact}")
