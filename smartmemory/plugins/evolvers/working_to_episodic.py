from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class WorkingToEpisodicConfig(MemoryBaseModel):
    """Typed config for WorkingToEpisodic evolver."""
    threshold: int = 40


@dataclass
class WorkingToEpisodicRequest(StageRequest):
    """Typed request DTO for WorkingToEpisodic evolver execution."""
    threshold: int = 40
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class WorkingToEpisodicEvolver(EvolverPlugin):
    """
    Evolves (summarizes) working memory buffer to episodic memory when overflowed (N turns).
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="working_to_episodic",
            version="1.0.0",
            author="SmartMemory Team",
            description="Promotes working memory to episodic when buffer threshold exceeded",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def evolve(self, memory, logger=None):
        # Example logic: summarize working memory if buffer exceeds threshold
        # Support both legacy dict config and typed config
        threshold = 40
        cfg = getattr(self, "config") or {}
        # Require typed config (fail-fast). No legacy dict support.
        if hasattr(cfg, "threshold"):
            threshold = int(getattr(cfg, "threshold", 40))
        else:
            raise TypeError(
                "WorkingToEpisodicEvolver requires a typed config with a 'threshold' attribute. "
                "Please provide WorkingToEpisodicConfig or a compatible typed config."
            )
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.working_to_episodic", {"memory_id": memory_id, "threshold": threshold}):
            working_items = memory.working.get_buffer()
            if len(working_items) >= threshold:
                summary = memory.working.summarize_buffer()
                memory.episodic.add(summary)

                # Archive working items with reference to the episodic summary
                memory.working.clear_buffer(archive_reason=f"promoted_to_episodic:{summary.item_id if hasattr(summary, 'item_id') else 'unknown'}")

                if logger:
                    logger.info(f"Promoted {len(working_items)} working items to episodic as summary (archived originals).")
