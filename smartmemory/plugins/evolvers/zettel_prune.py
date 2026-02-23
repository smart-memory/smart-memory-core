from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class ZettelPruneConfig(MemoryBaseModel):
    """Configuration for zettel pruning."""
    min_content_length: int = 50  # Minimum content length to keep
    min_connections: int = 0  # Minimum connections to keep
    similarity_threshold: float = 0.9  # Similarity threshold for duplicates
    dry_run: bool = False  # If True, only log what would be pruned


@dataclass
class ZettelPruneRequest(StageRequest):
    """Request for zettel pruning operation."""
    min_content_length: int = 50
    min_connections: int = 0
    similarity_threshold: float = 0.9
    dry_run: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class ZettelPruneEvolver(EvolverPlugin):
    """
    Prunes/merges low-quality or duplicate zettels for graph health.
    
    Uses soft delete (archival) to maintain temporal history.
    Identifies candidates based on:
    - Content length (too short)
    - Connection count (orphaned)
    - Content similarity (duplicates)
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="zettel_prune",
            version="2.0.0",
            author="SmartMemory Team",
            description="Prunes and merges low-quality or duplicate zettels using soft delete",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["zettelkasten", "pruning", "deduplication", "soft-delete"]
        )

    def evolve(self, memory, logger=None):
        """Execute zettel pruning with soft delete."""
        # Require typed config
        cfg = getattr(self, "config")
        if not isinstance(cfg, ZettelPruneConfig):
            raise TypeError("ZettelPruneEvolver requires ZettelPruneConfig.")
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.zettel_prune", {"memory_id": memory_id, "dry_run": cfg.dry_run}):
            # Get candidates for pruning with config parameters
            candidates = memory.zettel.get_low_quality_or_duplicates(
                min_content_length=cfg.min_content_length,
                min_connections=cfg.min_connections,
                similarity_threshold=cfg.similarity_threshold
            )

            if not candidates:
                if logger:
                    logger.info("No low-quality or duplicate zettels found")
                return

            pruned_count = 0
            skipped_count = 0

            for zettel in candidates:
                try:
                    # Determine prune reason
                    content_len = len(zettel.content) if zettel.content else 0
                    connections = memory.zettel.get_bidirectional_connections(zettel.item_id)
                    total_connections = sum(len(conn_list) for conn_list in connections.values())

                    reasons = []
                    if content_len < cfg.min_content_length:
                        reasons.append(f"short_content_{content_len}_chars")
                    if total_connections < cfg.min_connections:
                        reasons.append(f"orphaned_{total_connections}_connections")

                    reason = "_and_".join(reasons) if reasons else "duplicate"

                    if cfg.dry_run:
                        if logger:
                            logger.info(f"[DRY RUN] Would prune zettel {zettel.item_id}: {reason}")
                        skipped_count += 1
                    else:
                        # Soft delete via archival
                        memory.zettel.prune_or_merge(zettel)
                        pruned_count += 1

                        if logger:
                            logger.info(f"Archived zettel {zettel.item_id}: {reason}")

                except Exception as e:
                    if logger:
                        logger.error(f"Failed to prune zettel {zettel.item_id}: {e}")
                    skipped_count += 1

            if logger:
                if cfg.dry_run:
                    logger.info(f"[DRY RUN] Would prune {skipped_count} zettels")
                else:
                    logger.info(f"Archived {pruned_count} zettels, skipped {skipped_count}")
