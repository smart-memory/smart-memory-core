import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, List

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin


@dataclass
class EnhancedWorkingToEpisodicConfig(MemoryBaseModel):
    base_threshold: int = 40


@dataclass
class EnhancedWorkingToEpisodicRequest(StageRequest):
    base_threshold: int = 40
    context: Dict[str, any] = field(default_factory=dict)
    run_id: Optional[str] = None


class EnhancedWorkingToEpisodicEvolver(EvolverPlugin):
    """
    Enhanced working memory consolidation with semantic clustering and adaptive thresholds.

    Improvements over basic version:
    - Adaptive capacity based on cognitive load
    - Semantic clustering before summarization
    - Temporal decay weighting (recent items prioritized)
    - Context-aware grouping
    """

    def evolve(self, memory, logger=None):
        memory_id = getattr(memory, "item_id", None)
        with trace_span("pipeline.evolve.enhanced_working_to_episodic", {"memory_id": memory_id}):
            self._evolve_impl(memory, logger)

    def _evolve_impl(self, memory, logger=None):
        cfg = getattr(self, "config")
        if not hasattr(cfg, "base_threshold"):
            raise TypeError(
                "EnhancedWorkingToEpisodicEvolver requires a typed config with 'base_threshold'. "
                "Provide EnhancedWorkingToEpisodicConfig or a compatible typed config."
            )
        base_threshold = int(getattr(cfg, "base_threshold"))

        # Get working memory items with temporal weights
        working_items = memory.working.get_buffer()
        if len(working_items) < base_threshold:
            return

        # Calculate adaptive threshold based on cognitive load
        adaptive_threshold = self._calculate_adaptive_threshold(working_items, base_threshold)

        if len(working_items) >= adaptive_threshold:
            # Group items by semantic similarity and temporal proximity
            clusters = self._semantic_temporal_clustering(working_items)

            # Create episodic summaries for each cluster
            for cluster in clusters:
                episodic_summary = self._create_enhanced_summary(cluster)
                memory.episodic.add(episodic_summary)

                if logger:
                    logger.info(f"Promoted {len(cluster)} working items to episodic cluster")

            # Clear processed items from working memory
            memory.working.clear_buffer()

    def _calculate_adaptive_threshold(self, items: List[MemoryItem], base_threshold: int) -> int:
        """Calculate adaptive threshold based on cognitive load indicators."""
        # Factors that increase cognitive load (lower threshold):
        # - High semantic diversity
        # - Recent high-frequency additions
        # - Complex content (long text, many entities)

        if not items:
            return base_threshold

        # Semantic diversity factor
        content_lengths = [len(item.content) for item in items]
        avg_length = sum(content_lengths) / len(content_lengths)
        complexity_factor = min(1.5, avg_length / 100)  # Longer content = higher load

        # Temporal clustering factor
        now = datetime.now(timezone.utc)
        recent_items = [item for item in items
                        if (now - item.created_at).total_seconds() < 3600]  # Last hour
        recency_factor = len(recent_items) / len(items)

        # Adaptive threshold: lower when high load
        load_factor = (complexity_factor + recency_factor) / 2
        adaptive_threshold = int(base_threshold * (1 - 0.3 * load_factor))

        return max(10, adaptive_threshold)  # Minimum threshold of 10

    def _semantic_temporal_clustering(self, items: List[MemoryItem]) -> List[List[MemoryItem]]:
        """Group items by semantic similarity and temporal proximity."""
        if not items:
            return []

        # Simple clustering based on content similarity and time
        clusters = []
        processed = set()

        for i, item in enumerate(items):
            if i in processed:
                continue

            cluster = [item]
            processed.add(i)

            # Find similar items
            for j, other_item in enumerate(items[i + 1:], i + 1):
                if j in processed:
                    continue

                # Check semantic similarity (simple word overlap for now)
                similarity = self._calculate_similarity(item, other_item)
                temporal_proximity = self._calculate_temporal_proximity(item, other_item)

                if similarity > 0.3 and temporal_proximity > 0.5:
                    cluster.append(other_item)
                    processed.add(j)

            clusters.append(cluster)

        return clusters

    def _calculate_similarity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate semantic similarity between two items."""
        # Simple word overlap similarity (can be enhanced with embeddings)
        words1 = set(item1.content.lower().split())
        words2 = set(item2.content.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_temporal_proximity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate temporal proximity between two items."""
        time_diff = abs((item1.created_at - item2.created_at).total_seconds())
        # Proximity decreases exponentially with time difference
        return math.exp(-time_diff / 3600)  # 1-hour half-life

    def _create_enhanced_summary(self, cluster: List[MemoryItem]) -> MemoryItem:
        """Create an enhanced episodic summary from a cluster of working memory items."""
        # Weight recent items more heavily
        now = datetime.now(timezone.utc)
        weighted_content = []

        for item in cluster:
            age_hours = (now - item.created_at).total_seconds() / 3600
            weight = math.exp(-age_hours / 24)  # 24-hour half-life

            # Add weighted representation
            if weight > 0.1:  # Only include if weight is significant
                weighted_content.append(f"[w={weight:.2f}] {item.content}")

        # Create summary with metadata
        summary_content = "\n".join(weighted_content)
        summary_metadata = {
            "summarized": True,
            "source_count": len(cluster),
            "consolidation_type": "working_to_episodic",
            "consolidation_time": now.isoformat(),
            "cluster_coherence": self._calculate_cluster_coherence(cluster)
        }

        return MemoryItem(
            content=summary_content,
            memory_type="episodic",
            metadata=summary_metadata
        )

    def _calculate_cluster_coherence(self, cluster: List[MemoryItem]) -> float:
        """Calculate how coherent/related the items in a cluster are."""
        if len(cluster) <= 1:
            return 1.0

        similarities = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                sim = self._calculate_similarity(cluster[i], cluster[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0
