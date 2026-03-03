"""
Semantic memory using unified base classes and mixins.

Migrated from original 240+ line implementation to simplified 50-line version
while maintaining all functionality through unified patterns.
"""

import logging
from datetime import UTC
from typing import Optional, List

from smartmemory.configuration import MemoryConfig
from smartmemory.graph.types.semantic import SemanticMemoryGraph
from smartmemory.memory.base import HybridMemory
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.embedding import create_embeddings

logger = logging.getLogger(__name__)


class SemanticMemory(HybridMemory):
    """
    Semantic memory using unified patterns.
    
    Migrated from original 240+ line implementation to simplified version
    while maintaining all functionality through unified base classes and mixins.
    """

    def __init__(self, config: MemoryConfig = None, embedding_fn=None, *args, **kwargs):
        # Initialize with semantic memory type and store
        super().__init__(
            memory_type="semantic",
            config=config,
            *args, **kwargs
        )

        # Set up semantic-specific stages
        self.embedding_fn = embedding_fn or create_embeddings

        # Set up graph and store for proper delegation
        self.graph = SemanticMemoryGraph()
        # Set store to graph for unified base class delegation
        self.store = self.graph

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Semantic-specific add logic."""
        # Ensure embedding exists
        if not hasattr(item, 'embedding') or item.embedding is None:
            if self.embedding_fn:
                item.embedding = self.embedding_fn(str(item.content))

        # Use parent hybrid add (graph + vector)
        return super()._add_impl(item, **kwargs)

    def add_entities_relations(self, item: MemoryItem, entities: List, relations: List):
        """Persist extracted entities and relations."""
        extraction = {'item': item, 'entities': entities, 'relations': relations}
        if hasattr(self.graph, 'store_entities_relations'):
            self.graph.store_entities_relations(extraction, item)

    def get_low_relevance(self, threshold: float = 0.2) -> List[MemoryItem]:
        """Get facts with low relevance scores."""
        try:
            # Use search to find all items, then filter by relevance
            all_items = self._search_impl("*", top_k=1000)  # Get many items
            return [
                item for item in all_items
                if item.metadata.get('relevance', 1.0) < threshold
            ]
        except Exception as e:
            logger.error(f"Failed to get low relevance items: {e}")
            return []

    def search_by_embedding(self, embedding, top_k: int = 5) -> List[MemoryItem]:
        """Search by embedding similarity."""
        try:
            # Use vector store directly for embedding search
            results = self.vector_store.search(embedding, top_k=top_k)
            return [
                MemoryItem(
                    item_id=result.get('id', ''),
                    content=result.get('content', ''),
                    metadata=result.get('metadata') or {}
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed embedding search: {e}")
            return []

    def prune_by_relevance(
            self,
            embedding,
            threshold: float = 0.8,
            metadata_filter: Optional[dict] = None,
            adaptive: bool = False,
            percent: float = 0.0,
            soft_delete: bool = True,
            age_out: Optional[int] = None,
            llm_feedback: Optional[dict] = None,
    ) -> List[MemoryItem]:
        """
        Advanced prune with metadata filtering, adaptive thresholds, soft deletion, and LLM feedback.
        """
        try:
            # Use archiving mixin for safe pruning operations
            all_items = self._search_impl("*", top_k=1000)
            to_remove = []
            scores = []

            # Compute scores and filter
            for item in all_items:
                if metadata_filter and not all(item.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                if age_out:
                    valid_time = item.metadata.get("valid_time") or item.metadata.get("created_at")
                    if valid_time:
                        from datetime import datetime
                        if isinstance(valid_time, str):
                            valid_time = datetime.fromisoformat(valid_time)
                        if (datetime.now(UTC) - valid_time).days < age_out:
                            continue
                if llm_feedback and item.item_id in llm_feedback:
                    if llm_feedback[item.item_id] == "keep":
                        continue
                    elif llm_feedback[item.item_id] == "discard":
                        to_remove.append(item)
                        continue

                # Calculate similarity score
                item_emb = getattr(item, 'embedding', None)
                if item_emb is None and self.embedding_fn:
                    item_emb = self.embedding_fn(str(item.content))
                elif item_emb is None:
                    item_emb = MemoryItem.text_to_dummy_embedding(str(item.content))
                score = MemoryItem.cosine_similarity(embedding, item_emb)
                scores.append((item, score))

            # Adaptive threshold
            if adaptive and percent > 0 and scores:
                scores.sort(key=lambda x: x[1])
                cutoff = int(len(scores) * percent)
                to_remove.extend([item for item, _ in scores[:cutoff]])
            else:
                to_remove.extend([item for item, score in scores if score < threshold])

            # Soft delete or hard delete using mixin methods
            for item in to_remove:
                if soft_delete:
                    self.archive_item(item.item_id)  # Use archiving mixin
                else:
                    self.delete(item.item_id)

            return to_remove
        except Exception as e:
            logger.error(f"Failed to prune by relevance: {e}")
            return []
