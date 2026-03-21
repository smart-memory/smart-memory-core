"""
Search component for SmartMemory system.
Provides unified search interface across all memory types using enhanced similarity framework.
"""

import logging
from collections.abc import Iterable
from typing import List, Optional, Dict, Any, cast

from smartmemory.models.memory_item import MemoryItem
from smartmemory.similarity.framework import EnhancedSimilarityFramework

logger = logging.getLogger(__name__)

# System-internal node types that should never appear in user-facing search results.
_SYSTEM_NODE_TYPES = frozenset({"Version"})


class Search:
    """Search component for SmartMemory system."""

    def __init__(self, graph):
        """Initialize search component with graph backend."""
        self.graph = graph
        self.similarity_framework = EnhancedSimilarityFramework()

    def search(self, query: str, top_k: int = 5, memory_type: Optional[str] = None, **kwargs) -> List[MemoryItem]:
        """
        Search for memory items using enhanced similarity framework.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            memory_type: Optional filter by memory type
            **kwargs: Additional search options (e.g. enable_hybrid=True)
            
        Returns:
            List of MemoryItems ranked by similarity
        """
        query_text = query or ""

        # Initialize cache variable
        cache = None

        # Try Redis cache first for performance improvement
        try:
            from smartmemory.utils.cache import get_cache
            cache = get_cache()

            # Check cache for existing search results
            cached_results = cache.get_search_results(query_text, top_k, memory_type)
            if cached_results is not None:
                logger.debug(f"Cache hit for search: {query_text[:50]}...")
                return cached_results

            logger.debug(f"Cache miss for search: {query_text[:50]}...")
        except Exception as cache_error:
            logger.warning(f"Redis cache unavailable for search: {cache_error}")
            cache = None
        # First try SmartGraph's built-in search which is optimized
        try:
            graph_search = getattr(self.graph, 'search', None)
            search_fn = None
            if callable(graph_search):
                search_fn = graph_search
            else:
                candidate = getattr(graph_search, 'search', None)
                if callable(candidate):
                    search_fn = candidate

            if search_fn:
                results = search_fn(query_text, top_k=top_k * 2, **kwargs)
                if results:
                    results = list(cast(Iterable[MemoryItem], results))
                else:
                    results = []

                # Exclude system-internal nodes (Version, etc.)
                if results:
                    results = [
                        item for item in results
                        if getattr(item, 'memory_type', None) not in _SYSTEM_NODE_TYPES
                    ]

                # Filter by memory type if specified
                if memory_type and results:
                    results = [item for item in results if getattr(item, 'memory_type', None) == memory_type]

                # Apply recency sort if requested (recall uses this)
                if results and kwargs.get("sort_by") == "recency":
                    results.sort(key=lambda item: getattr(item, "created_at", None) or "0000-00-00", reverse=True)

                # Return top_k results if we got matches
                if results:
                    return results[:top_k]
                # Empty results — fall through to manual search
        except Exception as e:
            # Log the error but continue with fallback
            logger.warning(f"SmartGraph search failed: {e}, falling back to manual search", exc_info=True)

        # Fallback: Get all items manually and apply similarity
        all_items = []
        try:
            if hasattr(self.graph, 'get_all_node_ids'):
                # Get all node IDs and retrieve each node
                node_ids = self.graph.get_all_node_ids()
                for node_id in node_ids:
                    try:
                        item = self.graph.get_node(node_id)
                        if item and isinstance(item, MemoryItem):
                            all_items.append(item)
                    except Exception:
                        # Skip nodes that can't be retrieved
                        continue
        except Exception:
            # If even fallback fails, return empty results
            return []

        # Exclude system-internal nodes (Version, etc.)
        all_items = [item for item in all_items if getattr(item, 'memory_type', None) not in _SYSTEM_NODE_TYPES]

        # Filter by memory type if specified
        if memory_type:
            all_items = [item for item in all_items if getattr(item, 'memory_type', None) == memory_type]

        # If no items found, return empty
        if not all_items:
            return []

        if kwargs.get("sort_by") == "recency":
            all_items.sort(key=lambda item: getattr(item, "created_at", None) or "0000-00-00", reverse=True)
            return all_items[:top_k]

        # Create query item for similarity comparison
        query_item = MemoryItem(
            content=query_text,
            memory_type="query",
            metadata={}
        )

        # Calculate similarities and rank results
        scored_items = []
        for item in all_items:
            try:
                similarity = self.similarity_framework.calculate_similarity(query_item, item)
                scored_items.append((similarity, item))
            except Exception as e:
                # Log similarity calculation errors - they may indicate data quality issues
                logger.warning(f"Similarity calculation failed for item {getattr(item, 'item_id', 'unknown')}: {e}")
                # Continue processing other items, but don't silently ignore the error
                continue

        # Filter by similarity threshold — never return irrelevant results
        min_threshold = 0.3  # Minimum relevance to be worth returning
        scored_items = [(score, item) for score, item in scored_items if score >= min_threshold]

        # Sort by similarity (descending) and return top_k
        scored_items.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored_items[:top_k]]

        # Cache the results for future use
        if cache is not None and results:
            try:
                cache.set_search_results(query_text, top_k, results, memory_type)
                logger.debug(f"Cached search results for: {query_text[:50]}...")
            except Exception as cache_error:
                logger.warning(f"Failed to cache search results: {cache_error}")

        return results

    def embeddings_search(self, embedding, top_k: int = 5) -> List[MemoryItem]:
        """
        Search for memory items most similar to the given embedding using the vector store.
        Provides proper abstraction for vector store operations.
        
        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            
        Returns:
            List of MemoryItems ranked by embedding similarity
        """
        try:
            from smartmemory.stores.vector.vector_store import VectorStore
            vector_store = VectorStore()
            hits = vector_store.search(embedding, top_k=top_k)
            results = []
            for hit in hits:
                # Use the graph to get the full MemoryItem
                item = self.graph.get_node(hit['id'])
                if item:
                    results.append(item)
            return results
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []

    def search_by_embedding(self, embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        """
        Search using embedding vector.
        
        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            
        Returns:
            List of MemoryItems ranked by embedding similarity
        """
        return self.embeddings_search(embedding, top_k=top_k)

    def search_by_metadata(self, metadata_filters: Dict[str, Any], top_k: int = 5) -> List[MemoryItem]:
        """
        Search by metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to match
            top_k: Maximum number of results to return
            
        Returns:
            List of MemoryItems matching metadata filters
        """
        # Get all items from graph
        all_items = []
        try:
            if hasattr(self.graph, 'get_all_items'):
                all_items = self.graph.get_all_items()
            elif hasattr(self.graph, 'nodes'):
                for node_id in self.graph.nodes():
                    item = self.graph.get_node(node_id)
                    if item and isinstance(item, MemoryItem):
                        all_items.append(item)
        except Exception:
            return []

        # Filter by metadata
        matching_items = []
        for item in all_items:
            if not hasattr(item, 'metadata') or not item.metadata:
                continue

            matches = True
            for key, value in metadata_filters.items():
                if key not in item.metadata or item.metadata[key] != value:
                    matches = False
                    break

            if matches:
                matching_items.append(item)

        return matching_items[:top_k]
