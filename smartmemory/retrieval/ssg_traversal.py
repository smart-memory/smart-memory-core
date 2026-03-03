"""
Similarity Graph Traversal (SSG) algorithms for enhanced semantic retrieval.

Implements:
- query_traversal: Always prioritizes query similarity (100% test pass, 0.91 precision/recall)
- triangulation_fulldim: Full-dimensional geometric centroids (highest precision)

These algorithms improve upon basic top-k vector search by:
1. Finding an "anchor" chunk most similar to the query
2. Traversing graph edges to related chunks
3. Scoring candidates using query + graph context
4. Early stopping when quality degrades

Reference:
    Eric Lester. (2025). Novel Semantic Similarity Graph Traversal Algorithms 
    for Semantic Retrieval Augmented Generation Systems.
    https://github.com/glacier-creative-git/semantic-similarity-graph-traversal-semantic-rag-research
"""

import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np

from smartmemory.models.memory_item import MemoryItem
from smartmemory.utils import get_config

logger = logging.getLogger(__name__)


class SimilarityGraphTraversal:
    """
    SSG-based retrieval using graph traversal algorithms.
    
    Provides superior multi-hop reasoning compared to basic vector search
    by leveraging both semantic similarity and graph structure.
    """
    
    def __init__(self, smart_memory):
        """
        Initialize SSG traversal with SmartMemory instance.

        Args:
            smart_memory: SmartMemory instance with graph and vector store
        """
        self.sm = smart_memory
        self._backend = getattr(smart_memory._graph, "backend", smart_memory._graph)

        # Prefer the VectorStore already configured on SmartMemory (respects DI /
        # ContextVar overrides in lite mode).  Fall back to a fresh instance only
        # when the SmartMemory object doesn't expose one.
        from smartmemory.stores.vector.vector_store import VectorStore
        self.vector_store = getattr(smart_memory, "_vector_store", None) or VectorStore()
        
        # Load configuration
        self.config = get_config('ssg') or {}
        self.default_max_results = self.config.get('max_results', 15)
        self.min_before_early_stop = self.config.get('min_results_before_early_stop', 8)
        
        neighbor_config = self.config.get('neighbor_candidates', {})
        self.max_graph_neighbors = neighbor_config.get('graph_neighbors', 10)
        self.max_vector_neighbors = neighbor_config.get('vector_neighbors', 10)
        self.max_total_neighbors = neighbor_config.get('max_total', 20)
        
        # Similarity cache for performance
        self.similarity_cache: Dict[str, float] = {}
        
    def query_traversal(
        self, 
        query: str, 
        max_results: Optional[int] = None
    ) -> List[MemoryItem]:
        """
        Query-guided graph traversal algorithm.
        
        Always prioritizes similarity to the original query during traversal.
        Best performer from research: 100% test pass rate, 0.91 precision/recall.
        
        Algorithm:
        1. Embed query and find anchor chunk (most similar)
        2. Extract sentences from anchor
        3. Get neighbors (graph + vector similarity)
        4. Select neighbor most similar to query
        5. Repeat until max_results or early stopping
        
        Early stopping: When best extracted item > best candidate
        
        Note: Tenant isolation is handled by the underlying graph's scope_provider.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default from config)
            
        Returns:
            List of MemoryItem objects ranked by relevance
        """
        max_results = max_results or self.default_max_results
        
        logger.info(f"Starting query_traversal for query: '{query[:50]}...'")
        
        # 1. Embed query
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            logger.warning("Failed to embed query, returning empty results")
            return []
        
        # 2. Find anchor chunk (most similar to query)
        anchor_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=1
        )
        
        if not anchor_results:
            logger.warning(f"No anchor found for query: {query}")
            return []
        
        anchor_id = anchor_results[0]['id']
        logger.info(f"Found anchor: {anchor_id}")
        
        # 3. Initialize traversal state
        extracted_items: List[MemoryItem] = []
        visited_chunks: Set[str] = set()
        current_chunk_id = anchor_id
        
        # 4. Traverse graph prioritizing query similarity
        iteration = 0
        while len(extracted_items) < max_results:
            iteration += 1
            logger.debug(f"Traversal iteration {iteration}, current chunk: {current_chunk_id}")
            
            # Get current chunk as MemoryItem
            current_item = self._get_memory_item(current_chunk_id)
            if current_item and current_chunk_id not in visited_chunks:
                extracted_items.append(current_item)
                visited_chunks.add(current_chunk_id)
                logger.debug(f"Extracted item {current_chunk_id}, total: {len(extracted_items)}")
            
            # Get neighbor candidates
            neighbors = self._get_neighbors(current_chunk_id, visited_chunks)
            if not neighbors:
                logger.info(f"No more neighbors, stopping at {len(extracted_items)} results")
                break
            
            logger.debug(f"Found {len(neighbors)} neighbor candidates")
            
            # Score neighbors by query similarity
            best_neighbor = self._select_best_by_query_similarity(
                neighbors, 
                query_embedding
            )
            
            if best_neighbor is None:
                logger.info("No valid neighbor found, stopping traversal")
                break
            
            # Early stopping: if best extracted > best candidate
            if self._should_stop_early(extracted_items, best_neighbor, query_embedding):
                logger.info(f"Early stopping triggered at {len(extracted_items)} results")
                break
            
            current_chunk_id = best_neighbor
        
        logger.info(f"Query traversal completed: {len(extracted_items)} results in {iteration} iterations")
        return extracted_items
    
    def triangulation_fulldim(
        self, 
        query: str, 
        max_results: Optional[int] = None
    ) -> List[MemoryItem]:
        """
        Full-dimensional geometric triangulation algorithm.
        
        Uses triangle centroids in full embedding space for highest precision.
        Best for queries requiring high factual accuracy.
        
        Algorithm:
        1. Find anchor chunk
        2. For each neighbor candidate, compute triangle centroid:
           centroid = (query_vec + current_vec + candidate_vec) / 3
        3. Select candidate with centroid closest to query
        4. Repeat until max_results
        
        Note: Tenant isolation is handled by the underlying graph's scope_provider.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of MemoryItem objects ranked by relevance
        """
        max_results = max_results or self.default_max_results
        
        logger.info(f"Starting triangulation_fulldim for query: '{query[:50]}...'")
        
        # 1. Embed query and find anchor
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return []
        
        # 2. Find anchor chunk
        anchor_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=1
        )
        
        if not anchor_results:
            logger.warning(f"No anchor found for triangulation query: {query}")
            return []
        
        anchor_id = anchor_results[0]['id']
        logger.info(f"Found anchor: {anchor_id}")
        
        # 2. Initialize traversal
        extracted_items: List[MemoryItem] = []
        visited_chunks: Set[str] = set()
        current_chunk_id = anchor_id
        
        iteration = 0
        while len(extracted_items) < max_results:
            iteration += 1
            
            # Extract current item
            current_item = self._get_memory_item(current_chunk_id)
            if current_item and current_chunk_id not in visited_chunks:
                extracted_items.append(current_item)
                visited_chunks.add(current_chunk_id)
            
            # Get neighbors
            neighbors = self._get_neighbors(current_chunk_id, visited_chunks)
            if not neighbors:
                logger.info(f"No more neighbors, stopping at {len(extracted_items)} results")
                break
            
            # Select neighbor with centroid closest to query
            best_neighbor = self._select_by_centroid_distance(
                current_chunk_id,
                neighbors,
                query_embedding
            )
            
            if best_neighbor is None:
                logger.info("No valid neighbor found, stopping traversal")
                break
            
            current_chunk_id = best_neighbor
        
        logger.info(f"Triangulation completed: {len(extracted_items)} results in {iteration} iterations")
        return extracted_items
    
    # ==================== Helper Methods ====================
    
    def _embed_query(self, query: str) -> Optional[List[float]]:
        """
        Embed query using existing embedding pipeline.
        
        Args:
            query: Query string to embed
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            from smartmemory.plugins.embedding import create_embeddings
            
            embedding = create_embeddings(query)
            if embedding is None:
                logger.warning("create_embeddings returned None")
                return None
            
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return None
    
    def _get_memory_item(self, item_id: str) -> Optional[MemoryItem]:
        """
        Retrieve MemoryItem from graph by ID via backend ABC method.

        Args:
            item_id: Item identifier

        Returns:
            MemoryItem object or None if not found
        """
        try:
            node = self._backend.get_node(item_id)
            if node is None:
                return None
            return MemoryItem(
                item_id=node.get("item_id", item_id),
                content=node.get("content", ""),
                metadata=node,
            )
        except Exception as e:
            logger.debug(f"Failed to get item {item_id}: {e}")
            return None
    
    def _get_neighbors(
        self,
        chunk_id: str,
        visited: Set[str],
    ) -> List[str]:
        """
        Get unvisited neighbors from both graph relationships and vector similarity.

        Combines:
        1. Graph-based neighbors via backend ABC ``get_neighbors()``
        2. Vector-based neighbors (semantic similarity)

        Note: Tenant isolation is handled by the underlying backend's scope_provider.

        Args:
            chunk_id: Current chunk ID
            visited: Set of already visited chunk IDs

        Returns:
            List of neighbor chunk IDs (limited by config)
        """
        neighbors = []

        # 1. Graph-based neighbors via backend ABC method.
        try:
            graph_neighbors = self._backend.get_neighbors(chunk_id)
            for neighbor in graph_neighbors[: self.max_graph_neighbors]:
                nid = neighbor.get("item_id") if isinstance(neighbor, dict) else getattr(neighbor, "item_id", None)
                if nid and nid not in visited:
                    neighbors.append(nid)
        except Exception as e:
            logger.debug(f"Failed to get graph neighbors for {chunk_id}: {e}")
        
        # 2. Vector-based neighbors (semantic similarity)
        try:
            chunk_data = self.vector_store.get(chunk_id)
            if chunk_data and 'embedding' in chunk_data:
                chunk_embedding = chunk_data['embedding']
                
                # Handle JSON-encoded embeddings
                if isinstance(chunk_embedding, str):
                    import json
                    chunk_embedding = json.loads(chunk_embedding)
                
                similar_chunks = self.vector_store.search(
                    query_embedding=chunk_embedding,
                    top_k=self.max_vector_neighbors
                )
                
                for result in similar_chunks:
                    neighbor_id = result.get('id')
                    if neighbor_id and neighbor_id not in visited and neighbor_id != chunk_id:
                        neighbors.append(neighbor_id)
        except Exception as e:
            logger.debug(f"Failed to get vector neighbors for {chunk_id}: {e}")
        
        # Remove duplicates and limit total
        unique_neighbors = list(dict.fromkeys(neighbors))  # Preserve order
        limited_neighbors = unique_neighbors[:self.max_total_neighbors]
        
        logger.debug(f"Found {len(limited_neighbors)} unique neighbors for {chunk_id}")
        return limited_neighbors
    
    def _select_best_by_query_similarity(
        self, 
        candidates: List[str], 
        query_embedding: List[float]
    ) -> Optional[str]:
        """
        Select candidate most similar to the query.
        
        Args:
            candidates: List of candidate chunk IDs
            query_embedding: Query embedding vector
            
        Returns:
            ID of best candidate, or None if none valid
        """
        best_id = None
        best_score = -1.0
        
        for candidate_id in candidates:
            try:
                # Get candidate embedding
                candidate_data = self.vector_store.get(candidate_id)
                if not candidate_data or 'embedding' not in candidate_data:
                    continue
                
                candidate_embedding = candidate_data['embedding']
                
                # Handle JSON-encoded embeddings
                if isinstance(candidate_embedding, str):
                    import json
                    candidate_embedding = json.loads(candidate_embedding)
                
                # Compute similarity
                cache_key = f"{candidate_id}:query"
                if cache_key in self.similarity_cache:
                    score = self.similarity_cache[cache_key]
                else:
                    score = self._cosine_similarity(query_embedding, candidate_embedding)
                    self.similarity_cache[cache_key] = score
                
                if score > best_score:
                    best_score = score
                    best_id = candidate_id
                    
            except Exception as e:
                logger.debug(f"Failed to score candidate {candidate_id}: {e}")
                continue
        
        if best_id:
            logger.debug(f"Best candidate: {best_id} with score {best_score:.4f}")
        
        return best_id
    
    def _should_stop_early(
        self, 
        extracted: List[MemoryItem], 
        next_candidate: str,
        query_embedding: List[float]
    ) -> bool:
        """
        Determine if early stopping should be triggered.
        
        Early stopping occurs when:
        - We have minimum results (config: min_results_before_early_stop)
        - Best extracted item score > next candidate score
        
        This prevents over-retrieval of less relevant items.
        
        Args:
            extracted: List of already extracted items
            next_candidate: ID of next candidate to consider
            query_embedding: Query embedding vector
            
        Returns:
            True if should stop, False otherwise
        """
        if len(extracted) < self.min_before_early_stop:
            return False
        
        # Get best score from extracted items
        best_extracted_score = -1.0
        for item in extracted:
            try:
                item_data = self.vector_store.get(item.item_id)
                if not item_data or 'embedding' not in item_data:
                    continue
                
                item_embedding = item_data['embedding']
                if isinstance(item_embedding, str):
                    import json
                    item_embedding = json.loads(item_embedding)
                
                score = self._cosine_similarity(query_embedding, item_embedding)
                best_extracted_score = max(best_extracted_score, score)
                
            except Exception:
                continue
        
        # Get candidate score
        try:
            candidate_data = self.vector_store.get(next_candidate)
            if candidate_data and 'embedding' in candidate_data:
                candidate_embedding = candidate_data['embedding']
                if isinstance(candidate_embedding, str):
                    import json
                    candidate_embedding = json.loads(candidate_embedding)
                
                candidate_score = self._cosine_similarity(query_embedding, candidate_embedding)
                
                # Stop if best extracted > candidate
                should_stop = best_extracted_score > candidate_score
                if should_stop:
                    logger.debug(
                        f"Early stop: best_extracted={best_extracted_score:.4f} > "
                        f"candidate={candidate_score:.4f}"
                    )
                return should_stop
                
        except Exception as e:
            logger.debug(f"Failed to check early stopping: {e}")
        
        return False
    
    def _select_by_centroid_distance(
        self,
        current_id: str,
        candidates: List[str],
        query_embedding: List[float]
    ) -> Optional[str]:
        """
        Select candidate with triangle centroid closest to query.
        
        For each candidate, computes:
        centroid = (query_vec + current_vec + candidate_vec) / 3
        distance = ||centroid - query_vec||
        
        Selects candidate with minimum distance.
        
        Args:
            current_id: Current chunk ID
            candidates: List of candidate chunk IDs
            query_embedding: Query embedding vector
            
        Returns:
            ID of best candidate, or None if none valid
        """
        # Get current chunk embedding
        current_data = self.vector_store.get(current_id)
        if not current_data or 'embedding' not in current_data:
            logger.debug(f"No embedding for current chunk {current_id}")
            return None
        
        current_embedding = current_data['embedding']
        if isinstance(current_embedding, str):
            import json
            current_embedding = json.loads(current_embedding)
        
        # Convert to numpy arrays
        query_arr = np.array(query_embedding, dtype=np.float32)
        current_arr = np.array(current_embedding, dtype=np.float32)
        
        best_id = None
        best_distance = float('inf')
        
        for candidate_id in candidates:
            try:
                candidate_data = self.vector_store.get(candidate_id)
                if not candidate_data or 'embedding' not in candidate_data:
                    continue
                
                candidate_embedding = candidate_data['embedding']
                if isinstance(candidate_embedding, str):
                    import json
                    candidate_embedding = json.loads(candidate_embedding)
                
                candidate_arr = np.array(candidate_embedding, dtype=np.float32)
                
                # Compute triangle centroid
                centroid = (query_arr + current_arr + candidate_arr) / 3.0
                
                # Distance from centroid to query
                distance = np.linalg.norm(centroid - query_arr)
                
                if distance < best_distance:
                    best_distance = distance
                    best_id = candidate_id
                    
            except Exception as e:
                logger.debug(f"Failed to compute centroid for {candidate_id}: {e}")
                continue
        
        if best_id:
            logger.debug(f"Best centroid candidate: {best_id} with distance {best_distance:.4f}")
        
        return best_id
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score [-1, 1]
        """
        if not a or not b or len(a) != len(b):
            return -1.0
        
        try:
            a_arr = np.array(a, dtype=np.float32)
            b_arr = np.array(b, dtype=np.float32)
            
            dot = np.dot(a_arr, b_arr)
            norm_a = np.linalg.norm(a_arr)
            norm_b = np.linalg.norm(b_arr)
            
            if norm_a == 0 or norm_b == 0:
                return -1.0
            
            return float(dot / (norm_a * norm_b))
            
        except Exception as e:
            logger.debug(f"Failed to compute cosine similarity: {e}")
            return -1.0
    
    def clear_cache(self):
        """Clear similarity cache."""
        self.similarity_cache.clear()
        logger.debug("Similarity cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.similarity_cache),
            "cache_enabled": True
        }
