from typing import Dict, Any, List, Set, Optional, Tuple

from smartmemory.models.memory_item import MemoryItem
from smartmemory.memory.pipeline.stages.clustering import logger
from smartmemory.stores.vector.vector_store import VectorStore


class GlobalClustering:
    """
    Pipeline stage that clusters entities across the entire graph and merges duplicates.
    This is typically run as a background maintenance task, not on every ingestion.
    """

    def __init__(self, memory_instance):
        self.memory = memory_instance
        self._vector_store = None  # Lazy: created on first use inside _di_context()

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    def run(self, use_semhash: bool = True, semhash_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Run clustering with automatic scope filtering via ScopeProvider.

        Filtering is determined by the ScopeProvider that was injected into SmartMemory
        at initialization. This ensures consistent tenant/workspace/user isolation.

        Args:
            use_semhash: Whether to use SemHash for fast pre-deduplication (default: True)
            semhash_threshold: SemHash similarity threshold (default: 0.95)

        Returns a dict with stats about merged entities:
        - merged_count: Number of entities merged
        - clusters_found: Number of clusters identified
        - total_entities: Total number of entities processed
        - semhash_deduped: Number of items deduplicated by SemHash (if enabled)
        """
        try:
            # Get isolation filters from ScopeProvider (single source of truth)
            filters = {}
            if hasattr(self.memory, 'scope_provider') and self.memory.scope_provider:
                filters = self.memory.scope_provider.get_isolation_filters()

            # 1. Fetch all items with embeddings using backend-agnostic approach
            ids = []
            embeddings = []
            names = []  # For SemHash deduplication
            id_to_name = {}

            # Use SmartMemory's graph abstraction instead of direct backend access
            all_items = self.memory.get_all_items_debug()

            for item_id in self._get_all_item_ids():
                item = self.memory.get(item_id)
                if not item:
                    continue

                # Extract name for SemHash
                name = self._get_item_name(item)
                if name:
                    names.append(name)
                    id_to_name[item_id] = name

                # Get embedding from item or vector store
                embedding = self._get_embedding(item_id, item)
                if embedding:
                    ids.append(item_id)
                    embeddings.append(embedding)

            if not ids:
                return {"merged_count": 0, "clusters_found": 0, "total_entities": 0, "semhash_deduped": 0, **filters}

            # 2. SemHash pre-deduplication (fast, deterministic)
            semhash_deduped = 0
            semhash_clusters = {}
            if use_semhash and names:
                semhash_clusters, semhash_deduped = self._run_semhash_dedup(
                    ids, id_to_name, semhash_threshold
                )
                logger.info(f"SemHash pre-dedup: {semhash_deduped} duplicates found")

            # 3. Cluster embeddings (for remaining items)
            # We use a simple greedy clustering approach leveraging the vector index
            clusters = self._find_clusters(ids, embeddings)

            # 4. Merge SemHash clusters first
            merged_count = 0
            for canonical_name, member_ids in semhash_clusters.items():
                if len(member_ids) < 2:
                    continue

                canonical_id = member_ids[0]  # First one is canonical
                source_ids = member_ids[1:]

                if not hasattr(self.memory._graph.backend, "merge_nodes"):
                    logger.warning("Graph backend %s has no merge_nodes; skipping cluster merge", type(self.memory._graph.backend).__name__)
                    break
                success = self.memory._graph.backend.merge_nodes(canonical_id, source_ids)
                if success:
                    merged_count += len(source_ids)
                    self.vector_store.delete(source_ids)

            # 5. Merge embedding-based clusters
            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                # Identify canonical entity (e.g., longest name or specific criteria)
                canonical_id = self._identify_canonical(cluster)
                source_ids = [pid for pid in cluster if pid != canonical_id]

                if not source_ids:
                    continue

                # Merge in graph
                if not hasattr(self.memory._graph.backend, "merge_nodes"):
                    logger.warning("Graph backend %s has no merge_nodes; skipping cluster merge", type(self.memory._graph.backend).__name__)
                    break
                success = self.memory._graph.backend.merge_nodes(canonical_id, source_ids)
                if success:
                    merged_count += len(source_ids)

                    # Also remove merged vectors from vector store
                    self.vector_store.delete(source_ids)

            return {
                "merged_count": merged_count,
                "clusters_found": len(clusters) + len(semhash_clusters),
                "total_entities": len(ids),
                "semhash_deduped": semhash_deduped
            }

        except Exception as e:
            logger.error(f"Global clustering failed: {e}")
            return {"error": str(e)}

    def _get_all_item_ids(self) -> List[str]:
        """Get all item IDs using backend-agnostic approach."""
        try:
            # Use graph's nodes interface
            if hasattr(self.memory._graph, 'nodes') and hasattr(self.memory._graph.nodes, 'nodes'):
                return list(self.memory._graph.nodes.nodes())
            # Fallback to get_all_items_debug
            debug_info = self.memory.get_all_items_debug()
            return [s.get('item_id') for s in debug_info.get('sample_items', [])]
        except Exception as e:
            logger.warning(f"Failed to get item IDs: {e}")
            return []

    def _get_embedding(self, item_id: str, item: Any) -> List[float]:
        """Get embedding for an item from item or vector store."""
        # Try item's embedding attribute first
        if hasattr(item, 'embedding') and item.embedding is not None:
            emb = item.embedding
            return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

        # Try vector store lookup
        try:
            result = self.vector_store.get(item_id)
            if result and 'embedding' in result:
                return result['embedding']
        except Exception:
            pass

        return None

    def _find_clusters(self, ids: List[str], embeddings: List[List[float]]) -> List[List[str]]:
        """
        Find clusters of similar entities.
        Returns a list of clusters, where each cluster is a list of item_ids.
        """
        clusters = []
        processed_indices: Set[int] = set()

        # Distance threshold for clustering (lower = more similar)
        # Most vector stores return distance scores where 0 = identical
        distance_threshold = 0.1

        for i, embedding in enumerate(embeddings):
            if i in processed_indices:
                continue

            current_id = ids[i]

            # Search for similar items using the vector store
            # We use the embedding directly
            results = self.vector_store.search(embedding, top_k=10, is_global=True)

            cluster = [current_id]
            processed_indices.add(i)

            for result in results:
                res_id = result['id']
                score = result.get('score', 1.0)

                # Check if result is the item itself
                if res_id == current_id:
                    continue

                # Find index of res_id in our local list to mark as processed
                try:
                    res_idx = ids.index(res_id)
                except ValueError:
                    continue

                if res_idx in processed_indices:
                    continue

                # Check similarity
                if score < distance_threshold: # Very close
                     cluster.append(res_id)
                     processed_indices.add(res_idx)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _identify_canonical(self, cluster_ids: List[str]) -> str:
        """
        Identify the canonical entity ID from a cluster.
        Strategy: Prefer the one with the longest name (most descriptive),
        or highest confidence if available.
        """
        from smartmemory.models.memory_item import MemoryItem

        # Fetch full nodes to check properties
        candidates = []
        for item_id in cluster_ids:
            node = self.memory.get(item_id)
            if node:
                candidates.append(node)

        if not candidates:
            return cluster_ids[0]

        # Sort by name length (descending) as a proxy for completeness
        # Also consider 'confidence' metadata if present
        def score_candidate(item):
            name = ""
            if isinstance(item, MemoryItem):
                name = item.metadata.get('name', item.content)
            elif isinstance(item, dict):
                name = item.get('name', item.get('content', ''))
            else:
                name = str(item)

            return len(name)

        candidates.sort(key=score_candidate, reverse=True)

        best = candidates[0]
        if isinstance(best, MemoryItem):
            return best.item_id
        elif isinstance(best, dict):
            return best.get('item_id')
        else:
            return cluster_ids[0]

    def _get_item_name(self, item: Any) -> Optional[str]:
        """Extract name from an item for SemHash deduplication."""
        if isinstance(item, MemoryItem):
            return item.metadata.get('name') or item.content
        elif isinstance(item, dict):
            return item.get('name') or item.get('content')
        elif hasattr(item, 'name'):
            return item.name
        elif hasattr(item, 'content'):
            return item.content
        return str(item) if item else None

    def _run_semhash_dedup(
        self,
        ids: List[str],
        id_to_name: Dict[str, str],
        threshold: float = 0.95
    ) -> Tuple[Dict[str, List[str]], int]:
        """
        Run SemHash deduplication on items.

        Returns:
            Tuple of (clusters dict mapping canonical_name -> list of item_ids, dedup_count)
        """
        from smartmemory.utils.deduplication import SemHashDeduplicator

        # Build name -> ids mapping (multiple items can have same name)
        name_to_ids: Dict[str, List[str]] = {}
        for item_id, name in id_to_name.items():
            if name:
                if name not in name_to_ids:
                    name_to_ids[name] = []
                name_to_ids[name].append(item_id)

        if not name_to_ids:
            return {}, 0

        # Run SemHash on unique names
        deduplicator = SemHashDeduplicator(similarity_threshold=threshold)
        unique_names = list(name_to_ids.keys())
        deduped_names, duplicates = deduplicator.deduplicate(unique_names)

        # Build clusters: canonical_name -> list of all item_ids
        clusters: Dict[str, List[str]] = {}
        dedup_count = 0

        for dup_name, canonical_name in duplicates.items():
            if canonical_name not in clusters:
                clusters[canonical_name] = list(name_to_ids.get(canonical_name, []))
            # Add all IDs from the duplicate name
            dup_ids = name_to_ids.get(dup_name, [])
            clusters[canonical_name].extend(dup_ids)
            dedup_count += len(dup_ids)

        return clusters, dedup_count
