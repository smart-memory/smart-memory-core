from typing import Optional, List, Dict

from smartmemory.memory.pipeline.stages.clustering import logger


class EmbeddingClusterer:
    """
    KMeans-based clustering for efficient LLM deduplication.

    approach:
    1. Generate embeddings for all entities
    2. Cluster using KMeans into groups of ~128 items
    3. Run LLM deduplication within each cluster (parallel)

    This is much more efficient than running LLM on all pairs.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        cluster_size: int = 128,
        max_clusters: Optional[int] = None
    ):
        """
        Initialize embedding clusterer.

        Args:
            embedding_model: SentenceTransformer model name
            cluster_size: Target size for each cluster
            max_clusters: Maximum number of clusters (None = auto)
        """
        self.embedding_model_name = embedding_model
        self.cluster_size = cluster_size
        self.max_clusters = max_clusters
        self._model = None

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        return self._model

    def cluster_items(
        self,
        items: List[str],
        item_type: str = "entity"
    ) -> List[List[str]]:
        """
        Cluster items using KMeans on embeddings.

        Args:
            items: List of strings to cluster
            item_type: Type of items ("entity" or "relation") for logging

        Returns:
            List of clusters, where each cluster is a list of items
        """
        if not items or len(items) < 2:
            return [items] if items else []

        try:
            from sklearn.cluster import KMeans
            from scipy.spatial.distance import cdist
            import numpy as np
        except ImportError:
            logger.warning("sklearn/scipy not available, returning single cluster")
            return [items]

        model = self._get_model()

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(items)} {item_type}s...")
        embeddings = model.encode(items, show_progress_bar=False)

        # Determine number of clusters
        n_samples = len(items)
        num_clusters = max(1, n_samples // self.cluster_size)
        if self.max_clusters:
            num_clusters = min(num_clusters, self.max_clusters)

        if num_clusters <= 1:
            return [items]

        logger.info(f"Clustering {n_samples} {item_type}s into {num_clusters} clusters...")

        # Run KMeans
        kmeans = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=1,
            max_iter=20,
            tol=0.0,
            algorithm="lloyd",
        )
        kmeans.fit(embeddings.astype(np.float32))
        centroids = kmeans.cluster_centers_

        # Assign items to clusters with size limit
        distances = cdist(embeddings, centroids)
        assignments = np.argsort(distances, axis=1)

        clusters: List[List[int]] = [[] for _ in range(num_clusters)]
        assigned = np.zeros(n_samples, dtype=bool)

        # Greedy assignment respecting cluster size
        for rank in range(num_clusters):
            for i in range(n_samples):
                if assigned[i]:
                    continue
                cluster_id = assignments[i, rank]
                if len(clusters[cluster_id]) < self.cluster_size:
                    clusters[cluster_id].append(i)
                    assigned[i] = True

        # Handle unassigned items
        unassigned = np.where(~assigned)[0]
        if len(unassigned) > 0:
            clusters.append(unassigned.tolist())
            logger.debug(f"Added {len(unassigned)} unassigned items as separate cluster")

        # Convert indices to items
        item_clusters = [[items[idx] for idx in cluster] for cluster in clusters if cluster]

        logger.info(f"Created {len(item_clusters)} clusters, sizes: {[len(c) for c in item_clusters[:5]]}...")

        return item_clusters

    def generate_embeddings(
        self,
        items: List[str]
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for items.

        Args:
            items: List of strings

        Returns:
            Dict mapping item -> embedding vector
        """
        model = self._get_model()
        embeddings = model.encode(items, show_progress_bar=False)
        return {item: emb.tolist() for item, emb in zip(items, embeddings, strict=False)}
