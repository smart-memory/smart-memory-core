"""Relation type discovery service — clusters novel labels and promotes candidates.

Implements the batch analysis loop for CORE-EXT-1c:
1. Fetch novel relation labels above frequency threshold from the ontology graph
2. Cluster by embedding similarity (or string equality when no embedding_fn)
3. Filter clusters into promotion candidates
4. Auto-promote unambiguous candidates back into the normalizer vocabulary

Designed to be called periodically (e.g. by a background job or admin endpoint),
NOT inline during ingestion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from smartmemory.relations.normalizer import _normalize_key

logger = logging.getLogger(__name__)

RELOAD_CHANNEL_PREFIX = "smartmemory:relation_types:reload"


@dataclass
class RelationCluster:
    """A cluster of novel relation labels that appear to describe the same predicate."""

    centroid_label: str
    members: list[str] = field(default_factory=list)
    total_frequency: int = 0
    inferred_type_pairs: list[tuple[str, str]] = field(default_factory=list)
    embedding_coherence: float = 0.0


@dataclass
class RelationCandidate:
    """A candidate relation type proposed for promotion into the normalizer vocabulary."""

    proposed_name: str
    category: str = "custom"
    aliases: list[str] = field(default_factory=list)
    type_pairs: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    total_frequency: int = 0
    status: str = "proposed"  # "proposed" | "promoted" | "rejected"


class RelationDiscoveryService:
    """Batch discovery service for novel relation labels.

    Args:
        ontology: An OntologyGraph instance for reading/writing relation types.
        embedding_fn: Optional callable that maps a string to an embedding vector.
            When provided, clusters use cosine similarity; when None, falls back
            to normalized-key string equality.
        redis_client: Optional Redis client for publishing reload notifications
            after auto-promotion.
    """

    def __init__(
        self,
        ontology: Any,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        redis_client: Optional[Any] = None,
    ):
        self._ontology = ontology
        self._embedding_fn = embedding_fn
        self._redis = redis_client

    def cluster_novel_labels(
        self,
        workspace_id: str = "",
        min_frequency: int = 3,
    ) -> list[RelationCluster]:
        """Fetch novel labels above threshold and cluster by similarity.

        When no embedding_fn is available, clusters labels by _normalize_key
        string equality (exact match after lowercasing + underscore normalization).
        When embedding_fn is provided, groups labels whose embedding cosine
        similarity exceeds the threshold (default 0.75).

        Args:
            workspace_id: Scope for the ontology graph query.
            min_frequency: Minimum observation frequency to include a label.

        Returns:
            List of RelationCluster, one per distinct predicate group.
        """
        labels = self._ontology.get_novel_relation_labels(
            min_frequency=min_frequency,
            status="tracking",
        )
        if not labels:
            return []

        if self._embedding_fn is not None:
            return self._cluster_by_embedding(labels)
        return self._cluster_by_key(labels)

    def _cluster_by_key(self, labels: list[dict]) -> list[RelationCluster]:
        """Group labels by normalized key equality."""
        groups: dict[str, list[dict]] = {}
        for label in labels:
            key = _normalize_key(label["name"]) or label["name"]
            groups.setdefault(key, []).append(label)

        clusters = []
        for key, members in groups.items():
            total_freq = sum(m.get("frequency", 1) for m in members)
            type_pairs: list[tuple[str, str]] = []
            for m in members:
                for src in m.get("source_types", []):
                    for tgt in m.get("target_types", []):
                        pair = (src, tgt)
                        if pair not in type_pairs:
                            type_pairs.append(pair)
            clusters.append(
                RelationCluster(
                    centroid_label=key,
                    members=[m["name"] for m in members],
                    total_frequency=total_freq,
                    inferred_type_pairs=type_pairs,
                    embedding_coherence=1.0,  # exact match = perfect coherence
                )
            )
        return clusters

    def _cluster_by_embedding(self, labels: list[dict]) -> list[RelationCluster]:
        """Group labels by embedding cosine similarity (>=0.75)."""
        assert self._embedding_fn is not None

        # Compute embeddings for all labels
        label_embeddings: list[tuple[dict, list[float]]] = []
        for label in labels:
            try:
                emb = self._embedding_fn(label["name"])
                label_embeddings.append((label, emb))
            except Exception:
                logger.debug("Failed to embed label '%s', skipping", label["name"])

        if not label_embeddings:
            return self._cluster_by_key(labels)

        # Simple greedy clustering: assign each label to first cluster above threshold
        clusters: list[tuple[list[float], list[dict]]] = []  # (centroid_emb, members)
        threshold = 0.75

        for label, emb in label_embeddings:
            assigned = False
            for centroid_emb, members in clusters:
                sim = _cosine_similarity(centroid_emb, emb)
                if sim >= threshold:
                    members.append(label)
                    assigned = True
                    break
            if not assigned:
                clusters.append((emb, [label]))

        result = []
        for _centroid_emb, members in clusters:
            key = _normalize_key(members[0]["name"]) or members[0]["name"]
            total_freq = sum(m.get("frequency", 1) for m in members)
            type_pairs: list[tuple[str, str]] = []
            for m in members:
                for src in m.get("source_types", []):
                    for tgt in m.get("target_types", []):
                        pair = (src, tgt)
                        if pair not in type_pairs:
                            type_pairs.append(pair)

            # Compute average pairwise coherence
            coherence = 1.0
            if len(members) > 1:
                cluster_embs = []
                for m in members:
                    for lbl, e in label_embeddings:
                        if lbl is m:
                            cluster_embs.append(e)
                            break
                if len(cluster_embs) >= 2:
                    sims = []
                    for i in range(len(cluster_embs)):
                        for j in range(i + 1, len(cluster_embs)):
                            sims.append(_cosine_similarity(cluster_embs[i], cluster_embs[j]))
                    coherence = sum(sims) / len(sims) if sims else 1.0

            result.append(
                RelationCluster(
                    centroid_label=key,
                    members=[m["name"] for m in members],
                    total_frequency=total_freq,
                    inferred_type_pairs=type_pairs,
                    embedding_coherence=coherence,
                )
            )
        return result

    def propose_candidates(
        self,
        clusters: list[RelationCluster],
        min_cluster_frequency: int = 5,
    ) -> list[RelationCandidate]:
        """Filter clusters into promotion candidates.

        Only clusters with total_frequency >= min_cluster_frequency qualify.

        Args:
            clusters: Output of cluster_novel_labels().
            min_cluster_frequency: Minimum total frequency across cluster members.

        Returns:
            List of RelationCandidate ready for review or auto-promotion.
        """
        candidates = []
        for cluster in clusters:
            if cluster.total_frequency < min_cluster_frequency:
                continue
            aliases = [m for m in cluster.members if m != cluster.centroid_label]
            candidates.append(
                RelationCandidate(
                    proposed_name=cluster.centroid_label,
                    category="custom",
                    aliases=aliases,
                    type_pairs=cluster.inferred_type_pairs,
                    confidence=min(cluster.embedding_coherence, 1.0),
                    total_frequency=cluster.total_frequency,
                    status="proposed",
                )
            )
        return candidates

    def auto_promote(
        self,
        candidates: list[RelationCandidate],
        workspace_id: str = "",
        normalizer_reload_fn: Optional[Callable[[], None]] = None,
        min_frequency: int = 10,
    ) -> int:
        """Auto-promote unambiguous candidates into the normalizer vocabulary.

        Candidates whose source cluster had total_frequency >= min_frequency and
        confidence >= 0.5 are promoted via ontology.promote_relation_type().

        After any promotions:
        1. If normalizer_reload_fn provided, call it (in-process reload)
        2. If redis_client provided, publish reload notification

        Args:
            candidates: Output of propose_candidates().
            workspace_id: Used for Redis channel scoping.
            normalizer_reload_fn: Optional callback to reload normalizer in-process.
            min_frequency: Minimum frequency for auto-promotion (higher bar than propose).

        Returns:
            Number of candidates promoted.
        """
        promoted_count = 0
        for candidate in candidates:
            if candidate.status != "proposed":
                continue
            if candidate.confidence < 0.5:
                continue
            if candidate.total_frequency < min_frequency:
                continue
            try:
                self._ontology.promote_relation_type(
                    name=candidate.proposed_name,
                    category=candidate.category,
                    aliases=candidate.aliases,
                    type_pairs=candidate.type_pairs,
                )
                candidate.status = "promoted"
                promoted_count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to promote relation type '%s': %s",
                    candidate.proposed_name,
                    exc,
                )

        if promoted_count > 0:
            if normalizer_reload_fn is not None:
                try:
                    normalizer_reload_fn()
                except Exception as exc:
                    logger.warning("Normalizer reload callback failed: %s", exc)

            if self._redis is not None:
                try:
                    channel = f"{RELOAD_CHANNEL_PREFIX}:{workspace_id}"
                    self._redis.publish(channel, "reload")
                    logger.debug("Published relation type reload on %s", channel)
                except Exception as exc:
                    logger.warning("Failed to publish relation type reload: %s", exc)

        return promoted_count


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
