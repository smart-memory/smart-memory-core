"""Relation normalizer — maps free-text LLM predicates to canonical types.

Three-step cascade:
1. Exact alias lookup (O(1) dict hit)
2. Embedding similarity (cosine against canonical embeddings, threshold 0.75)
3. Fallback to "related_to" with confidence 0.0
"""

from __future__ import annotations

import math
import re
from typing import Callable

from smartmemory.relations.schema import ALIAS_INDEX, CANONICAL_RELATION_TYPES


class RelationNormalizer:
    """Normalize free-text predicates to canonical relation types."""

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]] | None = None,
        workspace_aliases: dict[str, str] | None = None,
    ):
        """
        Args:
            embedding_fn: Optional function that maps text → embedding vector.
                If None, Step 2 (embedding similarity) is skipped.
            workspace_aliases: Optional workspace-scoped alias overrides.
                Merged on top of global ALIAS_INDEX (workspace wins on collision).
        """
        self._alias_index = dict(ALIAS_INDEX)  # instance copy — never mutate module-level
        if workspace_aliases:
            self._alias_index.update(workspace_aliases)
        self._embedding_fn = embedding_fn
        self._canonical_embeddings: dict[str, list[float]] | None = None

    def normalize(self, raw_predicate: str) -> tuple[str, float]:
        """Normalize a free-text predicate to a canonical type.

        Returns:
            (canonical_type, normalization_confidence)
            confidence: 1.0 for alias hit, cosine score for embedding, 0.0 for fallback
        """
        if not raw_predicate or not raw_predicate.strip():
            return ("related_to", 0.0)

        # Step 1: Alias lookup
        key = _normalize_key(raw_predicate)
        if not key:
            return ("related_to", 0.0)

        canonical = self._alias_index.get(key)
        if canonical:
            return (canonical, 1.0)

        # Step 2: Embedding similarity (if available)
        if self._embedding_fn is not None:
            result = self._embedding_match(key)
            if result is not None:
                return result

        # Step 3: Fallback
        return ("related_to", 0.0)

    def _embedding_match(self, normalized_key: str) -> tuple[str, float] | None:
        """Find the closest canonical type by embedding similarity.

        Returns (canonical_type, cosine_score) if above 0.75 threshold, else None.
        """
        if self._canonical_embeddings is None:
            self._canonical_embeddings = self._compute_canonical_embeddings()

        try:
            query_emb = self._embedding_fn(normalized_key.replace("_", " "))  # type: ignore[misc]
        except Exception:
            return None

        best_type = None
        best_score = -1.0

        for canonical_name, emb in self._canonical_embeddings.items():
            score = _cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_type = canonical_name

        if best_type is not None and best_score >= 0.75:
            return (best_type, best_score)
        return None

    def _compute_canonical_embeddings(self) -> dict[str, list[float]]:
        """Compute embeddings for each canonical type name."""
        embeddings: dict[str, list[float]] = {}
        for name in CANONICAL_RELATION_TYPES:
            try:
                emb = self._embedding_fn(name.replace("_", " "))  # type: ignore[misc]
                embeddings[name] = emb
            except Exception:
                continue
        return embeddings


def _normalize_key(predicate: str) -> str:
    """Normalize a predicate string for alias lookup.

    Same logic as _normalize_predicate() in llm_single.py:461-477:
    lowercase, replace non-alphanumeric with _, collapse repeated _, strip.
    """
    if not predicate:
        return ""
    key = predicate.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key)
    key = key.strip("_")
    return key


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
