"""Plausibility scorer for relation triples.

Computes a weighted average of normalization confidence and type-pair score.
No grounding input — see design doc "Pipeline Ordering Constraint" section.
"""

from __future__ import annotations


class PlausibilityScorer:
    """Compute plausibility score for validated relation triples."""

    def __init__(self, w_norm: float = 0.5, w_pair: float = 0.5):
        """
        Args:
            w_norm: Weight for normalization confidence.
            w_pair: Weight for type-pair score.
        """
        self._w_norm = w_norm
        self._w_pair = w_pair

    def score(self, normalization_confidence: float, type_pair_score: float) -> float:
        """Weighted average of available signals.

        Returns:
            Score clamped to [0.0, 1.0].
        """
        total_weight = self._w_norm + self._w_pair
        if total_weight == 0.0:
            return 0.0
        raw = (self._w_norm * normalization_confidence + self._w_pair * type_pair_score) / total_weight
        return max(0.0, min(1.0, raw))
