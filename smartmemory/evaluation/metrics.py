"""Pure metric functions for evaluation results."""

from __future__ import annotations

from datetime import timedelta

from smartmemory.evaluation.dataset import InteractionLog
from smartmemory.evaluation.judge import SessionJudgment


def relevance_at_k(judgments: list[SessionJudgment]) -> float:
    """Mean judge score across all queries.

    The judge scores the full result set (already top_k from search), so this
    returns the mean of all per-query scores. No per-result granularity.

    Returns 0.0 if no scores exist.
    """
    all_scores = [s.score for j in judgments for s in j.scores]
    if not all_scores:
        return 0.0
    return sum(all_scores) / len(all_scores)


def hit_rate(judgments: list[SessionJudgment], threshold: int = 2) -> float:
    """Fraction of queries where at least one result scored >= threshold.

    Args:
        judgments: List of session judgments.
        threshold: Minimum score to count as a "hit" (default 2).

    Returns:
        Float between 0.0 and 1.0.
    """
    all_scores = [s.score for j in judgments for s in j.scores]
    if not all_scores:
        return 0.0
    hits = sum(1 for s in all_scores if s >= threshold)
    return hits / len(all_scores)


def repetition_rate(
    interactions: list[InteractionLog],
    similarity_threshold: float = 0.85,
    session_gap_minutes: int = 30,
) -> float:
    """Fraction of queries that are near-duplicates of an earlier query in a different session.

    Uses simple token overlap (Jaccard similarity) as a lightweight proxy.
    For more accurate results, use embedding-based similarity.

    Args:
        interactions: All interactions to analyze.
        similarity_threshold: Cosine/Jaccard threshold for "near-duplicate".
        session_gap_minutes: Minimum gap to consider queries as cross-session.

    Returns:
        Float between 0.0 and 1.0.
    """
    if len(interactions) < 2:
        return 0.0

    gap = timedelta(minutes=session_gap_minutes)
    repeated = 0
    total = len(interactions)

    for i, current in enumerate(interactions):
        for j in range(i):
            earlier = interactions[j]
            # Must be in different sessions
            if (current.ts - earlier.ts) < gap:
                continue
            # Jaccard similarity on word tokens
            words_a = set(current.query.lower().split())
            words_b = set(earlier.query.lower().split())
            if not words_a or not words_b:
                continue
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            similarity = intersection / union if union > 0 else 0.0
            if similarity >= similarity_threshold:
                repeated += 1
                break  # Count each query at most once

    return repeated / total


def mean_latency(interactions: list[InteractionLog]) -> float:
    """Mean search latency in milliseconds.

    Returns 0.0 if no interactions.
    """
    if not interactions:
        return 0.0
    return sum(i.latency_ms for i in interactions) / len(interactions)


def redundancy_rate(judgments: list[SessionJudgment]) -> float:
    """Fraction of queries where the judge flagged redundant results.

    Returns 0.0 if no scores.
    """
    all_scores = [s for j in judgments for s in j.scores]
    if not all_scores:
        return 0.0
    redundant = sum(1 for s in all_scores if s.redundant)
    return redundant / len(all_scores)
