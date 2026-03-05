"""Cross-query Reciprocal Rank Fusion for multi-query result merging."""

from typing import List


def rrf_merge(result_lists: List[list], top_k: int, rrf_k: int = 60) -> list:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of result lists, each sorted by relevance.
        top_k: Maximum number of results to return.
        rrf_k: RRF constant (default 60, matching VectorStore.search).

    Returns:
        Merged and re-ranked list of results, deduplicated by item_id.
    """
    scores: dict = {}
    items: dict = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, 1):
            item_id = getattr(item, "item_id", id(item))
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (rrf_k + rank))
            if item_id not in items:
                items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [items[iid] for iid in sorted_ids[:top_k]]
