"""Heuristic query decomposition for multi-topic search."""

import re
from typing import List

# Split on explicit conjunctions and punctuation separators
_SPLIT_PATTERN = re.compile(r'\b(?:and|or)\b|[,;]', re.IGNORECASE)

# Minimum length for a fragment to be considered a valid sub-query
_MIN_FRAGMENT_LEN = 3

# Maximum number of sub-queries (including original)
_MAX_SUB_QUERIES = 4


def decompose(query: str) -> List[str]:
    """Decompose a compound query into 1-4 sub-queries.

    Splits on 'and', 'or', commas, and semicolons.
    Always includes the original query as the first sub-query.

    Args:
        query: The original search query.

    Returns:
        List of sub-queries. Returns [query] if no decomposition is possible.
        Returns [] for empty/None input.
    """
    if not query or len(query.strip()) < _MIN_FRAGMENT_LEN:
        return [query] if query else []

    raw_fragments = _SPLIT_PATTERN.split(query)
    fragments = [f.strip() for f in raw_fragments if len(f.strip()) >= _MIN_FRAGMENT_LEN]

    # No decomposition if: no split happened, or all fragments were too short
    if len(fragments) <= 1 and len(raw_fragments) <= 1:
        return [query]

    if not fragments:
        return [query]

    # Original query first, then individual fragments up to the cap
    sub_queries = [query] + fragments[:_MAX_SUB_QUERIES - 1]
    return sub_queries
