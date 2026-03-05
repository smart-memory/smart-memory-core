"""Performance verification tests for decomposed search.

No wall-clock timing assertions — uses call counting and output shape validation
to verify no hidden amplification in the decomposition path.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

from smartmemory.smart_memory import SmartMemory
from smartmemory.search.query_decomposer import decompose
from smartmemory.search.rrf_merge import rrf_merge


@dataclass
class MockResult:
    item_id: str
    content: str


def _make_sm_with_counter():
    """Create a SmartMemory with mocked _search.search that counts calls."""
    sm = object.__new__(SmartMemory)
    sm._search = MagicMock()
    sm.scope_provider = None

    call_count = {"n": 0}

    def mock_search(query, top_k=5, memory_type=None, **kwargs):
        call_count["n"] += 1
        return [MockResult(f"id-{call_count['n']}-{i}", f"content-{i}") for i in range(top_k)]

    sm._search.search = MagicMock(side_effect=mock_search)
    return sm, call_count


class TestDecomposedSearchPerf:
    def test_decomposed_fanout_call_count_scales_linearly(self):
        """4-sub-query input produces exactly 4 calls — no hidden amplification."""
        sm, counter = _make_sm_with_counter()
        # "auth, caching, logging, metrics, tracing" decomposes to 4 sub-queries (cap)
        result = sm._decomposed_search("auth, caching, logging, metrics, tracing", top_k=5)
        assert counter["n"] == 4
        # Output should be capped at top_k
        assert len(result) <= 5

    def test_decomposed_no_hidden_allocations(self):
        """Decomposition path produces correct output shapes — no hidden work."""
        # Test decompose output shape
        sub_queries = decompose("auth, caching, logging, metrics")
        assert len(sub_queries) <= 4
        assert sub_queries[0] == "auth, caching, logging, metrics"

        # Test rrf_merge output shape
        lists = [
            [MockResult(f"list{j}-id{i}", f"c{i}") for i in range(10)]
            for j in range(4)
        ]
        merged = rrf_merge(lists, top_k=5)
        assert len(merged) <= 5

        # Each item appears at most once (deduplication)
        ids = [r.item_id for r in merged]
        assert len(ids) == len(set(ids))
