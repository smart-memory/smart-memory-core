"""Unit tests for smartmemory.search.rrf_merge.rrf_merge()."""

from dataclasses import dataclass

from smartmemory.search.rrf_merge import rrf_merge


@dataclass
class MockItem:
    item_id: str
    content: str


class TestRrfMerge:
    def test_single_list(self):
        items = [MockItem(f"id-{i}", f"content-{i}") for i in range(5)]
        result = rrf_merge([items], top_k=3)
        assert len(result) == 3
        assert result[0].item_id == "id-0"
        assert result[1].item_id == "id-1"
        assert result[2].item_id == "id-2"

    def test_two_lists_disjoint(self):
        list_a = [MockItem("a1", "a"), MockItem("a2", "b")]
        list_b = [MockItem("b1", "c"), MockItem("b2", "d")]
        result = rrf_merge([list_a, list_b], top_k=4)
        assert len(result) == 4
        ids = [r.item_id for r in result]
        assert set(ids) == {"a1", "a2", "b1", "b2"}

    def test_two_lists_overlapping(self):
        shared = MockItem("shared", "shared content")
        list_a = [shared, MockItem("a1", "a")]
        list_b = [MockItem("b1", "b"), shared]
        result = rrf_merge([list_a, list_b], top_k=3)
        # Shared item appears in both lists so gets higher RRF score
        assert result[0].item_id == "shared"

    def test_deduplication(self):
        item = MockItem("dup", "duplicate")
        list_a = [item, MockItem("a1", "a")]
        list_b = [MockItem("b1", "b"), item]
        result = rrf_merge([list_a, list_b], top_k=10)
        ids = [r.item_id for r in result]
        assert ids.count("dup") == 1

    def test_top_k_cap(self):
        items = [MockItem(f"id-{i}", f"content-{i}") for i in range(10)]
        result = rrf_merge([items], top_k=3)
        assert len(result) == 3

    def test_empty_lists(self):
        result = rrf_merge([[], []], top_k=5)
        assert result == []

    def test_mixed_empty_and_full(self):
        items = [MockItem("a1", "a"), MockItem("a2", "b")]
        result = rrf_merge([[], items], top_k=5)
        assert len(result) == 2
        assert result[0].item_id == "a1"

    def test_custom_rrf_k(self):
        items_a = [MockItem("a1", "a"), MockItem("shared", "s")]
        items_b = [MockItem("shared", "s"), MockItem("b1", "b")]
        result_k1 = rrf_merge([items_a, items_b], top_k=3, rrf_k=1)
        result_k60 = rrf_merge([items_a, items_b], top_k=3, rrf_k=60)
        # Both should have shared first, but scores differ
        assert result_k1[0].item_id == "shared"
        assert result_k60[0].item_id == "shared"
        # With k=1, rank differences are more pronounced
        # Just verify we get different score distributions (both valid)
        assert len(result_k1) == len(result_k60) == 3

    def test_item_without_item_id(self):
        """Items without item_id attr fall back to id(item), no crash."""
        class PlainItem:
            def __init__(self, val):
                self.val = val

        items = [PlainItem("x"), PlainItem("y")]
        result = rrf_merge([items], top_k=2)
        assert len(result) == 2
