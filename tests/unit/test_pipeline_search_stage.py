from unittest.mock import patch

from smartmemory.memory.pipeline.stages.search import Search
from smartmemory.models.memory_item import MemoryItem


class _SearchObject:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, query, top_k=5, **kwargs):
        self.calls.append((query, top_k, kwargs))
        return self._results


class _GraphWithSearchObject:
    def __init__(self, results):
        self.search = _SearchObject(results)
        self._items = {}

    def get_all_node_ids(self):
        return list(self._items)

    def get_node(self, item_id):
        return self._items[item_id]


class _GraphFallbackOnly:
    search = None

    def __init__(self, items):
        self._items = {item.item_id: item for item in items}

    def get_all_node_ids(self):
        return list(self._items)

    def get_node(self, item_id):
        return self._items[item_id]


def test_search_uses_search_object_method_when_graph_search_is_not_callable():
    item = MemoryItem(item_id="item-1", content="alpha result", memory_type="semantic")
    graph = _GraphWithSearchObject([item])
    stage = Search(graph)

    with patch("smartmemory.utils.cache.get_cache", side_effect=RuntimeError("no cache")):
        results = stage.search("alpha", top_k=3)

    assert results == [item]
    assert graph.search.calls == [("alpha", 6, {})]


def test_search_falls_through_to_manual_similarity_when_graph_search_returns_empty():
    item = MemoryItem(item_id="item-1", content="alpha result", memory_type="semantic")
    graph = _GraphWithSearchObject([])
    graph._items = {"item-1": item}
    stage = Search(graph)

    with (
        patch("smartmemory.utils.cache.get_cache", side_effect=RuntimeError("no cache")),
        patch.object(stage.similarity_framework, "calculate_similarity", return_value=0.9),
    ):
        results = stage.search("alpha", top_k=3)

    assert results == [item]


def test_search_blank_query_with_recency_sort_returns_recent_items_without_similarity():
    older = MemoryItem(item_id="item-1", content="older", memory_type="semantic")
    older.created_at = "2026-03-20T00:00:00"
    newer = MemoryItem(item_id="item-2", content="newer", memory_type="semantic")
    newer.created_at = "2026-03-21T00:00:00"
    stage = Search(_GraphFallbackOnly([older, newer]))

    with patch("smartmemory.utils.cache.get_cache", side_effect=RuntimeError("no cache")):
        results = stage.search("", top_k=1, sort_by="recency")

    assert results == [newer]
