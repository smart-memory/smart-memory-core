"""Unit tests for EnhancedSimilarityFramework and convenience functions."""


import pytest

pytestmark = pytest.mark.unit

from smartmemory.models.memory_item import MemoryItem
from smartmemory.similarity.enhanced_metrics import SimilarityResult
from smartmemory.similarity.framework import (
    SimilarityConfig,
    EnhancedSimilarityFramework,
    calculate_similarity,
    find_similar_items,
    cluster_similar_items,
)


def _make_item(content="test", item_id=None, metadata=None):
    kwargs = {"content": content}
    if item_id:
        kwargs["item_id"] = item_id
    if metadata:
        kwargs["metadata"] = metadata
    return MemoryItem(**kwargs)


# ---------------------------------------------------------------------------
# SimilarityConfig
# ---------------------------------------------------------------------------
class TestSimilarityConfig:
    def test_default_weights_sum_to_one(self):
        cfg = SimilarityConfig()
        total = (
            cfg.content_weight + cfg.semantic_weight + cfg.temporal_weight +
            cfg.graph_weight + cfg.metadata_weight + cfg.agent_workflow_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_auto_normalize_weights(self):
        cfg = SimilarityConfig(
            content_weight=1.0, semantic_weight=1.0, temporal_weight=1.0,
            graph_weight=1.0, metadata_weight=1.0, agent_workflow_weight=1.0,
        )
        total = (
            cfg.content_weight + cfg.semantic_weight + cfg.temporal_weight +
            cfg.graph_weight + cfg.metadata_weight + cfg.agent_workflow_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_default_thresholds(self):
        cfg = SimilarityConfig()
        assert cfg.similarity_threshold == 0.5
        assert cfg.high_similarity_threshold == 0.8

    def test_caching_defaults(self):
        cfg = SimilarityConfig()
        assert cfg.enable_caching is True
        assert cfg.max_cache_size == 1000


# ---------------------------------------------------------------------------
# EnhancedSimilarityFramework
# ---------------------------------------------------------------------------
class TestEnhancedSimilarityFramework:
    @pytest.fixture
    def framework(self):
        return EnhancedSimilarityFramework()

    def test_initializes_all_metrics(self, framework):
        expected = {"content", "semantic", "temporal", "graph", "metadata", "agent_workflow"}
        assert set(framework.metrics.keys()) == expected

    def test_calculate_similarity_returns_float(self, framework):
        item1 = _make_item("python programming language")
        item2 = _make_item("python data science library")
        score = framework.calculate_similarity(item1, item2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calculate_similarity_identical_items(self, framework):
        item1 = _make_item("the exact same content here", item_id="id1")
        item2 = _make_item("the exact same content here", item_id="id2")
        score = framework.calculate_similarity(item1, item2)
        assert score > 0.3

    def test_calculate_similarity_none_items(self, framework):
        item = _make_item("something")
        assert framework.calculate_similarity(None, item) == 0.0
        assert framework.calculate_similarity(item, None) == 0.0

    def test_calculate_similarity_detailed(self, framework):
        item1 = _make_item("machine learning algorithms")
        item2 = _make_item("deep learning neural networks")
        result = framework.calculate_similarity(item1, item2, return_detailed=True)
        assert isinstance(result, SimilarityResult)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "semantic_score")
        assert hasattr(result, "content_score")
        assert hasattr(result, "explanation")
        assert hasattr(result, "confidence")

    def test_calculate_similarity_specific_metrics(self, framework):
        item1 = _make_item("test content A")
        item2 = _make_item("test content B")
        result = framework.calculate_similarity(
            item1, item2, metrics=["content"], return_detailed=True
        )
        assert isinstance(result, SimilarityResult)

    def test_calculate_similarity_unknown_metric_skipped(self, framework):
        item1 = _make_item("a")
        item2 = _make_item("b")
        # Should not raise, just skip unknown metric
        score = framework.calculate_similarity(item1, item2, metrics=["nonexistent"])
        assert score == 0.0

    def test_caching_stores_and_retrieves(self, framework):
        item1 = _make_item("cache test A", item_id="cache1")
        item2 = _make_item("cache test B", item_id="cache2")

        score1 = framework.calculate_similarity(item1, item2)
        assert len(framework._similarity_cache) == 1

        score2 = framework.calculate_similarity(item1, item2)
        assert score1 == score2
        # Still 1 entry — second call was a cache hit, not a new computation
        assert len(framework._similarity_cache) == 1

    def test_cache_stats_reports_correctly(self, framework):
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        framework.calculate_similarity(item1, item2)

        stats = framework.get_cache_stats()
        assert stats["caching_enabled"] is True
        assert stats["cache_size"] == 1
        assert stats["max_cache_size"] == framework.config.max_cache_size

    def test_cache_initialized_as_dict_when_enabled(self, framework):
        assert framework._similarity_cache is not None
        assert isinstance(framework._similarity_cache, dict)

    def test_cache_key_symmetric(self, framework):
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        key1 = framework._get_cache_key(item1, item2, None)
        key2 = framework._get_cache_key(item2, item1, None)
        assert key1 == key2

    def test_cache_key_includes_metrics(self, framework):
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        key_all = framework._get_cache_key(item1, item2, None)
        key_specific = framework._get_cache_key(item1, item2, ["content"])
        assert key_all != key_specific

    def test_clear_cache(self, framework):
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        framework.calculate_similarity(item1, item2)
        assert len(framework._similarity_cache) >= 1

        framework.clear_cache()
        assert len(framework._similarity_cache) == 0

    def test_cache_disabled(self):
        cfg = SimilarityConfig(enable_caching=False)
        fw = EnhancedSimilarityFramework(config=cfg)
        assert fw._similarity_cache is None
        stats = fw.get_cache_stats()
        assert stats["caching_enabled"] is False

    def test_manage_cache_size_evicts(self):
        cfg = SimilarityConfig(max_cache_size=5)
        fw = EnhancedSimilarityFramework(config=cfg)
        # Fill cache beyond limit via real calculations
        for i in range(6):
            item1 = _make_item(f"item {i}", item_id=f"a_{i}")
            item2 = _make_item(f"other {i}", item_id=f"b_{i}")
            fw.calculate_similarity(item1, item2)
        assert len(fw._similarity_cache) <= 5


class TestFindSimilarItems:
    @pytest.fixture
    def framework(self):
        return EnhancedSimilarityFramework()

    def test_finds_similar_items(self, framework):
        target = _make_item("python programming language", item_id="target")
        candidates = [
            _make_item("python data science", item_id="c1"),
            _make_item("java programming language", item_id="c2"),
            _make_item("completely unrelated topic about cooking", item_id="c3"),
        ]
        results = framework.find_similar_items(target, candidates, threshold=0.0)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_skips_self(self, framework):
        target = _make_item("test", item_id="same_id")
        candidates = [_make_item("test", item_id="same_id")]
        results = framework.find_similar_items(target, candidates, threshold=0.0)
        assert len(results) == 0

    def test_respects_threshold(self, framework):
        target = _make_item("python", item_id="t")
        candidates = [_make_item("completely different xyz abc", item_id="c1")]
        results = framework.find_similar_items(target, candidates, threshold=0.99)
        assert len(results) == 0

    def test_respects_max_results(self, framework):
        target = _make_item("test content", item_id="t")
        candidates = [_make_item(f"test content {i}", item_id=f"c{i}") for i in range(10)]
        results = framework.find_similar_items(target, candidates, threshold=0.0, max_results=3)
        assert len(results) <= 3

    def test_sorted_by_similarity_descending(self, framework):
        target = _make_item("python programming", item_id="t")
        candidates = [
            _make_item("python programming language features", item_id="c1"),
            _make_item("xyz abc def", item_id="c2"),
        ]
        results = framework.find_similar_items(target, candidates, threshold=0.0)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]


class TestClusterItems:
    @pytest.fixture
    def framework(self):
        return EnhancedSimilarityFramework()

    def test_single_item_cluster(self, framework):
        items = [_make_item("alone", item_id="a")]
        clusters = framework.cluster_items(items, similarity_threshold=0.9)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_all_different_items(self, framework):
        items = [
            _make_item("alpha beta gamma", item_id="a"),
            _make_item("one two three", item_id="b"),
            _make_item("xyz uvw rst", item_id="c"),
        ]
        clusters = framework.cluster_items(items, similarity_threshold=0.99)
        # With very high threshold, each item should be its own cluster
        assert len(clusters) == 3

    def test_empty_items(self, framework):
        clusters = framework.cluster_items([])
        assert clusters == []


class TestSimilarityMatrix:
    @pytest.fixture
    def framework(self):
        return EnhancedSimilarityFramework()

    def test_diagonal_is_one(self, framework):
        items = [
            _make_item("a", item_id="id1"),
            _make_item("b", item_id="id2"),
        ]
        matrix = framework.get_similarity_matrix(items)
        assert matrix["id1"]["id1"] == 1.0
        assert matrix["id2"]["id2"] == 1.0

    def test_symmetric(self, framework):
        items = [
            _make_item("python programming", item_id="id1"),
            _make_item("java programming", item_id="id2"),
        ]
        matrix = framework.get_similarity_matrix(items)
        assert matrix["id1"]["id2"] == matrix["id2"]["id1"]

    def test_matrix_size(self, framework):
        items = [_make_item(f"item {i}", item_id=f"id{i}") for i in range(3)]
        matrix = framework.get_similarity_matrix(items)
        assert len(matrix) == 3
        for row in matrix.values():
            assert len(row) == 3


class TestUpdateConfig:
    def test_reinitializes_metrics(self):
        fw = EnhancedSimilarityFramework()
        new_cfg = SimilarityConfig(use_fuzzy_matching=False)
        fw.update_config(new_cfg)
        assert fw.config.use_fuzzy_matching is False

    def test_clears_cache_on_update(self):
        fw = EnhancedSimilarityFramework()
        item1 = _make_item("a", item_id="id1")
        item2 = _make_item("b", item_id="id2")
        fw.calculate_similarity(item1, item2)
        assert len(fw._similarity_cache) >= 1

        fw.update_config(SimilarityConfig())
        assert len(fw._similarity_cache) == 0


class TestWeightedScore:
    def test_zero_scores(self):
        fw = EnhancedSimilarityFramework()
        score = fw._calculate_weighted_score({})
        assert score == 0.0

    def test_all_ones(self):
        fw = EnhancedSimilarityFramework()
        scores = {k: 1.0 for k in fw.metrics.keys()}
        score = fw._calculate_weighted_score(scores)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_agent_workflow_boost(self):
        fw = EnhancedSimilarityFramework()
        scores_low = {"agent_workflow": 0.3, "content": 0.5, "semantic": 0.5,
                       "temporal": 0.5, "graph": 0.5, "metadata": 0.5}
        scores_high = {"agent_workflow": 0.8, "content": 0.5, "semantic": 0.5,
                        "temporal": 0.5, "graph": 0.5, "metadata": 0.5}
        score_low = fw._calculate_weighted_score(scores_low)
        score_high = fw._calculate_weighted_score(scores_high)
        # High agent workflow score should boost overall
        assert score_high > score_low


class TestConfidence:
    def test_all_same_scores_high_confidence(self):
        fw = EnhancedSimilarityFramework()
        scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        confidence = fw._calculate_confidence(scores)
        assert confidence == pytest.approx(1.0)

    def test_varied_scores_lower_confidence(self):
        fw = EnhancedSimilarityFramework()
        scores = {"a": 0.0, "b": 1.0}
        confidence = fw._calculate_confidence(scores)
        assert confidence < 1.0

    def test_empty_scores(self):
        fw = EnhancedSimilarityFramework()
        assert fw._calculate_confidence({}) == 0.0


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    def test_calculate_similarity_function(self):
        item1 = _make_item("python programming", item_id="a")
        item2 = _make_item("python data science", item_id="b")
        score = calculate_similarity(item1, item2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_find_similar_items_function(self):
        target = _make_item("python", item_id="t")
        candidates = [_make_item("python code", item_id="c1")]
        results = find_similar_items(target, candidates, threshold=0.0)
        assert isinstance(results, list)

    def test_cluster_similar_items_function(self):
        items = [
            _make_item("a", item_id="id1"),
            _make_item("b", item_id="id2"),
        ]
        clusters = cluster_similar_items(items, similarity_threshold=0.0)
        assert isinstance(clusters, list)
