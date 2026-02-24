"""Unit tests for InterferenceBasedConsolidationEvolver.

Tests config defaults, metadata, interference penalty logic, pair deduplication,
and graceful handling of items without embeddings. No FalkorDB required.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.enhanced.interference_based_consolidation import (
    InterferenceBasedConsolidationConfig,
    InterferenceBasedConsolidationEvolver,
)
from smartmemory.models.memory_item import MemoryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A simple normalized embedding: two orthogonal and one near-identical pair.
_VEC_A = [1.0, 0.0, 0.0]
_VEC_B = [0.0, 1.0, 0.0]
_VEC_C = [0.9999, 0.0141, 0.0]  # cos_sim with A ≈ 0.9999 (above threshold)


def _make_item(
    item_id: str = "item-1",
    content: str = "test memory",
    embedding: list | None = None,
    memory_type: str = "episodic",
    metadata: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        item_id=item_id,
        content=content,
        memory_type=memory_type,
        embedding=embedding,
        transaction_time=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


def _make_memory(items: list, neighbor_items: list | None = None) -> MagicMock:
    """Mock memory whose first search() call returns `items`, subsequent calls return `neighbor_items`."""
    memory = MagicMock()
    memory.item_id = None
    memory.update_properties.return_value = None
    if neighbor_items is None:
        # By default, return the same items for neighbor search
        memory.search.return_value = items
    else:
        # First call returns items, subsequent calls return neighbors
        memory.search.side_effect = [items] + [neighbor_items] * max(1, len(items))
    return memory


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestInterferenceBasedConsolidationMetadata:

    def test_metadata_name(self):
        meta = InterferenceBasedConsolidationEvolver.metadata()
        assert meta.name == "interference_based_consolidation"

    def test_metadata_plugin_type(self):
        meta = InterferenceBasedConsolidationEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version_is_set(self):
        meta = InterferenceBasedConsolidationEvolver.metadata()
        assert meta.version

    def test_metadata_tags_include_interference(self):
        meta = InterferenceBasedConsolidationEvolver.metadata()
        assert "interference" in meta.tags

    def test_metadata_tags_include_consolidation(self):
        meta = InterferenceBasedConsolidationEvolver.metadata()
        assert "consolidation" in meta.tags


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestInterferenceBasedConsolidationConfig:

    def test_default_similarity_threshold(self):
        cfg = InterferenceBasedConsolidationConfig()
        assert cfg.similarity_threshold == 0.85

    def test_default_interference_weight(self):
        cfg = InterferenceBasedConsolidationConfig()
        assert cfg.interference_weight == 0.05

    def test_default_top_k_neighbors(self):
        cfg = InterferenceBasedConsolidationConfig()
        assert cfg.top_k_neighbors == 5

    def test_default_max_memories(self):
        cfg = InterferenceBasedConsolidationConfig()
        assert cfg.max_memories == 200

    def test_custom_similarity_threshold(self):
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.90)
        assert cfg.similarity_threshold == 0.90


# ---------------------------------------------------------------------------
# Items without embeddings are skipped
# ---------------------------------------------------------------------------


class TestItemsWithoutEmbeddings:

    def test_item_without_embedding_skipped(self):
        """Items with no embedding must not trigger neighbor search or update_properties."""
        item = _make_item("no-emb", embedding=None)
        memory = _make_memory([item])
        evolver = InterferenceBasedConsolidationEvolver()
        evolver.evolve(memory)
        # search() called once for the initial fetch; no neighbor search
        assert memory.search.call_count == 1
        memory.update_properties.assert_not_called()

    def test_neighbor_without_embedding_skipped(self):
        """If the neighbor has no embedding, cosine_sim is skipped and no penalty is applied."""
        item_a = _make_item("a", embedding=_VEC_A)
        item_b = _make_item("b", embedding=None)  # neighbor has no embedding
        # First call returns [item_a], second call (neighbor search) returns [item_a, item_b]
        memory = _make_memory([item_a], neighbor_items=[item_a, item_b])
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.85)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        # item_b has no embedding → no penalty → update_properties never called
        memory.update_properties.assert_not_called()


# ---------------------------------------------------------------------------
# Empty / None search results
# ---------------------------------------------------------------------------


class TestEmptySearchResults:

    def test_empty_initial_search_no_error(self):
        memory = _make_memory([])
        evolver = InterferenceBasedConsolidationEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()

    def test_none_initial_search_no_error(self):
        memory = MagicMock()
        memory.item_id = None
        memory.search.return_value = None
        evolver = InterferenceBasedConsolidationEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()

    def test_none_neighbor_search_no_error(self):
        """Neighbor search returning None should be handled gracefully."""
        item_a = _make_item("a", embedding=_VEC_A)
        memory = MagicMock()
        memory.item_id = None
        memory.search.side_effect = [[item_a], None]
        memory.update_properties.return_value = None
        evolver = InterferenceBasedConsolidationEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()


# ---------------------------------------------------------------------------
# Self-comparison skipped
# ---------------------------------------------------------------------------


class TestSelfComparisonSkipped:

    def test_item_not_penalized_by_itself(self):
        """When neighbor search returns the same item, it must be skipped."""
        item = _make_item("a", embedding=_VEC_A)
        # Both initial and neighbor search return just item "a"
        memory = _make_memory([item], neighbor_items=[item])
        evolver = InterferenceBasedConsolidationEvolver()
        evolver.evolve(memory)
        # No penalty when only self appears in neighbor list
        memory.update_properties.assert_not_called()


# ---------------------------------------------------------------------------
# Below threshold: no penalty
# ---------------------------------------------------------------------------


class TestBelowSimilarityThreshold:

    def test_orthogonal_items_no_penalty(self):
        """Items with cosine similarity = 0 (orthogonal) must not be penalized."""
        item_a = _make_item("a", embedding=_VEC_A)
        item_b = _make_item("b", embedding=_VEC_B)
        # search returns both items; neighbor search returns both items
        memory = _make_memory([item_a, item_b], neighbor_items=[item_a, item_b])
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.85)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        # sim(A, B) = 0 < 0.85 → no penalty
        memory.update_properties.assert_not_called()


# ---------------------------------------------------------------------------
# Above threshold: penalty applied
# ---------------------------------------------------------------------------


class TestAboveSimilarityThreshold:

    def test_highly_similar_items_penalized(self):
        """Items with cosine similarity > threshold must both be penalized."""
        item_a = _make_item("a", embedding=_VEC_A)
        item_c = _make_item("c", embedding=_VEC_C)  # sim with A ≈ 0.9999
        memory = _make_memory([item_a, item_c], neighbor_items=[item_a, item_c])
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.85, interference_weight=0.05)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        # Both items should have been updated
        updated_ids = {c.args[0] for c in memory.update_properties.call_args_list}
        assert "a" in updated_ids
        assert "c" in updated_ids

    def test_penalty_formula_correctness(self):
        """Verify penalty = sim * weight applied to initial retention_score=1.0."""
        item_a = _make_item("a", embedding=_VEC_A, metadata={"retention_score": 1.0})
        item_c = _make_item("c", embedding=_VEC_C, metadata={"retention_score": 1.0})
        memory = _make_memory([item_a, item_c], neighbor_items=[item_a, item_c])
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.85, interference_weight=0.05)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        # Compute expected score: sim(A, C) ≈ 0.9999, penalty = 0.9999 * 0.05 ≈ 0.04999
        # new_score = 1.0 * (1 - 0.04999) ≈ 0.95001
        sim = MemoryItem.cosine_similarity(_VEC_A, _VEC_C)
        expected_score = max(0.0, 1.0 * (1 - sim * 0.05))
        # Find the update_properties call for item_a
        for c in memory.update_properties.call_args_list:
            if c.args[0] == "a":
                actual_score = c.args[1]["retention_score"]
                assert abs(actual_score - expected_score) < 1e-6, f"Expected {expected_score}, got {actual_score}"
                break
        else:
            pytest.fail("update_properties not called for item 'a'")

    def test_score_clamped_to_zero(self):
        """Retention score must never go below 0.0."""
        # Very low existing score
        item_a = _make_item("a", embedding=_VEC_A, metadata={"retention_score": 0.001})
        item_c = _make_item("c", embedding=_VEC_C, metadata={"retention_score": 0.001})
        memory = _make_memory([item_a, item_c], neighbor_items=[item_a, item_c])
        cfg = InterferenceBasedConsolidationConfig(
            similarity_threshold=0.85, interference_weight=0.99  # extreme weight
        )
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        for c in memory.update_properties.call_args_list:
            score = c.args[1].get("retention_score", 0.0)
            assert score >= 0.0, f"Retention score {score} is negative"

    def test_interference_count_incremented(self):
        """interference_count must be incremented in the update."""
        item_a = _make_item("a", embedding=_VEC_A, metadata={"interference_count": 2})
        item_c = _make_item("c", embedding=_VEC_C)
        memory = _make_memory([item_a, item_c], neighbor_items=[item_a, item_c])
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.85)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        for c in memory.update_properties.call_args_list:
            if c.args[0] == "a":
                assert c.args[1]["interference_count"] == 3  # was 2, incremented by 1
                break
        else:
            pytest.fail("update_properties not called for item 'a'")


# ---------------------------------------------------------------------------
# Pair deduplication
# ---------------------------------------------------------------------------


class TestPairDeduplication:

    def test_pair_only_penalized_once(self):
        """Even if both items appear in each other's neighbor lists, penalty applied once per pair."""
        item_a = _make_item("a", embedding=_VEC_A)
        item_c = _make_item("c", embedding=_VEC_C)
        # Both items returned as initial batch; neighbor search always returns both
        memory = _make_memory([item_a, item_c], neighbor_items=[item_a, item_c])
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.85)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        evolver.evolve(memory)
        # Count calls per item_id — each should appear exactly once
        calls_per_id: dict[str, int] = {}
        for c in memory.update_properties.call_args_list:
            iid = c.args[0]
            calls_per_id[iid] = calls_per_id.get(iid, 0) + 1
        for iid, count in calls_per_id.items():
            assert count == 1, f"Item {iid!r} updated {count} times (expected 1 — dedup failed)"


# ---------------------------------------------------------------------------
# Config injection
# ---------------------------------------------------------------------------


class TestConfigInjection:

    def test_default_config_used_when_none(self):
        evolver = InterferenceBasedConsolidationEvolver()
        assert evolver.config.similarity_threshold == 0.85

    def test_custom_config_applied(self):
        cfg = InterferenceBasedConsolidationConfig(similarity_threshold=0.95)
        evolver = InterferenceBasedConsolidationEvolver(config=cfg)
        assert evolver.config.similarity_threshold == 0.95
