"""Unit tests for HebbianCoRetrievalEvolver — CORE-EVO-ENH-3.

Tests boost formula correctness, threshold gating, max_boost cap, dedup by item_id,
missing-item handling, and plugin metadata. No FalkorDB or external services required.
"""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.enhanced.hebbian_co_retrieval import (
    HebbianCoRetrievalConfig,
    HebbianCoRetrievalEvolver,
)
from smartmemory.models.memory_item import MemoryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(item_id: str = "item-1", retention_score: float = 0.5) -> MemoryItem:
    return MemoryItem(
        item_id=item_id,
        content="test memory",
        memory_type="episodic",
        metadata={"retention_score": retention_score},
    )


def _make_memory(rows: list, items: dict | None = None) -> MagicMock:
    """Build a mock memory whose execute_cypher returns rows and get() returns items.

    Args:
        rows: List of (id_a, id_b, cnt) tuples returned by execute_cypher.
        items: Dict of item_id → MemoryItem. Missing keys return None.
    """
    items = items or {}
    memory = MagicMock()
    memory.execute_cypher.return_value = rows
    memory.get.side_effect = lambda item_id: items.get(item_id)
    memory.update_properties.return_value = None
    return memory


# ---------------------------------------------------------------------------
# Boost formula tests
# ---------------------------------------------------------------------------


class TestBoostFormula:
    def test_boost_above_threshold(self):
        """co_retrieval_count=10, threshold=3 → boost=(10-3)*0.02=0.14."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.3)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        item_a = _make_item("a", retention_score=0.5)
        item_b = _make_item("b", retention_score=0.6)
        memory = _make_memory(
            rows=[("a", "b", 10)],
            items={"a": item_a, "b": item_b},
        )

        evolver.evolve(memory)

        calls = memory.update_properties.call_args_list
        assert len(calls) == 2
        # id_a boost: min(0.3, (10-3)*0.02) = 0.14 → 0.5 + 0.14 = 0.64
        assert abs(calls[0][0][1]["retention_score"] - 0.64) < 1e-9
        # id_b boost: same formula → 0.6 + 0.14 = 0.74
        assert abs(calls[1][0][1]["retention_score"] - 0.74) < 1e-9

    def test_no_boost_below_threshold(self):
        """co_retrieval_count=2 < threshold=3 → no rows returned, no boost."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        # execute_cypher returns empty (Cypher WHERE filters it out)
        memory = _make_memory(rows=[])

        evolver.evolve(memory)

        memory.update_properties.assert_not_called()

    def test_max_boost_cap(self):
        """Very high co_retrieval_count → boost capped at max_boost."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.1)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        item_a = _make_item("a", retention_score=0.5)
        item_b = _make_item("b", retention_score=0.5)
        memory = _make_memory(
            rows=[("a", "b", 1000)],
            items={"a": item_a, "b": item_b},
        )

        evolver.evolve(memory)

        calls = memory.update_properties.call_args_list
        for c in calls:
            score = c[0][1]["retention_score"]
            # max_boost=0.1, base=0.5 → 0.6
            assert abs(score - 0.6) < 1e-9

    def test_retention_never_exceeds_1(self):
        """Base retention 0.95 + any boost → capped at 1.0."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.3)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        item_a = _make_item("a", retention_score=0.95)
        item_b = _make_item("b", retention_score=0.95)
        memory = _make_memory(
            rows=[("a", "b", 50)],
            items={"a": item_a, "b": item_b},
        )

        evolver.evolve(memory)

        for c in memory.update_properties.call_args_list:
            assert c[0][1]["retention_score"] <= 1.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_by_item_id(self):
        """Same node appears in two strong edges → only one boost applied."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.3)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        item_a = _make_item("a", retention_score=0.5)
        item_b = _make_item("b", retention_score=0.5)
        item_c = _make_item("c", retention_score=0.5)
        # Node "a" appears in two rows — should only get ONE boost
        memory = _make_memory(
            rows=[("a", "b", 10), ("a", "c", 8)],
            items={"a": item_a, "b": item_b, "c": item_c},
        )

        evolver.evolve(memory)

        # a, b, c each boosted exactly once
        ids_boosted = [c[0][0] for c in memory.update_properties.call_args_list]
        assert ids_boosted.count("a") == 1
        assert ids_boosted.count("b") == 1
        assert ids_boosted.count("c") == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_item_skipped(self):
        """memory.get() returns None → no error, other items still processed."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        item_b = _make_item("b", retention_score=0.5)
        # "a" is missing from store
        memory = _make_memory(
            rows=[("a", "b", 10)],
            items={"b": item_b},
        )

        evolver.evolve(memory)  # must not raise

        # Only "b" updated
        ids_boosted = [c[0][0] for c in memory.update_properties.call_args_list]
        assert "a" not in ids_boosted
        assert "b" in ids_boosted

    def test_execute_cypher_failure_does_not_raise(self):
        """If execute_cypher raises, evolve() catches and returns cleanly."""
        evolver = HebbianCoRetrievalEvolver()
        memory = MagicMock()
        memory.execute_cypher.side_effect = RuntimeError("db error")

        evolver.evolve(memory)  # must not raise

        memory.update_properties.assert_not_called()

    def test_empty_rows_no_updates(self):
        """No qualifying edges → no update_properties calls."""
        evolver = HebbianCoRetrievalEvolver()
        memory = _make_memory(rows=[])

        evolver.evolve(memory)

        memory.update_properties.assert_not_called()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_returns_correct_plugin_type(self):
        meta = HebbianCoRetrievalEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_name(self):
        meta = HebbianCoRetrievalEvolver.metadata()
        assert meta.name == "hebbian_co_retrieval"

    def test_config_defaults(self):
        cfg = HebbianCoRetrievalConfig()
        assert cfg.weight_threshold == 3.0
        assert cfg.max_edges_per_cycle == 500
        assert cfg.retention_boost_per_unit == 0.02
        assert cfg.max_boost == 0.3
        assert "episodic" in cfg.memory_types
