"""Unit tests for RetrievalBasedStrengtheningEvolver.

Tests formula correctness, json.loads() guard, config defaults, and update_properties
call shape. No FalkorDB or external services required.
"""

import json
import math
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.enhanced.retrieval_based_strengthening import (
    RetrievalBasedStrengtheningConfig,
    RetrievalBasedStrengtheningEvolver,
)
from smartmemory.models.memory_item import MemoryItem

_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    item_id: str = "item-1",
    content: str = "test memory",
    memory_type: str = "episodic",
    metadata: dict | None = None,
) -> MemoryItem:
    """Create a minimal MemoryItem for testing."""
    return MemoryItem(
        item_id=item_id,
        content=content,
        memory_type=memory_type,
        metadata=metadata or {},
    )


def _make_memory(items: list) -> MagicMock:
    """Return a mock memory object whose search() returns items."""
    memory = MagicMock()
    memory.search.return_value = items
    memory.update_properties.return_value = None
    memory.item_id = None
    return memory


def _profile(
    total_count: int = 10,
    avg_search_rank: float = 1.5,
    velocity_7d: float = 2.0,
    velocity_30d: float = 1.0,
) -> dict:
    """Build a minimal retrieval profile dict."""
    return {
        "total_count": total_count,
        "get_count": 0,
        "search_count": total_count,
        "avg_search_rank": avg_search_rank,
        "avg_similarity_score": 0.85,
        "velocity_7d": velocity_7d,
        "velocity_30d": velocity_30d,
        "min_search_rank": 1,
        "top1_count": 3,
        "sources": {},
    }


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestRetrievalBasedStrengtheningMetadata:
    def test_metadata_name(self):
        meta = RetrievalBasedStrengtheningEvolver.metadata()
        assert meta.name == "retrieval_based_strengthening"

    def test_metadata_plugin_type(self):
        meta = RetrievalBasedStrengtheningEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version_is_set(self):
        meta = RetrievalBasedStrengtheningEvolver.metadata()
        assert meta.version

    def test_metadata_tags_include_retrieval(self):
        meta = RetrievalBasedStrengtheningEvolver.metadata()
        assert "retrieval" in meta.tags

    def test_metadata_tags_include_retention(self):
        meta = RetrievalBasedStrengtheningEvolver.metadata()
        assert "retention" in meta.tags


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestRetrievalBasedStrengtheningConfig:
    def test_default_retrieval_weight(self):
        cfg = RetrievalBasedStrengtheningConfig()
        assert cfg.retrieval_weight == 0.1

    def test_default_unretrieved_decay_rate(self):
        cfg = RetrievalBasedStrengtheningConfig()
        assert cfg.unretrieved_decay_rate == 0.005

    def test_default_lookback_days(self):
        cfg = RetrievalBasedStrengtheningConfig()
        assert cfg.lookback_days == 30

    def test_default_max_memories(self):
        cfg = RetrievalBasedStrengtheningConfig()
        assert cfg.max_memories == 500

    def test_default_memory_types(self):
        cfg = RetrievalBasedStrengtheningConfig()
        assert "episodic" in cfg.memory_types
        assert "semantic" in cfg.memory_types
        assert "procedural" in cfg.memory_types

    def test_custom_retrieval_weight(self):
        cfg = RetrievalBasedStrengtheningConfig(retrieval_weight=0.5)
        assert cfg.retrieval_weight == 0.5


# ---------------------------------------------------------------------------
# Zero-retrieval penalty tests
# ---------------------------------------------------------------------------


class TestZeroRetrievalPenalty:
    """When no retrieval profile exists, an unretrieved decay penalty is applied."""

    def _run_no_profile(self, base_retention: float = 0.8) -> float:
        item = _make_item(metadata={"retention_score": base_retention})
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        assert memory.update_properties.called
        return memory.update_properties.call_args[0][1]["retention_score"]

    def test_penalty_applied_when_no_profile(self):
        """retention_score should decrease (penalty) when no profile present."""
        score = self._run_no_profile(base_retention=0.8)
        assert score < 0.8

    def test_penalty_applied_when_zero_count(self):
        """Zero total_count in profile triggers the same penalty path."""
        zero_profile = _profile(total_count=0)
        item = _make_item(metadata={"retention_score": 0.8, "retrieval__profile": zero_profile})
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        score = memory.update_properties.call_args[0][1]["retention_score"]
        assert score < 0.8

    def test_penalty_formula(self):
        """Verify penalty = base * (1 - rate * days)."""
        cfg = RetrievalBasedStrengtheningConfig(unretrieved_decay_rate=0.005, lookback_days=30)
        base = 0.8
        expected = base * (1.0 - 0.005 * 30)
        item = _make_item(metadata={"retention_score": base})
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver(config=cfg)
        evolver.evolve(memory)
        score = memory.update_properties.call_args[0][1]["retention_score"]
        assert abs(score - expected) < 1e-9

    def test_penalty_cannot_go_below_zero(self):
        """Score is clamped to 0.0 even if penalty is very large."""
        cfg = RetrievalBasedStrengtheningConfig(unretrieved_decay_rate=1.0, lookback_days=100)
        item = _make_item(metadata={"retention_score": 0.5})
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver(config=cfg)
        evolver.evolve(memory)
        score = memory.update_properties.call_args[0][1]["retention_score"]
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Boost formula tests
# ---------------------------------------------------------------------------


class TestBoostFormula:
    """Verify the rank_weight × log-count × recency_weight boost formula."""

    def _run_with_profile(
        self,
        prof: dict,
        base_retention: float = 0.5,
        cfg: RetrievalBasedStrengtheningConfig | None = None,
    ) -> float:
        item = _make_item(metadata={"retention_score": base_retention, "retrieval__profile": prof})
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver(config=cfg or RetrievalBasedStrengtheningConfig())
        evolver.evolve(memory)
        return memory.update_properties.call_args[0][1]["retention_score"]

    def test_boost_increases_retention(self):
        """A memory with retrieval history should have higher retention than base."""
        prof = _profile(total_count=20, avg_search_rank=1.0, velocity_7d=5.0, velocity_30d=2.0)
        score = self._run_with_profile(prof, base_retention=0.5)
        assert score > 0.5

    def test_rank_weight_formula(self):
        """rank_weight = 1 / (1 + avg_search_rank)."""
        avg_rank = 2.0
        rank_weight = 1.0 / (1.0 + avg_rank)
        assert abs(rank_weight - (1.0 / 3.0)) < 1e-9

    def test_boost_formula_exact(self):
        """Verify boost = weight × log(1+count) × rank_weight × recency_weight."""
        cfg = RetrievalBasedStrengtheningConfig(retrieval_weight=0.1)
        total_count = 10
        avg_search_rank = 1.5
        velocity_7d = 4.0
        velocity_30d = 2.0
        base = 0.5

        rank_weight = 1.0 / (1.0 + avg_search_rank)
        recency_weight = velocity_7d / (velocity_30d + _EPSILON)
        boost = cfg.retrieval_weight * math.log1p(total_count) * rank_weight * recency_weight
        expected = min(1.0, base + boost)

        prof = _profile(
            total_count=total_count,
            avg_search_rank=avg_search_rank,
            velocity_7d=velocity_7d,
            velocity_30d=velocity_30d,
        )
        actual = self._run_with_profile(prof, base_retention=base, cfg=cfg)
        assert abs(actual - expected) < 1e-9

    def test_retention_clamped_to_one(self):
        """Boost that would push score above 1.0 is clamped to exactly 1.0."""
        # Very high boost scenario
        cfg = RetrievalBasedStrengtheningConfig(retrieval_weight=10.0)
        prof = _profile(total_count=1000, avg_search_rank=0.1, velocity_7d=100.0, velocity_30d=1.0)
        score = self._run_with_profile(prof, base_retention=0.99, cfg=cfg)
        assert score <= 1.0

    def test_high_rank_means_lower_boost(self):
        """Items appearing at rank 10 should get less boost than rank 1 items."""
        prof_top = _profile(total_count=10, avg_search_rank=1.0, velocity_7d=3.0, velocity_30d=1.0)
        prof_bottom = _profile(total_count=10, avg_search_rank=10.0, velocity_7d=3.0, velocity_30d=1.0)
        score_top = self._run_with_profile(prof_top, base_retention=0.5)
        score_bottom = self._run_with_profile(prof_bottom, base_retention=0.5)
        assert score_top > score_bottom

    def test_high_recency_means_more_boost(self):
        """Higher velocity_7d relative to velocity_30d → more recency boost."""
        # Hot streak: lots of recent retrievals
        prof_hot = _profile(total_count=10, avg_search_rank=1.5, velocity_7d=10.0, velocity_30d=2.0)
        # Cold: no recent retrievals
        prof_cold = _profile(total_count=10, avg_search_rank=1.5, velocity_7d=0.1, velocity_30d=2.0)
        score_hot = self._run_with_profile(prof_hot, base_retention=0.5)
        score_cold = self._run_with_profile(prof_cold, base_retention=0.5)
        assert score_hot > score_cold


# ---------------------------------------------------------------------------
# JSON deserialisation guard tests
# ---------------------------------------------------------------------------


class TestJsonDeserialisationGuard:
    """retrieval__profile stored as JSON string (FalkorDB path) and as dict (test path)."""

    def _get_score(self, profile_value) -> float:
        """Run evolver with a specific profile value and return the written retention_score."""
        item = _make_item(metadata={"retention_score": 0.5, "retrieval__profile": profile_value})
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        return memory.update_properties.call_args[0][1]["retention_score"]

    def test_profile_as_dict(self):
        """Profile stored as a raw dict (in-memory/test path) is handled correctly."""
        prof = _profile(total_count=5, avg_search_rank=1.0, velocity_7d=2.0, velocity_30d=1.0)
        score = self._get_score(prof)
        assert score > 0.5  # boost applied

    def test_profile_as_json_string(self):
        """Profile stored as JSON string (FalkorDB production path) is deserialised correctly."""
        prof = _profile(total_count=5, avg_search_rank=1.0, velocity_7d=2.0, velocity_30d=1.0)
        score = self._get_score(json.dumps(prof))
        assert score > 0.5  # same boost as dict path

    def test_dict_and_json_string_produce_same_score(self):
        """Dict and JSON-string paths must produce identical scores."""
        prof = _profile(total_count=5, avg_search_rank=1.0, velocity_7d=2.0, velocity_30d=1.0)
        score_dict = self._get_score(prof)
        score_str = self._get_score(json.dumps(prof))
        assert abs(score_dict - score_str) < 1e-12

    def test_invalid_json_string_falls_back_to_penalty(self):
        """An unparseable JSON string should fall back to the zero-retrieval penalty path."""
        score = self._get_score("not-valid-json{")
        assert score < 0.5  # penalty, not boost

    def test_none_profile_applies_penalty(self):
        """Missing profile (None) should apply the unretrieved decay penalty."""
        score = self._get_score(None)
        assert score < 0.5

    def test_wrong_type_profile_applies_penalty(self):
        """Non-dict, non-string profile (e.g. integer) falls back to penalty."""
        score = self._get_score(42)
        assert score < 0.5


# ---------------------------------------------------------------------------
# update_properties call shape tests
# ---------------------------------------------------------------------------


class TestUpdatePropertiesCalls:
    def test_called_once_per_item(self):
        items = [_make_item("a"), _make_item("b")]
        memory = _make_memory(items)
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        assert memory.update_properties.call_count == 2

    def test_called_with_correct_item_id(self):
        item = _make_item("unique-xyz")
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        call_id = memory.update_properties.call_args[0][0]
        assert call_id == "unique-xyz"

    def test_empty_search_no_update(self):
        memory = _make_memory([])
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()

    def test_none_search_no_update(self):
        memory = MagicMock()
        memory.search.return_value = None
        memory.item_id = None
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()

    def test_search_called_with_correct_params(self):
        memory = _make_memory([])
        cfg = RetrievalBasedStrengtheningConfig(memory_types=["episodic"], max_memories=99)
        evolver = RetrievalBasedStrengtheningEvolver(config=cfg)
        evolver.evolve(memory)
        memory.search.assert_called_once()
        _, kwargs = memory.search.call_args
        assert kwargs.get("memory_types") == ["episodic"]
        assert kwargs.get("limit") == 99

    def test_retention_score_key_present_in_props(self):
        item = _make_item()
        memory = _make_memory([item])
        evolver = RetrievalBasedStrengtheningEvolver()
        evolver.evolve(memory)
        props = memory.update_properties.call_args[0][1]
        assert "retention_score" in props
