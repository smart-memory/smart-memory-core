"""Unit tests for ExponentialDecayEvolver.

Tests config defaults, metadata, and the Ebbinghaus forgetting curve logic
with mocked memory. No FalkorDB or external services required.
"""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.enhanced.exponential_decay import (
    ExponentialDecayConfig,
    ExponentialDecayEvolver,
)
from smartmemory.models.memory_item import MemoryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    item_id: str = "item-1",
    content: str = "test memory",
    transaction_time: datetime | None = None,
    metadata: dict | None = None,
) -> MemoryItem:
    """Create a minimal MemoryItem for testing."""
    tx = transaction_time or datetime.now(timezone.utc)
    return MemoryItem(
        item_id=item_id,
        content=content,
        memory_type="episodic",
        transaction_time=tx,
        metadata=metadata or {},
    )


def _make_memory(items: list) -> MagicMock:
    """Return a mock memory object whose search() returns items."""
    memory = MagicMock()
    memory.search.return_value = items
    memory.update_properties.return_value = None
    memory.item_id = None
    return memory


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestExponentialDecayMetadata:
    """Tests for plugin metadata."""

    def test_metadata_name(self):
        meta = ExponentialDecayEvolver.metadata()
        assert meta.name == "exponential_decay"

    def test_metadata_plugin_type(self):
        meta = ExponentialDecayEvolver.metadata()
        assert meta.plugin_type == "evolver"

    def test_metadata_version_is_set(self):
        meta = ExponentialDecayEvolver.metadata()
        assert meta.version

    def test_metadata_tags_include_decay(self):
        meta = ExponentialDecayEvolver.metadata()
        assert "decay" in meta.tags

    def test_metadata_tags_include_episodic(self):
        meta = ExponentialDecayEvolver.metadata()
        assert "episodic" in meta.tags


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestExponentialDecayConfig:
    """Tests for config dataclass defaults."""

    def test_default_stability(self):
        cfg = ExponentialDecayConfig()
        assert cfg.default_stability == 30.0

    def test_default_archive_threshold(self):
        cfg = ExponentialDecayConfig()
        assert cfg.archive_threshold == 0.1

    def test_default_max_memories(self):
        cfg = ExponentialDecayConfig()
        assert cfg.max_memories == 500

    def test_custom_stability(self):
        cfg = ExponentialDecayConfig(default_stability=14.0)
        assert cfg.default_stability == 14.0

    def test_custom_archive_threshold(self):
        cfg = ExponentialDecayConfig(archive_threshold=0.05)
        assert cfg.archive_threshold == 0.05


# ---------------------------------------------------------------------------
# Retention formula tests
# ---------------------------------------------------------------------------


class TestExponentialDecayFormula:
    """Tests for the Ebbinghaus retention formula."""

    def _run_evolve(self, item: MemoryItem, cfg: ExponentialDecayConfig | None = None):
        """Run evolver and return the props dict passed to update_properties."""
        memory = _make_memory([item])
        evolver = ExponentialDecayEvolver(config=cfg or ExponentialDecayConfig())
        evolver.evolve(memory)
        assert memory.update_properties.called, "update_properties should be called"
        # Return the props from the first call
        return memory.update_properties.call_args[0][1]

    def test_retention_formula_30_days(self):
        """After 30 days with stability=30: retention = exp(-1) ≈ 0.3679."""
        tx = datetime.now(timezone.utc) - timedelta(days=30)
        item = _make_item(transaction_time=tx)
        props = self._run_evolve(item)
        expected = math.exp(-1.0)
        assert abs(props["retention_score"] - expected) < 1e-6

    def test_retention_formula_zero_elapsed(self):
        """At creation time: retention = exp(0) = 1.0."""
        item = _make_item(transaction_time=datetime.now(timezone.utc))
        props = self._run_evolve(item)
        # elapsed is ~0 seconds → exp(~0) ≈ 1.0
        assert props["retention_score"] > 0.999

    def test_retention_formula_high_stability(self):
        """High stability = slower decay. After 30 days with stability=90: retention = exp(-1/3)."""
        tx = datetime.now(timezone.utc) - timedelta(days=30)
        item = _make_item(transaction_time=tx)
        cfg = ExponentialDecayConfig(default_stability=90.0)
        props = self._run_evolve(item, cfg)
        expected = math.exp(-30.0 / 90.0)
        assert abs(props["retention_score"] - expected) < 1e-6

    def test_stability_read_from_metadata(self):
        """Per-memory stability in metadata overrides config default."""
        tx = datetime.now(timezone.utc) - timedelta(days=30)
        item = _make_item(transaction_time=tx, metadata={"stability": 60.0})
        props = self._run_evolve(item)
        expected = math.exp(-30.0 / 60.0)
        assert abs(props["retention_score"] - expected) < 1e-6

    def test_no_archive_when_retention_above_threshold(self):
        """Memory with high retention should NOT be archived."""
        tx = datetime.now(timezone.utc) - timedelta(days=5)
        item = _make_item(transaction_time=tx)
        props = self._run_evolve(item, ExponentialDecayConfig(archive_threshold=0.1))
        assert props.get("archived") is not True

    def test_archive_triggered_when_retention_below_threshold(self):
        """Memory with very low retention (old + short stability) → archived=True."""
        tx = datetime.now(timezone.utc) - timedelta(days=200)
        item = _make_item(transaction_time=tx)
        cfg = ExponentialDecayConfig(default_stability=30.0, archive_threshold=0.1)
        props = self._run_evolve(item, cfg)
        # exp(-200/30) ≈ 1.2e-3, well below 0.1
        assert props["archived"] is True
        assert props["archive_reason"] == "exponential_decay"

    def test_archive_reason_set_when_archived(self):
        """archive_reason must be 'exponential_decay' when archived."""
        tx = datetime.now(timezone.utc) - timedelta(days=500)
        item = _make_item(transaction_time=tx)
        props = self._run_evolve(item)
        assert props.get("archive_reason") == "exponential_decay"


# ---------------------------------------------------------------------------
# Update_properties call shape tests
# ---------------------------------------------------------------------------


class TestExponentialDecayUpdateCalls:
    """Tests for how update_properties is called."""

    def test_update_properties_called_once_per_item(self):
        """One call to update_properties per memory item."""
        items = [
            _make_item("a", transaction_time=datetime.now(timezone.utc) - timedelta(days=10)),
            _make_item("b", transaction_time=datetime.now(timezone.utc) - timedelta(days=20)),
        ]
        memory = _make_memory(items)
        evolver = ExponentialDecayEvolver()
        evolver.evolve(memory)
        assert memory.update_properties.call_count == 2

    def test_update_properties_uses_correct_item_id(self):
        """update_properties is called with the item's own item_id."""
        tx = datetime.now(timezone.utc) - timedelta(days=10)
        item = _make_item("unique-id-123", transaction_time=tx)
        memory = _make_memory([item])
        evolver = ExponentialDecayEvolver()
        evolver.evolve(memory)
        call_id = memory.update_properties.call_args[0][0]
        assert call_id == "unique-id-123"

    def test_empty_search_result_no_update(self):
        """When search returns empty list, update_properties is never called."""
        memory = _make_memory([])
        evolver = ExponentialDecayEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()

    def test_none_search_result_no_update(self):
        """When search returns None, update_properties is never called."""
        memory = MagicMock()
        memory.search.return_value = None
        memory.item_id = None
        evolver = ExponentialDecayEvolver()
        evolver.evolve(memory)
        memory.update_properties.assert_not_called()

    def test_search_called_with_correct_params(self):
        """search() is called with memory_types=['episodic'] and correct limit."""
        memory = _make_memory([])
        cfg = ExponentialDecayConfig(max_memories=42)
        evolver = ExponentialDecayEvolver(config=cfg)
        evolver.evolve(memory)
        memory.search.assert_called_once()
        _, kwargs = memory.search.call_args
        assert kwargs.get("memory_types") == ["episodic"]
        assert kwargs.get("limit") == 42


# ---------------------------------------------------------------------------
# Config injection tests
# ---------------------------------------------------------------------------


class TestExponentialDecayConfigInjection:
    """Tests for config injection at instantiation time."""

    def test_default_config_used_when_none(self):
        evolver = ExponentialDecayEvolver()
        assert evolver.config.default_stability == 30.0

    def test_custom_config_applied(self):
        cfg = ExponentialDecayConfig(default_stability=14.0)
        evolver = ExponentialDecayEvolver(config=cfg)
        assert evolver.config.default_stability == 14.0
