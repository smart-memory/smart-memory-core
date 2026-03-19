"""Unit tests for HebbianCoRetrievalEvolver — CORE-EVO-ENH-3 + CORE-EVO-LIVE-1 backend API rewrite.

Tests boost formula correctness, threshold gating, max_boost cap, dedup by item_id,
missing-item handling, and plugin metadata. No FalkorDB or external services required.

CORE-EVO-LIVE-1: Rewritten from Cypher mocks to backend API mocks (get_all_edges,
get_node, set_properties) to match the rewritten _evolve_impl.
"""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.plugins.evolvers.enhanced.hebbian_co_retrieval import (
    HebbianCoRetrievalConfig,
    HebbianCoRetrievalEvolver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(item_id: str, retention_score: float = 0.5) -> dict:
    return {"item_id": item_id, "retention_score": retention_score, "memory_type": "episodic"}


def _make_edge(source_id: str, target_id: str, co_retrieval_count: float) -> dict:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": "CO_RETRIEVED",
        "properties": {"co_retrieval_count": co_retrieval_count},
    }


def _make_memory(edges: list, nodes: dict | None = None) -> MagicMock:
    """Build a mock memory whose backend uses the new API.

    Args:
        edges: List of edge dicts returned by get_all_edges().
        nodes: Dict of item_id → node dict. Missing keys return None.
    """
    nodes = nodes or {}
    backend = MagicMock()
    backend.get_all_edges.return_value = edges
    backend.get_node.side_effect = lambda item_id: nodes.get(item_id)
    backend.set_properties.return_value = True

    graph = MagicMock()
    graph.backend = backend

    memory = MagicMock()
    memory._graph = graph
    return memory


# ---------------------------------------------------------------------------
# Boost formula tests
# ---------------------------------------------------------------------------


class TestBoostFormula:
    def test_boost_above_threshold(self):
        """co_retrieval_count=10, threshold=3 → boost=(10-3)*0.02=0.14."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.3)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        edges = [_make_edge("a", "b", 10)]
        nodes = {"a": _make_node("a", 0.5), "b": _make_node("b", 0.6)}
        memory = _make_memory(edges, nodes)

        evolver.evolve(memory)

        calls = memory._graph.backend.set_properties.call_args_list
        assert len(calls) == 2
        # id_a boost: min(0.3, (10-3)*0.02) = 0.14 → 0.5 + 0.14 = 0.64
        assert abs(calls[0][0][1]["retention_score"] - 0.64) < 1e-9
        # id_b boost: same formula → 0.6 + 0.14 = 0.74
        assert abs(calls[1][0][1]["retention_score"] - 0.74) < 1e-9

    def test_no_boost_below_threshold(self):
        """co_retrieval_count=2 < threshold=3 → no qualifying edges, no boost."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        edges = [_make_edge("a", "b", 2)]  # Below threshold
        memory = _make_memory(edges)

        evolver.evolve(memory)

        memory._graph.backend.set_properties.assert_not_called()

    def test_max_boost_cap(self):
        """Very high co_retrieval_count → boost capped at max_boost."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.1)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        edges = [_make_edge("a", "b", 1000)]
        nodes = {"a": _make_node("a", 0.5), "b": _make_node("b", 0.5)}
        memory = _make_memory(edges, nodes)

        evolver.evolve(memory)

        calls = memory._graph.backend.set_properties.call_args_list
        for c in calls:
            score = c[0][1]["retention_score"]
            # max_boost=0.1, base=0.5 → 0.6
            assert abs(score - 0.6) < 1e-9

    def test_retention_never_exceeds_1(self):
        """Base retention 0.95 + any boost → capped at 1.0."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.3)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        edges = [_make_edge("a", "b", 50)]
        nodes = {"a": _make_node("a", 0.95), "b": _make_node("b", 0.95)}
        memory = _make_memory(edges, nodes)

        evolver.evolve(memory)

        for c in memory._graph.backend.set_properties.call_args_list:
            assert c[0][1]["retention_score"] <= 1.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_by_item_id(self):
        """Same node appears in two strong edges → only one boost applied."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0, retention_boost_per_unit=0.02, max_boost=0.3)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        edges = [_make_edge("a", "b", 10), _make_edge("a", "c", 8)]
        nodes = {"a": _make_node("a", 0.5), "b": _make_node("b", 0.5), "c": _make_node("c", 0.5)}
        memory = _make_memory(edges, nodes)

        evolver.evolve(memory)

        # a, b, c each boosted exactly once
        ids_boosted = [c[0][0] for c in memory._graph.backend.set_properties.call_args_list]
        assert ids_boosted.count("a") == 1
        assert ids_boosted.count("b") == 1
        assert ids_boosted.count("c") == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_item_skipped(self):
        """backend.get_node() returns None → no error, other items still processed."""
        cfg = HebbianCoRetrievalConfig(weight_threshold=3.0)
        evolver = HebbianCoRetrievalEvolver(config=cfg)

        edges = [_make_edge("a", "b", 10)]
        nodes = {"b": _make_node("b", 0.5)}  # "a" missing
        memory = _make_memory(edges, nodes)

        evolver.evolve(memory)  # must not raise

        ids_boosted = [c[0][0] for c in memory._graph.backend.set_properties.call_args_list]
        assert "a" not in ids_boosted
        assert "b" in ids_boosted

    def test_get_all_edges_failure_does_not_raise(self):
        """If get_all_edges raises, evolve() catches and returns cleanly."""
        evolver = HebbianCoRetrievalEvolver()
        backend = MagicMock()
        backend.get_all_edges.side_effect = RuntimeError("db error")
        graph = MagicMock()
        graph.backend = backend
        memory = MagicMock()
        memory._graph = graph

        evolver.evolve(memory)  # must not raise

        backend.set_properties.assert_not_called()

    def test_empty_edges_no_updates(self):
        """No qualifying edges → no set_properties calls."""
        evolver = HebbianCoRetrievalEvolver()
        memory = _make_memory(edges=[])

        evolver.evolve(memory)

        memory._graph.backend.set_properties.assert_not_called()


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
