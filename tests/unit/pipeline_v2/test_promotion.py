"""Unit tests for PromotionEvaluator and OntologyConstrainStage promotion flow."""

import pytest


pytestmark = pytest.mark.unit

from smartmemory.ontology.promotion import (
    COMMON_WORD_BLOCKLIST,
    PromotionCandidate,
    PromotionEvaluator,
)
from smartmemory.pipeline.config import PromotionConfig

# Reuse the extended mock from Step 1 tests
from tests.unit.pipeline_v2.test_ontology_graph_extended import ExtendedMockBackend
from smartmemory.graph.ontology_graph import OntologyGraph


@pytest.fixture
def backend():
    return ExtendedMockBackend()


@pytest.fixture
def graph(backend):
    return OntologyGraph(workspace_id="test", backend=backend)


@pytest.fixture
def config():
    return PromotionConfig(
        min_frequency=2,
        min_confidence=0.7,
        min_type_consistency=0.8,
        min_name_length=3,
        reasoning_validation=False,
    )


@pytest.fixture
def evaluator(graph, config):
    return PromotionEvaluator(graph, config)


# ------------------------------------------------------------------ #
# Gate 1: min_name_length
# ------------------------------------------------------------------ #


def test_rejects_short_name(evaluator):
    candidate = PromotionCandidate("ab", "Person", 0.9)
    result = evaluator.evaluate(candidate)
    assert result.promoted is False
    assert "too short" in result.reason


def test_accepts_valid_length_name(evaluator, graph):
    # Must pass frequency gate too
    graph.increment_frequency("Person", 0.9)
    graph.increment_frequency("Person", 0.9)
    candidate = PromotionCandidate("Alice", "Person", 0.9)
    result = evaluator.evaluate(candidate)
    assert result.promoted is True


# ------------------------------------------------------------------ #
# Gate 2: common word blocklist
# ------------------------------------------------------------------ #


def test_rejects_common_word(evaluator):
    candidate = PromotionCandidate("this", "Concept", 0.9)
    result = evaluator.evaluate(candidate)
    assert result.promoted is False
    assert "common word" in result.reason


def test_blocklist_has_expected_words():
    assert "the" in COMMON_WORD_BLOCKLIST
    assert "something" in COMMON_WORD_BLOCKLIST
    assert "Python" not in COMMON_WORD_BLOCKLIST  # proper nouns not in list


# ------------------------------------------------------------------ #
# Gate 3: min_confidence
# ------------------------------------------------------------------ #


def test_rejects_low_confidence(evaluator):
    candidate = PromotionCandidate("FastAPI", "Technology", 0.3)
    result = evaluator.evaluate(candidate)
    assert result.promoted is False
    assert "Confidence" in result.reason


# ------------------------------------------------------------------ #
# Gate 4: min_frequency
# ------------------------------------------------------------------ #


def test_rejects_insufficient_frequency(evaluator, graph):
    # Only 1 occurrence, threshold is 2
    graph.increment_frequency("Technology", 0.9)
    candidate = PromotionCandidate("FastAPI", "Technology", 0.9)
    result = evaluator.evaluate(candidate)
    assert result.promoted is False
    assert "Frequency" in result.reason


def test_passes_with_sufficient_frequency(evaluator, graph):
    graph.increment_frequency("Technology", 0.9)
    graph.increment_frequency("Technology", 0.8)
    candidate = PromotionCandidate("FastAPI", "Technology", 0.85)
    result = evaluator.evaluate(candidate)
    assert result.promoted is True


# ------------------------------------------------------------------ #
# Gate 5: min_type_consistency
# ------------------------------------------------------------------ #


def test_rejects_inconsistent_type_assignment(evaluator, graph):
    graph.increment_frequency("Technology", 0.9)
    graph.increment_frequency("Technology", 0.9)
    # Add conflicting patterns: same name mapped to different types
    graph.add_entity_pattern("react", "Technology", 0.9)
    graph.add_entity_pattern("react", "Organization", 0.8)

    candidate = PromotionCandidate("react", "Technology", 0.9)
    result = evaluator.evaluate(candidate)
    # count is 1 for each, total 2, consistency = 0.5 < 0.8
    assert result.promoted is False
    assert "consistency" in result.reason


# ------------------------------------------------------------------ #
# Promote action
# ------------------------------------------------------------------ #


def test_promote_creates_entity_pattern(evaluator, graph, backend):
    candidate = PromotionCandidate("FastAPI", "Technology", 0.9)
    evaluator.promote(candidate)

    assert ("fastapi", "Technology") in backend._patterns
    pattern = backend._patterns[("fastapi", "Technology")]
    assert pattern["source"] == "promoted"


def test_promote_sets_type_to_confirmed(evaluator, graph, backend):
    graph.add_provisional("Technology")
    candidate = PromotionCandidate("FastAPI", "Technology", 0.9)
    evaluator.promote(candidate)

    assert backend._types["Technology"]["status"] == "confirmed"


# ------------------------------------------------------------------ #
# OntologyConstrainStage integration
# ------------------------------------------------------------------ #


def test_constrain_stage_tracks_frequency(backend):
    """The constrain stage should increment_frequency for accepted entities."""
    from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage
    from smartmemory.pipeline.state import PipelineState
    from smartmemory.pipeline.config import PipelineConfig

    graph = OntologyGraph(workspace_id="test", backend=backend)
    graph.seed_types()

    stage = OntologyConstrainStage(graph)
    state = PipelineState(
        ruler_entities=[{"name": "Python", "entity_type": "Technology", "confidence": 0.9, "source": "ruler"}],
        llm_entities=[],
        llm_relations=[],
    )
    config = PipelineConfig()

    stage.execute(state, config)

    # Should have incremented frequency for Technology
    assert backend._types["Technology"]["frequency"] >= 1


def test_constrain_stage_enqueues_to_promotion_queue(backend):
    """With a promotion queue, candidates are enqueued instead of directly promoted."""
    from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage
    from smartmemory.pipeline.state import PipelineState
    from smartmemory.pipeline.config import PipelineConfig
    from unittest.mock import MagicMock

    graph = OntologyGraph(workspace_id="test", backend=backend)
    mock_queue = MagicMock()

    stage = OntologyConstrainStage(graph, promotion_queue=mock_queue)
    state = PipelineState(
        ruler_entities=[],
        llm_entities=[{"name": "FastAPI", "entity_type": "WebFramework", "confidence": 0.8, "source": "llm"}],
        llm_relations=[],
    )
    config = PipelineConfig()

    stage.execute(state, config)

    mock_queue.enqueue.assert_called_once()
    payload = mock_queue.enqueue.call_args[0][0]
    assert payload["entity_name"] == "FastAPI"
    assert payload["entity_type"] == "WebFramework"
