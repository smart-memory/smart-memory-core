"""Unit tests for ReasoningValidator."""

import json
import pytest


pytestmark = pytest.mark.unit

from smartmemory.ontology.reasoning_validator import ReasoningValidator
from smartmemory.ontology.promotion import PromotionCandidate


@pytest.fixture
def candidate():
    return PromotionCandidate(entity_name="FastAPI", entity_type="Technology", confidence=0.9)


@pytest.fixture
def stats():
    return {"frequency": 5, "assignments": [{"type": "Technology", "count": 4}, {"type": "Concept", "count": 1}]}


# ------------------------------------------------------------------ #
# LLM response parsing
# ------------------------------------------------------------------ #


def test_validate_accept_response(candidate, stats):
    def mock_lm(prompt):
        return json.dumps({"verdict": "accept", "explanation": "Well-known web framework"})

    validator = ReasoningValidator(lm=mock_lm)
    result = validator.validate(candidate, stats)

    assert result.is_valid is True
    assert result.verdict == "accept"
    assert "web framework" in result.explanation


def test_validate_reject_response(candidate, stats):
    def mock_lm(prompt):
        return json.dumps({"verdict": "reject", "explanation": "Too generic to be a type"})

    validator = ReasoningValidator(lm=mock_lm)
    result = validator.validate(candidate, stats)

    assert result.is_valid is False
    assert result.verdict == "reject"


def test_validate_malformed_json_fallback(candidate, stats):
    def mock_lm(prompt):
        return "I think we should reject this because it's unclear"

    validator = ReasoningValidator(lm=mock_lm)
    result = validator.validate(candidate, stats)

    assert result.verdict == "reject"  # "reject" keyword found


def test_validate_malformed_json_accept_fallback(candidate, stats):
    def mock_lm(prompt):
        return "This looks like a valid technology entity"

    validator = ReasoningValidator(lm=mock_lm)
    result = validator.validate(candidate, stats)

    assert result.verdict == "accept"  # no "reject" keyword


def test_validate_llm_failure_defaults_to_accept(candidate, stats):
    def mock_lm(prompt):
        raise RuntimeError("LLM is down")

    validator = ReasoningValidator(lm=mock_lm)
    result = validator.validate(candidate, stats)

    assert result.is_valid is True
    assert result.verdict == "accept"
    assert "error" in result.explanation.lower()


# ------------------------------------------------------------------ #
# Reasoning trace
# ------------------------------------------------------------------ #


def test_validate_produces_reasoning_trace(candidate, stats):
    def mock_lm(prompt):
        return json.dumps({"verdict": "accept", "explanation": "Valid type"})

    validator = ReasoningValidator(lm=mock_lm)
    result = validator.validate(candidate, stats)

    assert result.reasoning_trace is not None
    assert hasattr(result.reasoning_trace, "trace_id")
    assert len(result.reasoning_trace.steps) == 3


# ------------------------------------------------------------------ #
# Integration with PromotionEvaluator
# ------------------------------------------------------------------ #


def test_promotion_evaluator_calls_reasoning_validator():
    """When reasoning_validation=True, evaluator delegates to ReasoningValidator."""
    from smartmemory.ontology.promotion import PromotionEvaluator
    from smartmemory.pipeline.config import PromotionConfig
    from smartmemory.graph.ontology_graph import OntologyGraph
    from tests.unit.pipeline_v2.test_ontology_graph_extended import ExtendedMockBackend

    backend = ExtendedMockBackend()
    graph = OntologyGraph(workspace_id="test", backend=backend)

    # Meet all statistical gates
    graph.increment_frequency("Technology", 0.9)
    graph.increment_frequency("Technology", 0.85)

    config = PromotionConfig(
        min_frequency=2,
        min_confidence=0.7,
        min_name_length=3,
        reasoning_validation=True,
    )

    # Monkey-patch ReasoningValidator to use a mock LLM
    import smartmemory.ontology.reasoning_validator as rv_module
    original_init = rv_module.ReasoningValidator.__init__

    def patched_init(self, smart_memory=None, lm=None):
        original_init(self, smart_memory=smart_memory, lm=lambda p: json.dumps({"verdict": "reject", "explanation": "Not a valid type"}))

    rv_module.ReasoningValidator.__init__ = patched_init
    try:
        evaluator = PromotionEvaluator(graph, config)
        result = evaluator.evaluate(PromotionCandidate("FastAPI", "Technology", 0.9))

        assert result.promoted is False
        assert "Reasoning validation rejected" in result.reason
    finally:
        rv_module.ReasoningValidator.__init__ = original_init
