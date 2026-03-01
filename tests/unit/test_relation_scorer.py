"""Tests for smartmemory/relations/scorer.py."""

from smartmemory.relations.scorer import PlausibilityScorer


class TestPlausibilityScorer:
    def test_perfect_scores(self):
        scorer = PlausibilityScorer()
        assert scorer.score(1.0, 1.0) == 1.0

    def test_zero_scores(self):
        scorer = PlausibilityScorer()
        assert scorer.score(0.0, 0.0) == 0.0

    def test_mixed_equal_weights(self):
        scorer = PlausibilityScorer(w_norm=0.5, w_pair=0.5)
        assert scorer.score(1.0, 0.0) == 0.5

    def test_custom_weights(self):
        scorer = PlausibilityScorer(w_norm=0.8, w_pair=0.2)
        result = scorer.score(1.0, 0.0)
        assert abs(result - 0.8) < 1e-9

    def test_output_clamped_low(self):
        scorer = PlausibilityScorer()
        result = scorer.score(-1.0, -1.0)
        assert result == 0.0

    def test_output_clamped_high(self):
        scorer = PlausibilityScorer()
        result = scorer.score(2.0, 2.0)
        assert result == 1.0

    def test_zero_weights(self):
        scorer = PlausibilityScorer(w_norm=0.0, w_pair=0.0)
        assert scorer.score(1.0, 1.0) == 0.0

    def test_asymmetric_inputs(self):
        scorer = PlausibilityScorer(w_norm=0.5, w_pair=0.5)
        result = scorer.score(0.8, 0.5)
        expected = (0.5 * 0.8 + 0.5 * 0.5) / 1.0
        assert abs(result - expected) < 1e-9
