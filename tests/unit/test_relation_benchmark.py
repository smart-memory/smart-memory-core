"""Tests for smartmemory/relations/benchmark.py."""

from smartmemory.relations.benchmark import RelationQualityBenchmark, RelationBenchmarkResult
from smartmemory.relations.normalizer import RelationNormalizer


class TestRelationBenchmark:
    def setup_method(self):
        self.normalizer = RelationNormalizer()
        self.benchmark = RelationQualityBenchmark(self.normalizer)

    def test_empty_dataset(self):
        result = self.benchmark.evaluate([])
        assert isinstance(result, RelationBenchmarkResult)
        assert result.total == 0
        assert result.correct == 0
        assert result.f1 == 0.0

    def test_correct_match(self):
        dataset = [("Alice works at Acme", "Alice", "works_at", "Acme")]
        extracted = [[("Alice", "works_at", "Acme")]]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.correct == 1
        assert result.missing == 0

    def test_incorrect_type(self):
        dataset = [("Alice works at Acme", "Alice", "works_at", "Acme")]
        extracted = [[("Alice", "located_in", "Acme")]]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.incorrect_type == 1

    def test_incorrect_direction(self):
        dataset = [("Alice works at Acme", "Alice", "works_at", "Acme")]
        extracted = [[("Acme", "works_at", "Alice")]]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.incorrect_direction == 1

    def test_spurious(self):
        dataset = [("Alice works at Acme", "Alice", "works_at", "Acme")]
        extracted = [[("Alice", "works_at", "Acme"), ("Alice", "founded", "Acme")]]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.correct == 1
        assert result.spurious == 1

    def test_missing(self):
        dataset = [("Alice works at Acme", "Alice", "works_at", "Acme")]
        extracted = [[]]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.missing == 1

    def test_precision_recall_f1(self):
        dataset = [
            ("Alice works at Acme", "Alice", "works_at", "Acme"),
            ("Bob lives in NYC", "Bob", "lives_in", "NYC"),
        ]
        extracted = [
            [("Alice", "works_at", "Acme")],
            [("Bob", "lives_in", "NYC")],
        ]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_by_type_breakdown(self):
        dataset = [
            ("Alice works at Acme", "Alice", "works_at", "Acme"),
            ("Alice works at BigCo", "Alice", "works_at", "BigCo"),
        ]
        extracted = [
            [("Alice", "works_at", "Acme")],
            [("Alice", "located_in", "BigCo")],
        ]
        result = self.benchmark.evaluate(dataset, extracted)
        assert "works_at" in result.by_type
        assert result.by_type["works_at"]["correct"] == 1
        assert result.by_type["works_at"]["incorrect_type"] == 1

    def test_alias_normalization_in_expected(self):
        """Expected predicate is an alias — should normalize to canonical before matching."""
        dataset = [("Alice works at Acme", "Alice", "employed_by", "Acme")]
        extracted = [[("Alice", "works_at", "Acme")]]
        result = self.benchmark.evaluate(dataset, extracted)
        assert result.correct == 1

    def test_no_extracted_counts_as_missing(self):
        dataset = [("Alice works at Acme", "Alice", "works_at", "Acme")]
        result = self.benchmark.evaluate(dataset, extracted=None)
        assert result.missing == 1
