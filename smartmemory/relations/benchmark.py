"""Relation quality benchmark harness.

Evaluates relation extraction quality by comparing normalized canonical types
against expected predicates. Reports precision, recall, F1, and per-type breakdown.

Pattern: smartmemory/grounding/benchmark.py — BenchmarkResult + evaluate().
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from smartmemory.relations.normalizer import RelationNormalizer

logger = logging.getLogger(__name__)


@dataclass
class RelationBenchmarkResult:
    """Results from a relation quality benchmark evaluation."""

    total: int = 0
    correct: int = 0
    incorrect_type: int = 0
    incorrect_direction: int = 0
    spurious: int = 0
    missing: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    by_type: dict[str, dict] = field(default_factory=dict)
    type_pair_validity_rate: float = 0.0


class RelationQualityBenchmark:
    """Evaluate relation extraction quality against a labeled dataset.

    Usage:
        normalizer = RelationNormalizer()
        benchmark = RelationQualityBenchmark(normalizer)
        result = benchmark.evaluate([
            ("Python is a programming language", "Python", "type_of", "programming language"),
            ("Alice works at Acme", "Alice", "works_at", "Acme"),
        ])
        print(f"F1: {result.f1:.1%}")
    """

    def __init__(self, normalizer: RelationNormalizer):
        self._normalizer = normalizer

    def evaluate(
        self,
        dataset: list[tuple[str, str, str, str]],
        extracted: list[list[tuple[str, str, str]]] | None = None,
    ) -> RelationBenchmarkResult:
        """Evaluate relation extraction quality.

        Args:
            dataset: List of (text, expected_subject, expected_predicate, expected_object).
            extracted: Optional pre-extracted results. If None, only normalization is evaluated.
                       Each entry is a list of (subject, predicate, object) tuples for the
                       corresponding dataset item.

        Returns:
            RelationBenchmarkResult with aggregate and per-type metrics.
        """
        if not dataset:
            return RelationBenchmarkResult()

        total = len(dataset)
        correct = 0
        incorrect_type = 0
        incorrect_direction = 0
        spurious = 0
        missing = 0
        type_stats: dict[str, dict] = defaultdict(
            lambda: {"correct": 0, "incorrect_type": 0, "incorrect_direction": 0, "missing": 0}
        )

        for i, (_text, exp_subj, exp_pred, exp_obj) in enumerate(dataset):
            # Normalize expected predicate to canonical form
            expected_canonical, _ = self._normalizer.normalize(exp_pred)

            if extracted is not None and i < len(extracted):
                triples = extracted[i]
                matched = False
                for subj, pred, obj in triples:
                    actual_canonical, _ = self._normalizer.normalize(pred)

                    subj_match = subj.lower() == exp_subj.lower()
                    obj_match = obj.lower() == exp_obj.lower()

                    if subj_match and obj_match and actual_canonical == expected_canonical:
                        correct += 1
                        type_stats[expected_canonical]["correct"] += 1
                        matched = True
                        break
                    elif subj_match and obj_match and actual_canonical != expected_canonical:
                        incorrect_type += 1
                        type_stats[expected_canonical]["incorrect_type"] += 1
                        matched = True
                        break
                    elif (
                        subj.lower() == exp_obj.lower()
                        and obj.lower() == exp_subj.lower()
                        and actual_canonical == expected_canonical
                    ):
                        incorrect_direction += 1
                        type_stats[expected_canonical]["incorrect_direction"] += 1
                        matched = True
                        break

                if not matched:
                    # Check for spurious (extra relations) vs missing
                    if triples:
                        missing += 1
                        type_stats[expected_canonical]["missing"] += 1
                    else:
                        missing += 1
                        type_stats[expected_canonical]["missing"] += 1

                # Count spurious: triples that don't match any expected
                unmatched = len(triples) - (1 if matched else 0)
                if unmatched > 0:
                    spurious += unmatched
            else:
                # No extraction provided — count as missing
                missing += 1
                type_stats[expected_canonical]["missing"] += 1

        # Compute P/R/F1
        true_positives = correct
        extracted_total = correct + incorrect_type + incorrect_direction + spurious
        expected_total = correct + incorrect_type + incorrect_direction + missing

        precision = true_positives / extracted_total if extracted_total > 0 else 0.0
        recall = true_positives / expected_total if expected_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return RelationBenchmarkResult(
            total=total,
            correct=correct,
            incorrect_type=incorrect_type,
            incorrect_direction=incorrect_direction,
            spurious=spurious,
            missing=missing,
            precision=precision,
            recall=recall,
            f1=f1,
            by_type=dict(type_stats),
        )
