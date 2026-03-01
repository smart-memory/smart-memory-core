"""Grounding benchmark harness.

Evaluates grounding accuracy against a labeled dataset of
(entity_mention, expected_qid) pairs. Reports hit rate,
disambiguation accuracy, and false positive rate by domain.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from smartmemory.grounding.store import PublicKnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a grounding benchmark evaluation."""

    total: int = 0
    hits: int = 0
    misses: int = 0
    false_positives: int = 0
    hit_rate: float = 0.0
    disambiguation_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    by_domain: dict[str, dict] = field(default_factory=dict)


class GroundingBenchmark:
    """Evaluate grounding accuracy against a labeled dataset.

    Usage:
        store = SQLitePublicKnowledgeStore("snapshot.sqlite")
        benchmark = GroundingBenchmark(store)
        result = benchmark.evaluate([
            ("Python", "Q28865"),
            ("Django", "Q2622004"),
            ("Nonexistent", ""),
        ])
        print(f"Hit rate: {result.hit_rate:.1%}")
    """

    def __init__(self, store: PublicKnowledgeStore):
        self.store = store

    def evaluate(self, dataset: list[tuple[str, str]]) -> BenchmarkResult:
        """Evaluate grounding against a labeled dataset.

        Args:
            dataset: List of (entity_mention, expected_qid) pairs.
                Empty string for expected_qid means the mention should NOT be grounded.

        Returns:
            BenchmarkResult with aggregate and per-domain metrics.
        """
        total = len(dataset)
        hits = 0
        misses = 0
        false_positives = 0
        domain_stats: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "hits": 0, "misses": 0, "false_positives": 0}
        )

        for mention, expected_qid in dataset:
            candidates = self.store.lookup_by_alias(mention)

            # Determine domain for per-domain breakdown
            domain = "unknown"
            if candidates:
                domain = candidates[0].domain or "unknown"

            domain_stats[domain]["total"] += 1

            if not expected_qid:
                # Should NOT be grounded
                if candidates:
                    false_positives += 1
                    domain_stats[domain]["false_positives"] += 1
                continue

            # Should be grounded to expected_qid
            matched = any(c.qid == expected_qid for c in candidates)
            if matched:
                hits += 1
                domain_stats[domain]["hits"] += 1
            else:
                misses += 1
                domain_stats[domain]["misses"] += 1

        # Compute rates
        groundable = hits + misses  # items that should have been grounded
        hit_rate = hits / groundable if groundable > 0 else 0.0
        disambiguation_accuracy = hits / total if total > 0 else 0.0
        fp_denominator = false_positives + hits + misses
        false_positive_rate = false_positives / fp_denominator if fp_denominator > 0 else 0.0

        # Per-domain rates
        by_domain = {}
        for domain, stats in domain_stats.items():
            d_groundable = stats["hits"] + stats["misses"]
            by_domain[domain] = {
                "total": stats["total"],
                "hits": stats["hits"],
                "misses": stats["misses"],
                "false_positives": stats["false_positives"],
                "hit_rate": stats["hits"] / d_groundable if d_groundable > 0 else 0.0,
            }

        return BenchmarkResult(
            total=total,
            hits=hits,
            misses=misses,
            false_positives=false_positives,
            hit_rate=hit_rate,
            disambiguation_accuracy=disambiguation_accuracy,
            false_positive_rate=false_positive_rate,
            by_domain=by_domain,
        )
