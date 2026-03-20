"""Report formatting and run comparison."""

from __future__ import annotations

import json
from pathlib import Path

from smartmemory.evaluation.runner import EvalResult, _runs_dir


def format_report(result: EvalResult, previous: EvalResult | None = None) -> str:
    """Format an evaluation result as a human-readable report.

    Args:
        result: Current evaluation result.
        previous: Optional previous result for comparison.

    Returns:
        Formatted report string.
    """
    lines = [
        f"SmartMemory Evaluation — {result.timestamp[:10]}",
        f"Period: {result.period_days} days | Interactions: {result.interaction_count} | Sessions: {result.session_count}",
        "",
    ]

    if result.interaction_count == 0:
        lines.append("  No interactions found for this period.")
        lines.append("  Enable logging: EVAL_LOGGING=true")
        return "\n".join(lines)

    lines.extend([
        f"  Relevance@1:     {result.relevance_at_1:.1f} / 3.0",
        f"  Relevance@5:     {result.relevance_at_5:.1f} / 3.0",
        f"  Hit rate:        {result.hit_rate:.0%}",
        f"  Repetition (7d): {result.repetition_rate_7d:.0%}",
        f"  Redundancy:      {result.redundancy_rate:.0%}",
        f"  Mean latency:    {result.mean_latency_ms:.0f}ms",
    ])

    if previous and previous.interaction_count > 0:
        lines.extend([
            "",
            f"  vs previous run ({previous.timestamp[:10]}):",
        ])

        def _delta(current: float, prev: float, pct: bool = False, lower_is_better: bool = False) -> str:
            diff = current - prev
            if pct:
                sign = "+" if diff > 0 else ""
                arrow = _arrow(diff, lower_is_better)
                return f"{sign}{diff:.0%}  {arrow}"
            sign = "+" if diff > 0 else ""
            arrow = _arrow(diff, lower_is_better)
            return f"{sign}{diff:.1f}  {arrow}"

        def _arrow(diff: float, lower_is_better: bool) -> str:
            if abs(diff) < 0.01:
                return "—"
            improved = (diff < 0) if lower_is_better else (diff > 0)
            return "↑" if improved else "↓"

        lines.extend([
            f"  Relevance@1:     {_delta(result.relevance_at_1, previous.relevance_at_1)}",
            f"  Hit rate:        {_delta(result.hit_rate, previous.hit_rate, pct=True)}",
            f"  Repetition:      {_delta(result.repetition_rate_7d, previous.repetition_rate_7d, pct=True, lower_is_better=True)}",
            f"  Mean latency:    {_delta(result.mean_latency_ms, previous.mean_latency_ms, lower_is_better=True)}",
        ])

    return "\n".join(lines)


def load_latest_runs(runs_dir: Path | None = None, count: int = 2) -> list[EvalResult]:
    """Load the most recent evaluation runs.

    Args:
        runs_dir: Directory containing run JSON files.
        count: Number of runs to load.

    Returns:
        List of EvalResult objects, newest first.
    """
    runs_dir = runs_dir or _runs_dir()
    if not runs_dir.exists():
        return []

    files = sorted(runs_dir.glob("eval_*.json"), reverse=True)
    results: list[EvalResult] = []
    for f in files[:count]:
        try:
            results.append(EvalResult.load(f))
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def export_judged_interactions(
    result: EvalResult,
    output_path: Path,
) -> int:
    """Export evaluation judgments as JSONL.

    Args:
        result: Evaluation result with judgments.
        output_path: Path to write JSONL output.

    Returns:
        Number of records written.
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for judgment in result.judgments:
            f.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            count += 1
    return count
