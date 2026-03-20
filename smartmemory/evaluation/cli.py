"""CLI for SmartMemory evaluation.

Usage:
    sm-eval run                          # Evaluate last 7 days
    sm-eval run --days 30                # Evaluate last 30 days
    sm-eval run --since 2026-03-01       # Evaluate from specific date
    sm-eval report                       # Show last run or compare last two
    sm-eval export --format jsonl        # Export judged interactions
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from smartmemory.evaluation.runner import EvalRunner
from smartmemory.evaluation.report import format_report, load_latest_runs, export_judged_interactions


def cmd_run(args: argparse.Namespace) -> None:
    """Run evaluation and print report."""
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    runner = EvalRunner(
        interactions_path=Path(args.data) if args.data else None,
        judge_model=args.model,
    )

    if since:
        print(f"Running evaluation (since {args.since})...")
    else:
        print(f"Running evaluation (last {args.days} days)...")
    result = runner.run(days=args.days, since=since)

    # Save run
    path = result.save()
    print(f"Run saved: {path}\n")

    # Load previous for comparison
    runs = load_latest_runs()
    previous = runs[1] if len(runs) > 1 else None

    print(format_report(result, previous))


def cmd_report(args: argparse.Namespace) -> None:
    """Show most recent evaluation report, with comparison if available."""
    runs = load_latest_runs(count=2)
    if not runs:
        print("No evaluation runs found. Run `sm-eval run` first.")
        sys.exit(1)

    current = runs[0]
    previous = runs[1] if len(runs) > 1 else None
    print(format_report(current, previous))


def cmd_export(args: argparse.Namespace) -> None:
    """Export judged interactions."""
    runs = load_latest_runs(count=1)
    if not runs:
        print("No evaluation runs found. Run `sm-eval run` first.")
        sys.exit(1)

    output = Path(args.output) if args.output else Path("eval_export.jsonl")
    count = export_judged_interactions(runs[0], output)
    print(f"Exported {count} judgments to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sm-eval",
        description="SmartMemory evaluation — measure memory retrieval quality",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument("--days", type=int, default=7, help="Number of days to evaluate (default: 7)")
    run_parser.add_argument("--since", type=str, help="Start date (ISO format, e.g. 2026-03-01)")
    run_parser.add_argument("--data", type=str, help="Path to interactions.jsonl (default: ~/.smartmemory/eval/)")
    run_parser.add_argument("--model", type=str, help="Judge model (default: gpt-4o-mini)")

    # report
    subparsers.add_parser("report", help="Show most recent evaluation report")

    # export
    export_parser = subparsers.add_parser("export", help="Export judged interactions")
    export_parser.add_argument("--output", "-o", type=str, help="Output file path (default: eval_export.jsonl)")
    export_parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl"], help="Export format")

    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "report": cmd_report,
        "export": cmd_export,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
