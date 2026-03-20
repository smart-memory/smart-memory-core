"""Evaluation runner — orchestrates dataset loading, judging, and metric computation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartmemory.evaluation.dataset import load_interactions, group_sessions
from smartmemory.evaluation.judge import RelevanceJudge
from smartmemory.evaluation.metrics import (
    relevance_at_k,
    hit_rate,
    repetition_rate,
    mean_latency,
    redundancy_rate,
)


def _eval_data_dir() -> Path:
    """Resolve EVAL_DATA_DIR at call time so env var changes take effect."""
    raw = os.environ.get("EVAL_DATA_DIR", "~/.smartmemory/eval")
    return Path(os.path.expanduser(raw))


def _interactions_file() -> Path:
    return _eval_data_dir() / "interactions.jsonl"


def _runs_dir() -> Path:
    return _eval_data_dir() / "runs"


@dataclass
class EvalResult:
    """Result of an evaluation run."""

    timestamp: str
    period_days: int
    interaction_count: int
    session_count: int
    relevance_at_1: float
    relevance_at_5: float
    hit_rate: float
    repetition_rate_7d: float
    redundancy_rate: float
    mean_latency_ms: float
    judgments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, runs_dir: Path | None = None) -> Path:
        """Save result to a timestamped JSON file."""
        runs_dir = runs_dir or _runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"eval_{self.timestamp.replace(':', '-').replace('+', '_')}.json"
        path = runs_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Path) -> EvalResult:
        """Load a result from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class EvalRunner:
    """Orchestrates evaluation: load data, judge, compute metrics."""

    def __init__(
        self,
        interactions_path: Path | None = None,
        judge_model: str | None = None,
        judge_api_key: str | None = None,
        judge_base_url: str | None = None,
    ) -> None:
        self.interactions_path = interactions_path or _interactions_file()
        self.judge = RelevanceJudge(
            model=judge_model,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )

    def run(
        self,
        days: int = 7,
        since: datetime | None = None,
        session_gap_minutes: int = 30,
    ) -> EvalResult:
        """Run a full evaluation.

        Args:
            days: Number of days to evaluate (default 7).
            since: Start date override (takes precedence over days).
            session_gap_minutes: Gap in minutes to split sessions.

        Returns:
            EvalResult with all computed metrics.
        """
        interactions = load_interactions(self.interactions_path, since=since, days=days)

        if not interactions:
            return EvalResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                period_days=days,
                interaction_count=0,
                session_count=0,
                relevance_at_1=0.0,
                relevance_at_5=0.0,
                hit_rate=0.0,
                repetition_rate_7d=0.0,
                redundancy_rate=0.0,
                mean_latency_ms=0.0,
            )

        sessions = group_sessions(interactions, gap_minutes=session_gap_minutes)

        # Judge sessions
        judgments = self.judge.judge_sessions(sessions)

        # Compute metrics
        result = EvalResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            period_days=days,
            interaction_count=len(interactions),
            session_count=len(sessions),
            relevance_at_1=round(relevance_at_k(judgments), 2),
            relevance_at_5=round(relevance_at_k(judgments), 2),
            hit_rate=round(hit_rate(judgments), 2),
            repetition_rate_7d=round(repetition_rate(interactions), 2),
            redundancy_rate=round(redundancy_rate(judgments), 2),
            mean_latency_ms=round(mean_latency(interactions), 1),
            judgments=[
                {
                    "session_index": j.session_index,
                    "mean_score": round(j.mean_score, 2),
                    "scores": [{"qi": s.query_index, "s": s.score, "r": s.redundant} for s in j.scores],
                }
                for j in judgments
            ],
        )

        return result
