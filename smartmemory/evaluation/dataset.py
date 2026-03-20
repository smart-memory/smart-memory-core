"""Load and group interaction logs into evaluation sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class InteractionLog:
    """A single search interaction from the JSONL log."""

    id: str
    ts: datetime
    query: str
    top_k: int
    memory_type: str | None
    decompose: bool
    result_count: int
    latency_ms: float
    results: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InteractionLog:
        ts_raw = d["ts"]
        if isinstance(ts_raw, str):
            # Handle both Z suffix and +00:00
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        else:
            ts = ts_raw

        return cls(
            id=d["id"],
            ts=ts,
            query=d["query"],
            top_k=d.get("top_k", 5),
            memory_type=d.get("memory_type"),
            decompose=d.get("decompose", False),
            result_count=d.get("result_count", 0),
            latency_ms=d.get("latency_ms", 0.0),
            results=d.get("results", []),
        )


@dataclass
class Session:
    """A group of interactions within 30 minutes of each other."""

    interactions: list[InteractionLog] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return self.interactions[0].ts

    @property
    def end(self) -> datetime:
        return self.interactions[-1].ts

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


def load_interactions(
    path: Path,
    since: datetime | None = None,
    days: int | None = None,
) -> list[InteractionLog]:
    """Load interactions from a JSONL file, optionally filtering by date.

    Args:
        path: Path to interactions.jsonl file.
        since: Only include interactions after this datetime.
        days: Only include interactions from the last N days (ignored if since is set).

    Returns:
        List of InteractionLog objects, sorted by timestamp.
    """
    if not path.exists():
        return []

    if since is None and days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days)

    interactions: list[InteractionLog] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                interaction = InteractionLog.from_dict(d)
                # Normalize naive timestamps to UTC for comparison
                ts = interaction.ts
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                    interaction.ts = ts
                if since and ts < since:
                    continue
                interactions.append(interaction)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # Skip malformed lines

    interactions.sort(key=lambda x: x.ts)
    return interactions


def group_sessions(
    interactions: list[InteractionLog],
    gap_minutes: int = 30,
) -> list[Session]:
    """Group interactions into sessions by time gap.

    Two interactions belong to the same session if they are within
    gap_minutes of each other.

    Args:
        interactions: Sorted list of interactions.
        gap_minutes: Maximum gap in minutes between interactions in the same session.

    Returns:
        List of Session objects.
    """
    if not interactions:
        return []

    gap = timedelta(minutes=gap_minutes)
    sessions: list[Session] = []
    current = Session(interactions=[interactions[0]])

    for interaction in interactions[1:]:
        if interaction.ts - current.interactions[-1].ts > gap:
            sessions.append(current)
            current = Session(interactions=[interaction])
        else:
            current.interactions.append(interaction)

    sessions.append(current)
    return sessions
