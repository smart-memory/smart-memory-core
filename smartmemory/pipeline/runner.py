"""PipelineRunner — orchestrates StageCommands through a Transport.

Supports full run, breakpoints (run_to), resumption (run_from),
rollback (undo_to), and per-stage retry with configurable failure modes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from smartmemory.observability.tracing import trace_span
from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.protocol import StageCommand
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.token_tracker import PipelineTokenTracker
from smartmemory.pipeline.transport import InProcessTransport, Transport

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute an ordered list of StageCommands."""

    def __init__(
        self,
        stages: List[StageCommand],
        transport: Optional[Transport] = None,
    ):
        self.stages = stages
        self.transport: Transport = transport or InProcessTransport()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        text: str,
        config: PipelineConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineState:
        """Execute the full pipeline."""
        meta = metadata or {}
        tracker = PipelineTokenTracker(
            workspace_id=config.workspace_id,
            profile_name=meta.get("profile_name"),
        )
        state = PipelineState(
            text=text,
            raw_metadata=meta,
            mode=config.mode,
            workspace_id=config.workspace_id,
            started_at=datetime.now(timezone.utc),
            token_tracker=tracker,
            memory_type=meta.get("memory_type") or None,
        )
        return self._run_stages(state, config, self.stages)

    def run_to(
        self,
        text: str,
        config: PipelineConfig,
        stop_after: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineState:
        """Execute until (and including) the named stage, then stop."""
        meta = metadata or {}
        tracker = PipelineTokenTracker(
            workspace_id=config.workspace_id,
            profile_name=meta.get("profile_name"),
        )
        state = PipelineState(
            text=text,
            raw_metadata=meta,
            mode=config.mode,
            workspace_id=config.workspace_id,
            started_at=datetime.now(timezone.utc),
            token_tracker=tracker,
            memory_type=meta.get("memory_type") or None,
        )
        stages = self._stages_up_to(stop_after)
        return self._run_stages(state, config, stages)

    def run_from(
        self,
        state: PipelineState,
        config: PipelineConfig,
        start_from: Optional[str] = None,
        stop_after: Optional[str] = None,
    ) -> PipelineState:
        """Resume from a checkpoint.

        If *start_from* is ``None``, auto-detects the next stage from
        ``state.stage_history``.
        """
        if start_from is None:
            start_from = self._next_stage_name(state)
            if start_from is None:
                return state  # all stages already completed

        # Ensure a live tracker exists (deserialized checkpoints have None).
        if not isinstance(state.token_tracker, PipelineTokenTracker):
            tracker = PipelineTokenTracker(
                workspace_id=config.workspace_id,
            )
            state = replace(state, token_tracker=tracker)

        stages = self._stages_from(start_from, stop_after)
        return self._run_stages(state, config, stages)

    def undo_to(self, state: PipelineState, target: str) -> PipelineState:
        """Roll back stages in reverse until *target* (exclusive)."""
        history = list(state.stage_history)
        if target not in history:
            raise ValueError(f"Stage '{target}' not in history: {history}")

        target_idx = history.index(target)
        to_undo = history[target_idx + 1 :]

        stage_map = {s.name: s for s in self.stages}
        for stage_name in reversed(to_undo):
            stage = stage_map.get(stage_name)
            if stage is None:
                logger.warning("Cannot undo unknown stage '%s'", stage_name)
                continue
            state = stage.undo(state)
            state = replace(state, stage_history=[h for h in state.stage_history if h != stage_name])

        return state

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_stages(
        self,
        state: PipelineState,
        config: PipelineConfig,
        stages: List[StageCommand],
    ) -> PipelineState:
        for stage in stages:
            state = self._execute_stage(stage, state, config)
        state = replace(
            state,
            completed_at=datetime.now(timezone.utc),
        )
        return state

    def _execute_stage(
        self,
        stage: StageCommand,
        state: PipelineState,
        config: PipelineConfig,
    ) -> PipelineState:
        """Execute a single stage with retry logic.

        Uses per-stage retry policy from ``config.stage_retry_policies`` if set,
        otherwise falls back to the global ``config.retry``.
        """
        # Resolve retry parameters: per-stage policy takes precedence
        stage_policy = config.stage_retry_policies.get(stage.name)
        if stage_policy:
            max_retries = stage_policy.max_retries
            backoff = stage_policy.backoff_seconds
            backoff_multiplier = stage_policy.backoff_multiplier
            on_failure = stage_policy.on_failure
        else:
            max_retries = config.retry.max_retries
            backoff = config.retry.backoff_seconds
            backoff_multiplier = 2.0
            on_failure = config.retry.on_failure

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                span_attrs = {
                    "memory_id": getattr(state, "item_id", None) or state.raw_metadata.get("run_id", ""),
                    "attempt": attempt + 1,
                }
                with trace_span(f"pipeline.{stage.name}", span_attrs) as span:
                    t0 = time.perf_counter()
                    new_state = self.transport.execute(stage, state, config)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    span.attributes["transport_ms"] = elapsed
                    span.attributes["entity_count"] = len(getattr(new_state, "entities", []) or [])
                    span.attributes["relation_count"] = len(getattr(new_state, "relations", []) or [])

                new_state = replace(
                    new_state,
                    stage_history=[*new_state.stage_history, stage.name],
                    stage_timings={**new_state.stage_timings, stage.name: elapsed},
                )
                return new_state
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "Stage '%s' failed (attempt %d/%d): %s — retrying in %.1fs",
                        stage.name,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= backoff_multiplier

        # All retries exhausted — undo partial work
        try:
            state = stage.undo(state)
        except Exception as undo_exc:
            logger.warning("Undo for stage '%s' failed: %s", stage.name, undo_exc)

        if on_failure == "skip":
            logger.warning("Stage '%s' failed after %d attempts — skipping", stage.name, max_retries + 1)
            return replace(
                state,
                stage_history=[*state.stage_history, f"{stage.name}:skipped"],
            )

        raise RuntimeError(f"Stage '{stage.name}' failed after {max_retries + 1} attempts") from last_exc

    # ------------------------------------------------------------------ #
    # Stage selection helpers
    # ------------------------------------------------------------------ #

    def _stages_up_to(self, stop_after: str) -> List[StageCommand]:
        result = []
        for s in self.stages:
            result.append(s)
            if s.name == stop_after:
                return result
        raise ValueError(f"Stage '{stop_after}' not found in pipeline stages")

    def _stages_from(self, start_from: str, stop_after: Optional[str] = None) -> List[StageCommand]:
        result = []
        started = False
        for s in self.stages:
            if s.name == start_from:
                started = True
            if started:
                result.append(s)
            if stop_after and s.name == stop_after:
                break
        if not started:
            raise ValueError(f"Stage '{start_from}' not found in pipeline stages")
        return result

    def _next_stage_name(self, state: PipelineState) -> Optional[str]:
        """Determine the next stage to run based on history."""
        completed = set(state.stage_history)
        for stage in self.stages:
            if stage.name not in completed:
                return stage.name
        return None
