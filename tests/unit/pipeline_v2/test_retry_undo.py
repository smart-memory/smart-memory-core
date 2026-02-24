"""Unit tests for per-stage retry policies with undo on failure."""

import dataclasses
import time

import pytest


pytestmark = pytest.mark.unit

from smartmemory.pipeline.config import PipelineConfig, RetryConfig, StageRetryPolicy
from smartmemory.pipeline.runner import PipelineRunner


# ------------------------------------------------------------------ #
# Test helpers
# ------------------------------------------------------------------ #


class MockStage:
    """Stage that optionally fails N times before succeeding."""

    def __init__(self, name, fail_count=0):
        self._name = name
        self._fail_count = fail_count
        self._attempts = 0
        self.executed = False
        self.undone = False

    @property
    def name(self):
        return self._name

    def execute(self, state, config):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise RuntimeError(f"{self._name} transient failure #{self._attempts}")
        self.executed = True
        return dataclasses.replace(state)

    def undo(self, state):
        self.undone = True
        return dataclasses.replace(state)

    @property
    def attempts(self):
        return self._attempts


class AlwaysFailStage:
    """Stage that always fails."""

    def __init__(self, name):
        self._name = name
        self.undone = False

    @property
    def name(self):
        return self._name

    def execute(self, state, config):
        raise RuntimeError(f"{self._name} permanent failure")

    def undo(self, state):
        self.undone = True
        return dataclasses.replace(state)


# ------------------------------------------------------------------ #
# StageRetryPolicy dataclass tests
# ------------------------------------------------------------------ #


class TestStageRetryPolicy:
    def test_default_values(self):
        policy = StageRetryPolicy()
        assert policy.max_retries == 2
        assert policy.backoff_seconds == 1.0
        assert policy.backoff_multiplier == 2.0
        assert policy.on_failure == "abort"

    def test_custom_values(self):
        policy = StageRetryPolicy(max_retries=5, backoff_seconds=0.5, backoff_multiplier=3.0, on_failure="skip")
        assert policy.max_retries == 5
        assert policy.on_failure == "skip"

    def test_invalid_on_failure(self):
        with pytest.raises(ValueError, match="on_failure"):
            StageRetryPolicy(on_failure="crash")


class TestPipelineConfigStageRetryPolicies:
    def test_empty_by_default(self):
        config = PipelineConfig()
        assert config.stage_retry_policies == {}

    def test_set_per_stage(self):
        config = PipelineConfig(
            stage_retry_policies={
                "extract": StageRetryPolicy(max_retries=5),
                "store": StageRetryPolicy(on_failure="skip"),
            }
        )
        assert config.stage_retry_policies["extract"].max_retries == 5
        assert config.stage_retry_policies["store"].on_failure == "skip"


# ------------------------------------------------------------------ #
# Retry succeeds on transient failure
# ------------------------------------------------------------------ #


class TestRetrySuccess:
    def test_retry_succeeds_after_transient_failure(self):
        """Stage fails once then succeeds — should complete normally."""
        stage = MockStage("extract", fail_count=1)
        runner = PipelineRunner(stages=[stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=2, backoff_seconds=0.01))

        state = runner.run("test", config)

        assert stage.executed
        assert stage.attempts == 2
        assert "extract" in state.stage_history
        assert not stage.undone

    def test_per_stage_retry_overrides_global(self):
        """Per-stage policy with higher retries allows more attempts."""
        stage = MockStage("extract", fail_count=3)
        runner = PipelineRunner(stages=[stage])
        config = PipelineConfig(
            retry=RetryConfig(max_retries=1, backoff_seconds=0.01),
            stage_retry_policies={
                "extract": StageRetryPolicy(max_retries=5, backoff_seconds=0.01),
            },
        )

        state = runner.run("test", config)

        assert stage.executed
        assert stage.attempts == 4  # 3 failures + 1 success


# ------------------------------------------------------------------ #
# Undo called on permanent failure
# ------------------------------------------------------------------ #


class TestUndoOnFailure:
    def test_undo_called_when_retries_exhausted(self):
        """When all retries are exhausted, undo is called before raising."""
        stage = AlwaysFailStage("extract")
        runner = PipelineRunner(stages=[stage])
        config = PipelineConfig(retry=RetryConfig(max_retries=1, backoff_seconds=0.01))

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            runner.run("test", config)

        assert stage.undone

    def test_undo_called_with_per_stage_policy(self):
        """Per-stage policy also triggers undo on exhaustion."""
        stage = AlwaysFailStage("store")
        runner = PipelineRunner(stages=[stage])
        config = PipelineConfig(
            stage_retry_policies={
                "store": StageRetryPolicy(max_retries=0, backoff_seconds=0.01),
            },
        )

        with pytest.raises(RuntimeError, match="failed after 1 attempts"):
            runner.run("test", config)

        assert stage.undone


# ------------------------------------------------------------------ #
# Fallback to skip
# ------------------------------------------------------------------ #


class TestFallbackSkip:
    def test_skip_on_failure(self):
        """When on_failure='skip', stage failure is recorded but pipeline continues."""
        fail_stage = AlwaysFailStage("enrich")
        ok_stage = MockStage("evolve")
        runner = PipelineRunner(stages=[fail_stage, ok_stage])
        config = PipelineConfig(
            retry=RetryConfig(max_retries=0, backoff_seconds=0.01, on_failure="skip"),
        )

        state = runner.run("test", config)

        assert "enrich:skipped" in state.stage_history
        assert ok_stage.executed
        assert "evolve" in state.stage_history

    def test_skip_with_per_stage_policy(self):
        """Per-stage skip policy allows pipeline to continue past failure."""
        fail_stage = AlwaysFailStage("enrich")
        ok_stage = MockStage("evolve")
        runner = PipelineRunner(stages=[fail_stage, ok_stage])
        config = PipelineConfig(
            retry=RetryConfig(max_retries=2, backoff_seconds=0.01),
            stage_retry_policies={
                "enrich": StageRetryPolicy(max_retries=0, backoff_seconds=0.01, on_failure="skip"),
            },
        )

        state = runner.run("test", config)

        assert "enrich:skipped" in state.stage_history
        assert ok_stage.executed


# ------------------------------------------------------------------ #
# Backoff timing
# ------------------------------------------------------------------ #


class TestBackoffTiming:
    def test_backoff_multiplier_applied(self):
        """Per-stage backoff_multiplier is applied between retries."""
        stage = MockStage("extract", fail_count=2)
        runner = PipelineRunner(stages=[stage])
        config = PipelineConfig(
            stage_retry_policies={
                "extract": StageRetryPolicy(
                    max_retries=3,
                    backoff_seconds=0.01,
                    backoff_multiplier=2.0,
                ),
            },
        )

        t0 = time.perf_counter()
        state = runner.run("test", config)
        elapsed = time.perf_counter() - t0

        assert stage.executed
        # Should have waited ~0.01 + ~0.02 = ~0.03s (with multiplier)
        assert elapsed >= 0.02
        assert elapsed < 1.0  # sanity check — shouldn't take long
