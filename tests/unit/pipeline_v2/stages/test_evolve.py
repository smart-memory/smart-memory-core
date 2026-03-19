"""Unit tests for EvolveStage."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig, EvolveConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.evolve import EvolveStage


class TestEvolveStage:
    """Tests for the evolve pipeline stage."""

    def _make_stage(self):
        """Build an EvolveStage with a mocked SmartMemory instance."""
        memory = MagicMock()
        memory._evolution = MagicMock()
        memory._clustering = MagicMock()
        # CORE-EVO-LIVE-1: ensure no evolution worker so batch path runs
        memory._evolution_worker = None
        return EvolveStage(memory), memory

    def test_evolve_preview_mode_noop(self):
        """In preview mode, state is returned unchanged."""
        stage, memory = self._make_stage()
        state = PipelineState(text="Preview evolution.")
        config = PipelineConfig.preview()

        result = stage.execute(state, config)

        assert result is state
        memory._evolution.run_evolution_cycle.assert_not_called()
        memory._clustering.run.assert_not_called()

    def test_evolve_runs_evolution(self):
        """When run_evolution is True, run_evolution_cycle is called."""
        stage, memory = self._make_stage()
        state = PipelineState(text="Evolve this.")
        config = PipelineConfig.default()
        config.evolve = EvolveConfig(run_evolution=True, run_clustering=False)

        result = stage.execute(state, config)

        memory._evolution.run_evolution_cycle.assert_called_once()
        memory._clustering.run.assert_not_called()
        assert result.evolutions["evolution"] == "completed"

    def test_evolve_runs_clustering(self):
        """When run_clustering is True, clustering.run is called with use_semhash=True."""
        stage, memory = self._make_stage()
        state = PipelineState(text="Cluster this.")
        config = PipelineConfig.default()
        config.evolve = EvolveConfig(run_evolution=False, run_clustering=True)

        result = stage.execute(state, config)

        memory._clustering.run.assert_called_once_with(use_semhash=True)
        memory._evolution.run_evolution_cycle.assert_not_called()
        assert result.evolutions["clustering"] == "completed"

    def test_evolve_skips_evolution_when_disabled(self):
        """When both evolution and clustering are disabled, results are empty."""
        stage, memory = self._make_stage()
        state = PipelineState(text="Skip evolution.")
        config = PipelineConfig.default()
        config.evolve = EvolveConfig(run_evolution=False, run_clustering=False)

        result = stage.execute(state, config)

        memory._evolution.run_evolution_cycle.assert_not_called()
        memory._clustering.run.assert_not_called()
        assert result.evolutions == {}

    def test_evolve_handles_evolution_failure(self):
        """When evolution raises, results contain 'failed' and clustering still runs."""
        stage, memory = self._make_stage()
        memory._evolution.run_evolution_cycle.side_effect = RuntimeError("evolution broke")
        state = PipelineState(text="Failing evolution.")
        config = PipelineConfig.default()
        config.evolve = EvolveConfig(run_evolution=True, run_clustering=True)

        result = stage.execute(state, config)

        assert "failed" in result.evolutions["evolution"]
        assert result.evolutions["clustering"] == "completed"

    def test_evolve_handles_clustering_failure(self):
        """When clustering raises, results contain 'failed'."""
        stage, memory = self._make_stage()
        memory._clustering.run.side_effect = RuntimeError("clustering broke")
        state = PipelineState(text="Failing clustering.")
        config = PipelineConfig.default()
        config.evolve = EvolveConfig(run_evolution=False, run_clustering=True)

        result = stage.execute(state, config)

        assert "failed" in result.evolutions["clustering"]

    def test_undo_clears_evolutions(self):
        """Undo resets evolutions to empty dict."""
        stage, _ = self._make_stage()
        state = PipelineState(
            text="Content.",
            evolutions={"evolution": "completed", "clustering": "completed"},
        )

        result = stage.undo(state)

        assert result.evolutions == {}
