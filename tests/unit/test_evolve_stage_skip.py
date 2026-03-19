"""Unit tests for CORE-EVO-LIVE-1: EvolveStage skips batch when worker active."""

from smartmemory.pipeline.config import EvolveConfig, PipelineConfig
from smartmemory.pipeline.stages.evolve import EvolveStage
from smartmemory.pipeline.state import PipelineState


class FakeEvolutionWorker:
    """Fake worker that reports as active."""

    @property
    def is_active(self):
        return True


class FakeEvolution:
    """Fake EvolutionOrchestrator that tracks calls."""

    def __init__(self):
        self.cycle_called = False

    def run_evolution_cycle(self):
        self.cycle_called = True


class FakeClustering:
    def run(self, **kwargs):
        pass


class FakeMemory:
    def __init__(self, worker_active: bool = False):
        self._evolution = FakeEvolution()
        self._clustering = FakeClustering()
        self._evolution_worker = FakeEvolutionWorker() if worker_active else None


class TestEvolveStageSkip:
    def test_skips_batch_when_worker_active(self):
        memory = FakeMemory(worker_active=True)
        stage = EvolveStage(memory)
        config = PipelineConfig(evolve=EvolveConfig(run_evolution=True))
        state = PipelineState(text="test")

        result = stage.execute(state, config)
        assert "skipped" in result.evolutions.get("evolution", "")
        assert memory._evolution.cycle_called is False

    def test_runs_batch_when_no_worker(self):
        memory = FakeMemory(worker_active=False)
        stage = EvolveStage(memory)
        config = PipelineConfig(evolve=EvolveConfig(run_evolution=True))
        state = PipelineState(text="test")

        stage.execute(state, config)
        assert memory._evolution.cycle_called is True

    def test_skips_when_evolution_disabled(self):
        memory = FakeMemory(worker_active=False)
        stage = EvolveStage(memory)
        config = PipelineConfig(evolve=EvolveConfig(run_evolution=False, run_clustering=False))
        state = PipelineState(text="test")

        result = stage.execute(state, config)
        assert "evolution" not in result.evolutions
        assert memory._evolution.cycle_called is False
