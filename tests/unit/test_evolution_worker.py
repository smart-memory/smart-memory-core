"""Unit tests for CORE-EVO-LIVE-1: EvolutionWorker lifecycle."""

import time

from smartmemory.evolution.events import EvolutionAction, EvolutionContext, MutationEvent
from smartmemory.evolution.queue import EvolutionQueue
from smartmemory.evolution.router import EvolutionRouter
from smartmemory.evolution.worker import EvolutionWorker


class IncrementalEvolver:
    TRIGGERS = {("working", "add")}
    actions_returned: list = []

    def evolve_incremental(self, ctx: EvolutionContext) -> list:
        return self.actions_returned

    def evolve(self, memory, logger=None):
        pass


class FakeBackend:
    """Minimal backend that records set_properties calls."""

    def __init__(self):
        self.set_properties_calls = []

    def set_properties(self, item_id, properties):
        self.set_properties_calls.append((item_id, properties))
        return True

    def transaction_context(self):
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            yield

        return _noop()

    def search_nodes(self, query):
        return []


class FakeGraph:
    def get_node(self, item_id):
        return {"item_id": item_id}

    def get_neighbors(self, item_id):
        return []


class TestEvolutionWorker:
    def test_worker_starts_and_stops(self):
        queue = EvolutionQueue()
        router = EvolutionRouter([])
        backend = FakeBackend()
        graph = FakeGraph()

        worker = EvolutionWorker(queue, router, backend, graph)
        worker.ensure_started()
        assert worker.is_alive()
        assert worker.is_active

        worker.shutdown(drain=True, timeout=2.0)
        assert not worker.is_active

    def test_worker_processes_events(self):
        evolver = IncrementalEvolver()
        evolver.actions_returned = [
            EvolutionAction(operation="update_property", target_id="item-1", properties={"score": 0.9})
        ]

        queue = EvolutionQueue()
        router = EvolutionRouter([evolver])
        backend = FakeBackend()
        graph = FakeGraph()

        worker = EvolutionWorker(queue, router, backend, graph)
        worker.ensure_started()

        # Emit event
        queue.put(MutationEvent(
            item_id="item-1", memory_type="working", operation="add", workspace_id="ws"
        ))

        # Wait for processing
        time.sleep(0.2)

        worker.shutdown(drain=True, timeout=2.0)

        # Verify backend received the write
        assert len(backend.set_properties_calls) >= 1
        assert backend.set_properties_calls[0] == ("item-1", {"score": 0.9})

    def test_worker_drains_on_shutdown(self):
        queue = EvolutionQueue()
        router = EvolutionRouter([])
        backend = FakeBackend()
        graph = FakeGraph()

        worker = EvolutionWorker(queue, router, backend, graph)
        worker.ensure_started()

        # Put events and immediately shut down
        for i in range(5):
            queue.put(MutationEvent(
                item_id=f"item-{i}", memory_type="semantic", operation="add", workspace_id="ws"
            ))

        worker.shutdown(drain=True, timeout=2.0)
        # Queue should be empty after drain
        assert len(queue) == 0
