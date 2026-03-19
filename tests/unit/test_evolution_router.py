"""Unit tests for CORE-EVO-LIVE-1: EvolutionRouter."""

from smartmemory.evolution.events import MutationEvent
from smartmemory.evolution.router import EvolutionRouter


class FakeEvolver:
    TRIGGERS = {("working", "add"), ("episodic", "add")}

    def evolve_incremental(self, ctx):
        return []


class BatchOnlyEvolver:
    TRIGGERS = set()

    def evolve(self, memory, logger=None):
        pass


class TestEvolutionRouter:
    def test_routes_to_matching_evolver(self):
        evolver = FakeEvolver()
        router = EvolutionRouter([evolver])
        event = MutationEvent(
            item_id="a", memory_type="working", operation="add", workspace_id="ws"
        )
        matched = router.route(event)
        assert evolver in matched

    def test_no_match_for_unregistered_trigger(self):
        evolver = FakeEvolver()
        router = EvolutionRouter([evolver])
        event = MutationEvent(
            item_id="a", memory_type="semantic", operation="delete", workspace_id="ws"
        )
        matched = router.route(event)
        assert matched == []

    def test_batch_only_evolver_never_matches(self):
        batch_evolver = BatchOnlyEvolver()
        router = EvolutionRouter([batch_evolver])
        event = MutationEvent(
            item_id="a", memory_type="working", operation="add", workspace_id="ws"
        )
        assert router.route(event) == []

    def test_multiple_evolvers_for_same_trigger(self):
        e1 = FakeEvolver()
        e2 = FakeEvolver()
        router = EvolutionRouter([e1, e2])
        event = MutationEvent(
            item_id="a", memory_type="episodic", operation="add", workspace_id="ws"
        )
        matched = router.route(event)
        assert len(matched) == 2
