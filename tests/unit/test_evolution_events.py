"""Unit tests for CORE-EVO-LIVE-1: MutationEvent, EvolutionAction, EvolutionContext."""

from smartmemory.evolution.events import EvolutionAction, EvolutionContext, MutationEvent


class TestMutationEvent:
    def test_create_add_event(self):
        ev = MutationEvent(
            item_id="item-1", memory_type="working", operation="add", workspace_id="ws-1"
        )
        assert ev.item_id == "item-1"
        assert ev.operation == "add"
        assert ev.timestamp > 0
        assert ev.properties is None
        assert ev.neighbors is None

    def test_create_delete_event_with_snapshot(self):
        ev = MutationEvent(
            item_id="item-2",
            memory_type="semantic",
            operation="delete",
            workspace_id="ws-1",
            properties={"content": "test"},
            neighbors=[{"id": "item-3", "edge_type": "RELATED", "direction": "outgoing"}],
        )
        assert ev.operation == "delete"
        assert ev.properties == {"content": "test"}
        assert len(ev.neighbors) == 1


class TestEvolutionAction:
    def test_update_property_action(self):
        action = EvolutionAction(
            operation="update_property",
            target_id="item-1",
            properties={"confidence": 0.9},
        )
        assert action.operation == "update_property"
        assert action.properties["confidence"] == 0.9

    def test_run_batch_evolver_action(self):
        sentinel = object()
        action = EvolutionAction(operation="run_batch_evolver", evolver=sentinel)
        assert action.evolver is sentinel


class TestEvolutionContext:
    def test_context_wraps_event(self):
        ev = MutationEvent(
            item_id="item-1", memory_type="working", operation="add", workspace_id="ws-1"
        )

        class FakeGraph:
            def get_node(self, item_id):
                return {"item_id": item_id, "content": "hello"}

            def get_neighbors(self, item_id):
                return [{"item_id": "n1", "content": "neighbor"}]

        class FakeBackend:
            def search_nodes(self, query):
                return [{"item_id": "w1"}, {"item_id": "w2"}]

        ctx = EvolutionContext(
            event=ev, graph=FakeGraph(), backend=FakeBackend(), workspace_id="ws-1"
        )
        assert ctx.get_item("item-1")["content"] == "hello"
        assert len(ctx.get_neighbors("item-1")) == 1
        assert ctx.count_by_type("working") == 2
        assert len(ctx.search(memory_type="working")) == 2
