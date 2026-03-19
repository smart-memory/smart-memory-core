"""Unit tests for CORE-EVO-LIVE-1: EvolutionQueue and coalescing state machine."""

from smartmemory.evolution.events import MutationEvent
from smartmemory.evolution.queue import EvolutionQueue


def _event(item_id: str, op: str, mem_type: str = "semantic", props: dict = None) -> MutationEvent:
    return MutationEvent(
        item_id=item_id, memory_type=mem_type, operation=op,
        workspace_id="ws-1", properties=props,
    )


class TestEvolutionQueue:
    def test_put_and_drain(self):
        q = EvolutionQueue()
        q.put(_event("a", "add"))
        q.put(_event("b", "add"))
        events = q.drain()
        assert len(events) == 2
        assert events[0].item_id == "a"
        assert events[1].item_id == "b"

    def test_drain_empty(self):
        q = EvolutionQueue()
        assert q.drain() == []

    def test_len(self):
        q = EvolutionQueue()
        assert len(q) == 0
        q.put(_event("a", "add"))
        assert len(q) == 1

    def test_wait_returns_true_when_signalled(self):
        import threading

        q = EvolutionQueue()

        def put_after_delay():
            import time
            time.sleep(0.01)
            q.put(_event("a", "add"))

        t = threading.Thread(target=put_after_delay)
        t.start()
        result = q.wait(timeout=1.0)
        assert result is True
        t.join()


class TestCoalescing:
    def test_add_then_update_collapses_to_add(self):
        events = [
            _event("a", "add", props={"k1": "v1"}),
            _event("a", "update", props={"k2": "v2"}),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 1
        assert result[0].operation == "add"
        assert result[0].properties == {"k1": "v1", "k2": "v2"}

    def test_add_then_delete_is_noop(self):
        events = [
            _event("a", "add"),
            _event("a", "delete"),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 0

    def test_update_then_update_merges(self):
        events = [
            _event("a", "update", props={"k1": "v1"}),
            _event("a", "update", props={"k2": "v2"}),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 1
        assert result[0].operation == "update"
        assert result[0].properties == {"k1": "v1", "k2": "v2"}

    def test_update_then_delete_keeps_delete(self):
        events = [
            _event("a", "update", props={"k1": "v1"}),
            _event("a", "delete"),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 1
        assert result[0].operation == "delete"

    def test_delete_then_add_keeps_add(self):
        """Fast recreate within debounce window — delete+add → add."""
        events = [
            _event("a", "delete"),
            _event("a", "add", props={"content": "recreated"}),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 1
        assert result[0].operation == "add"

    def test_delete_then_update_stays_delete(self):
        """Stale update on deleted item — delete wins."""
        events = [
            _event("a", "delete"),
            _event("a", "update"),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 1
        assert result[0].operation == "delete"

    def test_different_items_not_coalesced(self):
        events = [
            _event("a", "add"),
            _event("b", "add"),
            _event("a", "update", props={"x": 1}),
        ]
        result = EvolutionQueue.coalesce(events)
        assert len(result) == 2
        ids = {e.item_id for e in result}
        assert ids == {"a", "b"}

    def test_empty_list(self):
        assert EvolutionQueue.coalesce([]) == []
