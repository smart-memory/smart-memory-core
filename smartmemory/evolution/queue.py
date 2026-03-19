"""CORE-EVO-LIVE-1: Thread-safe evolution queue with state machine coalescing.

EvolutionQueue wraps a collections.deque and threading.Event for zero-dependency
inter-thread communication. Coalescing uses a per-item state machine to collapse
rapid mutations within a debounce window.
"""

from __future__ import annotations

import collections
import threading
from typing import Dict, List

from smartmemory.evolution.events import MutationEvent


class EvolutionQueue:
    """Thread-safe mutation event queue with coalescing support."""

    def __init__(self) -> None:
        self._deque: collections.deque[MutationEvent] = collections.deque()
        self._event = threading.Event()

    def put(self, event: MutationEvent) -> None:
        """Append a mutation event and wake the worker."""
        self._deque.append(event)
        self._event.set()

    def drain(self) -> List[MutationEvent]:
        """Pop all queued events (non-blocking)."""
        events: List[MutationEvent] = []
        while self._deque:
            try:
                events.append(self._deque.popleft())
            except IndexError:
                break
        return events

    def wait(self, timeout: float) -> bool:
        """Block until an event is queued or timeout expires.

        Returns True if signalled, False on timeout.
        """
        signalled = self._event.wait(timeout=timeout)
        self._event.clear()
        return signalled

    def __len__(self) -> int:
        return len(self._deque)

    @staticmethod
    def coalesce(events: List[MutationEvent]) -> List[MutationEvent]:
        """Coalesce events per item_id using a state machine.

        Rules:
        - add + update → add (with merged properties)
        - add + delete → no-op (item never existed from evolver perspective)
        - update + delete → delete (delete wins)
        - update + update → single update (latest properties)
        - delete + anything → delete wins (no further mutations possible)
        """
        state: Dict[str, MutationEvent] = {}

        for ev in events:
            key = ev.item_id
            if key not in state:
                state[key] = ev
                continue

            prev = state[key]

            if prev.operation == "add" and ev.operation == "update":
                # add + update → add with merged properties
                merged_props = dict(prev.properties or {})
                merged_props.update(ev.properties or {})
                state[key] = MutationEvent(
                    item_id=ev.item_id,
                    memory_type=ev.memory_type,
                    operation="add",
                    workspace_id=ev.workspace_id,
                    timestamp=ev.timestamp,
                    properties=merged_props,
                    neighbors=prev.neighbors,
                )
            elif prev.operation == "add" and ev.operation == "delete":
                # add + delete → no-op
                del state[key]
            elif prev.operation == "update" and ev.operation == "update":
                # update + update → single update with latest properties
                merged_props = dict(prev.properties or {})
                merged_props.update(ev.properties or {})
                state[key] = MutationEvent(
                    item_id=ev.item_id,
                    memory_type=ev.memory_type,
                    operation="update",
                    workspace_id=ev.workspace_id,
                    timestamp=ev.timestamp,
                    properties=merged_props,
                    neighbors=None,
                )
            elif prev.operation == "update" and ev.operation == "delete":
                # update + delete → delete wins
                state[key] = ev
            elif prev.operation == "delete" and ev.operation == "add":
                # delete + add → item recreated: treat as add (fast recreate within debounce window)
                state[key] = ev
            elif prev.operation == "delete":
                # delete + update → stale update on deleted item, keep delete
                pass
            else:
                # Unrecognized combination — keep latest
                state[key] = ev

        return list(state.values())
