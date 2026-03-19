"""CORE-EVO-LIVE-1: Write batcher.

Collects EvolutionActions from all evolvers in a cycle, then flushes them
through the shared backend in a single transaction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from smartmemory.evolution.events import EvolutionAction

if TYPE_CHECKING:
    from smartmemory.graph.backends.backend import SmartGraphBackend

logger = logging.getLogger(__name__)


class WriteBatcher:
    """Collect evolution actions and flush them through a backend transaction."""

    def __init__(self) -> None:
        self._actions: List[EvolutionAction] = []

    def add(self, action: EvolutionAction) -> None:
        """Queue an action for the next flush."""
        self._actions.append(action)

    def add_all(self, actions: List[EvolutionAction]) -> None:
        """Queue multiple actions."""
        self._actions.extend(actions)

    @property
    def pending_count(self) -> int:
        return len(self._actions)

    def flush(self, backend: SmartGraphBackend, memory: Any = None) -> int:
        """Flush all queued actions through the backend.

        Args:
            backend: The shared foreground SmartGraphBackend.
            memory: Optional SmartMemory instance for run_batch_evolver actions.

        Returns:
            Number of actions executed.
        """
        if not self._actions:
            return 0

        count = 0
        # Separate transaction-safe actions from non-transactional ones
        tx_actions = [a for a in self._actions if a.operation != "run_batch_evolver"]
        batch_actions = [a for a in self._actions if a.operation == "run_batch_evolver"]

        # Execute graph mutations atomically in a single transaction
        if tx_actions:
            with backend.transaction_context():
                for action in tx_actions:
                    self._execute_action(action, backend, memory)
                    count += 1

        # run_batch_evolver executes outside the transaction (it does its own writes)
        # Deduplicate by evolver class — same evolver should only run once per cycle
        seen_evolvers: set = set()
        for action in batch_actions:
            evolver_key = type(action.evolver).__name__ if action.evolver else id(action)
            if evolver_key in seen_evolvers:
                continue
            seen_evolvers.add(evolver_key)
            try:
                self._execute_action(action, backend, memory)
                count += 1
            except Exception as exc:
                logger.warning(
                    "WriteBatcher: batch evolver %s failed: %s",
                    evolver_key,
                    exc,
                )

        self._actions.clear()
        return count

    def _execute_action(self, action: EvolutionAction, backend: SmartGraphBackend, memory: Any) -> None:
        """Execute a single EvolutionAction through the backend API."""
        if action.operation == "update_property":
            backend.set_properties(action.target_id, action.properties or {})
        elif action.operation == "add_edge":
            backend.add_edge(
                source_id=action.source_id,
                target_id=action.target_id,
                edge_type=action.edge_type,
                properties=action.properties or {},
            )
        elif action.operation == "remove_node":
            backend.remove_node(action.target_id)
        elif action.operation == "archive":
            backend.set_properties(action.target_id, {"archived": True})
        elif action.operation == "run_batch_evolver":
            # Delegate to the evolver's batch evolve() method
            if action.evolver and memory:
                try:
                    action.evolver.evolve(memory)
                except Exception as exc:
                    logger.warning("WriteBatcher: batch evolver %s failed: %s", action.evolver, exc)
        else:
            logger.warning("WriteBatcher: unknown action %s", action.operation)

    def clear(self) -> None:
        """Discard all pending actions."""
        self._actions.clear()
