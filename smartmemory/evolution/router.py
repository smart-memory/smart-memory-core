"""CORE-EVO-LIVE-1: Evolution event router.

Maps mutation events to relevant evolvers based on their declared TRIGGERS.
Builds the routing table once at construction, dispatches per-event via dict lookup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from smartmemory.evolution.events import MutationEvent

logger = logging.getLogger(__name__)


class EvolutionRouter:
    """Route mutation events to evolvers that declared matching TRIGGERS."""

    def __init__(self, evolvers: List[Any]) -> None:
        """Build routing table from evolver TRIGGERS class attributes.

        Args:
            evolvers: List of evolver instances (must have TRIGGERS: set[tuple[str, str]]).
        """
        self._routing_table: Dict[Tuple[str, str], List[Any]] = {}
        self._all_evolvers = evolvers

        for evolver in evolvers:
            triggers: Set[Tuple[str, str]] = getattr(evolver, "TRIGGERS", set())
            for trigger in triggers:
                self._routing_table.setdefault(trigger, []).append(evolver)

        logger.debug(
            "EvolutionRouter: %d evolvers, %d trigger entries",
            len(evolvers),
            len(self._routing_table),
        )

    def route(self, event: MutationEvent) -> List[Any]:
        """Return list of evolvers whose TRIGGERS match this event.

        Match key is (memory_type, operation).
        """
        key = (event.memory_type, event.operation)
        matched = self._routing_table.get(key, [])
        if not matched:
            logger.debug(
                "EvolutionRouter: no evolvers for %s/%s on %s",
                event.memory_type,
                event.operation,
                event.item_id,
            )
        return matched
