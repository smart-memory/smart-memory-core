"""CORE-EVO-LIVE-1: Mutation events, evolution actions, and evolution context.

MutationEvent — emitted by CRUD on add/update/delete.
EvolutionAction — returned by incremental evolvers, compiled by WriteBatcher.
EvolutionContext — thread-safe facade provided to incremental evolvers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from smartmemory.graph.backends.backend import SmartGraphBackend
    from smartmemory.graph.compute import GraphComputeLayer


@dataclass
class MutationEvent:
    """Lightweight event emitted by CRUD on every mutation."""

    item_id: str
    memory_type: str
    operation: str  # "add", "update", "delete"
    workspace_id: str
    timestamp: float = field(default_factory=time.monotonic)
    properties: Optional[Dict[str, Any]] = None  # changed props (update) or full snapshot (delete)
    neighbors: Optional[List[Dict[str, Any]]] = None  # pre-delete neighbors (delete only)


@dataclass
class EvolutionAction:
    """Single write operation returned by an incremental evolver."""

    operation: str  # "update_property", "add_edge", "remove_node", "archive", "run_batch_evolver"
    target_id: str = ""
    source_id: str = ""
    edge_type: str = ""
    properties: Optional[Dict[str, Any]] = None
    evolver: Any = None  # EvolverPlugin instance for run_batch_evolver


@dataclass
class EvolutionContext:
    """Thread-safe facade for incremental evolvers.

    Does NOT share the caller's SmartMemory instance. Provides:
    - Read access via the shared thread-safe GraphComputeLayer (RLock-protected).
    - Write access through EvolutionActions (compiled by WriteBatcher later).
    - Backend queries for workspace-scoped searches.
    """

    event: MutationEvent
    graph: GraphComputeLayer  # shared, thread-safe (RLock-protected)
    backend: SmartGraphBackend  # shared foreground backend
    workspace_id: str

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Read item from GraphComputeLayer (~150ns)."""
        return self.graph.get_node(item_id)

    def get_neighbors(self, item_id: str) -> List[Dict[str, Any]]:
        """Read 1-hop neighbors from GraphComputeLayer."""
        return self.graph.get_neighbors(item_id)

    def count_by_type(self, memory_type: str) -> int:
        """Count nodes of a given type in workspace via backend query."""
        results = self.backend.search_nodes({"memory_type": memory_type})
        return len(results)

    def search(self, **query: Any) -> List[Dict[str, Any]]:
        """Search within workspace — delegates to backend.search_nodes()."""
        return self.backend.search_nodes(dict(query))
