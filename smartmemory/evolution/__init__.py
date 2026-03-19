"""
Evolution module for SmartMemory.

Provides procedure evolution tracking, diff computation, and timeline support.
"""

from smartmemory.evolution.models import (
    ContentSnapshot,
    EventDiff,
    EventSource,
    EvolutionEvent,
    MatchStatsSnapshot,
)
from smartmemory.evolution.diff_engine import ProcedureDiffEngine

# Existing exports
from smartmemory.evolution.cycle import run_evolution_cycle
from smartmemory.evolution.flow import EvolutionFlow
from smartmemory.evolution.registry import (
    EVOLVER_REGISTRY,
    EvolverRegistry,
    EvolverSpec,
    get_evolver_by_key,
    list_evolver_specs,
    register_builtin_evolvers,
)

# CORE-EVO-LIVE-1: Incremental evolution components
from smartmemory.evolution.events import EvolutionAction, EvolutionContext, MutationEvent
from smartmemory.evolution.queue import EvolutionQueue
from smartmemory.evolution.router import EvolutionRouter
from smartmemory.evolution.batcher import WriteBatcher
from smartmemory.evolution.worker import EvolutionWorker

# EvolutionEventStore and EvolutionTracker require pymongo (service-mode only).
# Lazy import so `import smartmemory` works without pymongo installed.
_LAZY_IMPORTS = {"EvolutionEventStore": "store", "EvolutionTracker": "tracker"}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(f"smartmemory.evolution.{_LAZY_IMPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # New evolution timeline exports
    "ContentSnapshot",
    "EventDiff",
    "EventSource",
    "EvolutionEvent",
    "MatchStatsSnapshot",
    "ProcedureDiffEngine",
    "EvolutionEventStore",
    "EvolutionTracker",
    # Existing exports
    "run_evolution_cycle",
    "EvolutionFlow",
    "EVOLVER_REGISTRY",
    "EvolverRegistry",
    "EvolverSpec",
    "get_evolver_by_key",
    "list_evolver_specs",
    "register_builtin_evolvers",
    # CORE-EVO-LIVE-1
    "MutationEvent",
    "EvolutionAction",
    "EvolutionContext",
    "EvolutionQueue",
    "EvolutionRouter",
    "WriteBatcher",
    "EvolutionWorker",
]
