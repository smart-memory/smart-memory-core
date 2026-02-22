"""
Enhanced Memory Evolution Algorithms

Based on cognitive science research and consolidation theory, these algorithms
implement more sophisticated memory evolution mechanisms that better models
human memory consolidation processes.
"""

import importlib
from typing import Any, Dict, Optional, Type

from smartmemory.plugins.evolvers.enhanced.exponential_decay import ExponentialDecayEvolver
from smartmemory.plugins.evolvers.enhanced.interference_based_consolidation import InterferenceBasedConsolidationEvolver
from smartmemory.plugins.evolvers.enhanced.working_to_episodic import EnhancedWorkingToEpisodicEvolver

# Note: RetrievalBasedStrengtheningEvolver is deferred to CORE-EVO-ENH-2 (requires retrieval-tracking infrastructure)
# from smartmemory.plugins.evolvers.enhanced.retrieval_based_strengthening import RetrievalBasedStrengtheningEvolver

# Enhanced evolver list including new algorithms
ENHANCED_EVOLVERS = [
    EnhancedWorkingToEpisodicEvolver,
    ExponentialDecayEvolver,
    InterferenceBasedConsolidationEvolver,

    # Keep the good existing ones
    # EpisodicToSemanticEvolver,  # This one is already optimal
]


def _build_typed_config(evolver_cls: Type, config_snapshot: Optional[Dict[str, Any]]):
    """Infer and construct the typed Config class co-located with the evolver class.
    Convention: replace suffix 'Evolver' with 'Config' within the same module.
    """
    module = importlib.import_module(evolver_cls.__module__)
    evolver_name = evolver_cls.__name__
    if not evolver_name.endswith("Evolver"):
        raise TypeError(
            f"Evolver class {evolver_name} does not follow naming convention with 'Evolver' suffix"
        )
    config_name = evolver_name.replace("Evolver", "Config")
    ConfigType: Optional[Type] = getattr(module, config_name, None)
    if ConfigType is None:
        raise TypeError(
            f"Missing companion Config class '{config_name}' for evolver {evolver_name}"
        )
    if config_snapshot is None:
        return ConfigType()
    if not isinstance(config_snapshot, dict):
        if isinstance(config_snapshot, ConfigType):
            return config_snapshot
        raise TypeError("Config snapshot must be a dict or the typed Config instance")
    return ConfigType(**config_snapshot)


def run_enhanced_evolution_cycle(memory, config: Optional[Dict[str, Any]] = None, logger=None):
    """Run enhanced evolution cycle with improved algorithms.

    Args:
        memory: SmartMemory instance passed to each evolver.
        config: Optional flat dict of config overrides. MUST be None when running
            multiple evolvers — each evolver's Config dataclass has distinct fields,
            and passing a dict with keys from one evolver's schema will raise TypeError
            in _build_typed_config for the others, silently disabling them. To configure
            individual evolvers, instantiate them directly with their typed Config.
        logger: Optional logger instance.
    """
    for EvolverClass in ENHANCED_EVOLVERS:
        typed_config = _build_typed_config(EvolverClass, config)
        evolver = EvolverClass(config=typed_config)
        try:
            evolver.evolve(memory, logger=logger)
        except Exception as e:
            if logger:
                logger.error(f"Error in {EvolverClass.__name__}: {e}")
