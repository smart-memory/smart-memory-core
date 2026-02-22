"""
SmartMemory Evolvers - Memory evolution plugins.

Evolvers transform memories across types and synthesize new memories from patterns.
"""

# Base evolver
from .base import Evolver

# Decision evolvers
from .decision_confidence import DecisionConfidenceEvolver

# Enhanced evolvers
from .enhanced.exponential_decay import ExponentialDecayEvolver
from .enhanced.interference_based_consolidation import InterferenceBasedConsolidationEvolver

# Maintenance evolvers
from .episodic_decay import EpisodicDecayEvolver
from .episodic_to_semantic import EpisodicToSemanticEvolver
from .semantic_decay import SemanticDecayEvolver
from .episodic_to_zettel import EpisodicToZettelEvolver
from .observation_synthesis import ObservationSynthesisEvolver
from .opinion_reinforcement import OpinionReinforcementEvolver

# Synthesis evolvers
from .opinion_synthesis import OpinionSynthesisEvolver

# Type promotion evolvers
from .working_to_episodic import WorkingToEpisodicEvolver
from .working_to_procedural import WorkingToProceduralEvolver

__all__ = [
    # Base
    'Evolver',

    # Type promotion
    'WorkingToEpisodicEvolver',
    'EpisodicToSemanticEvolver',
    'WorkingToProceduralEvolver',
    'EpisodicToZettelEvolver',

    # Maintenance
    'EpisodicDecayEvolver',
    'SemanticDecayEvolver',

    # Synthesis
    'OpinionSynthesisEvolver',
    'ObservationSynthesisEvolver',
    'OpinionReinforcementEvolver',

    # Decision
    'DecisionConfidenceEvolver',

    # Enhanced
    'ExponentialDecayEvolver',
    'InterferenceBasedConsolidationEvolver',
]
