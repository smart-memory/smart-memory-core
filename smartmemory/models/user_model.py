"""
UserModel: Integrates explicit (IPIP, questionnaires) and implicit (behavioral, inferred) psychological user modeling.
- Stores Big Five/IPIP traits, inferred traits, and history.
- Updateable from direct input, behavioral logs, or deduction/induction.
- Exportable as a Zettel node/memory item.
- Designed for integration with agentic routines and memory.
"""
import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from smartmemory.models.base import MemoryBaseModel


@dataclass
class UserModel(MemoryBaseModel):
    """
    UserModel: Integrates explicit (IPIP, questionnaires) and implicit (behavioral, inferred) psychological user modeling.
    - Stores Big Five/IPIP traits, inferred traits, and history.
    - Updateable from direct input, behavioral logs, or deduction/induction.
    - Exportable as a Zettel node/memory item.
    - Designed for integration with agentic routines and memory.
    """
    big_five: Dict[str, float] = field(default_factory=lambda: {'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5})
    last_explicit: Optional[Dict[str, Any]] = None
    inferred_traits: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update_from_explicit(self, trait_scores: Dict[str, float], responses: Optional[Dict[str, Any]] = None):
        """Update models from explicit psychometric input (e.g., IPIP, other questionnaires)."""
        self.big_five = {k: float(v) for k, v in trait_scores.items() if k in self.big_five}
        self.last_explicit = responses
        self.history.append({
            'type': 'explicit',
            'timestamp': datetime.datetime.now(datetime.UTC).isoformat(),
            'trait_scores': trait_scores,
            'responses': responses
        })

    def update_from_behavior(self, behavior: Dict[str, Any]):
        """Update models from behavioral signals (e.g., agent feedback, choices, time spent, etc.)."""
        # Example: infer openness from exploration, conscientiousness from task completion
        for k, v in behavior.items():
            self.inferred_traits[k] = v
        self.history.append({
            'type': 'behavior',
            'timestamp': datetime.datetime.now(UTC).isoformat(),
            'behavior': behavior
        })

    def update_from_inference(self, deductions: Dict[str, Any]):
        """Update models from indirect inference, deduction, or induction (e.g., LLM analysis, pattern mining)."""
        for k, v in deductions.items():
            self.inferred_traits[k] = v
        self.history.append({
            'type': 'inference',
            'timestamp': datetime.datetime.now(UTC).isoformat(),
            'deductions': deductions
        })

    def get_profile(self) -> Dict[str, Any]:
        """Return the full user profile, combining explicit and inferred traits."""
        profile = dict(self.big_five)
        profile.update(self.inferred_traits)
        return profile

    def to_zettel(self) -> Dict[str, Any]:
        """Export as a Zettel node for linking/recall/visualization."""
        return {
            'type': 'user_profile',
            'big_five': self.big_five,
            'inferred_traits': self.inferred_traits,
            'history': self.history
        }
