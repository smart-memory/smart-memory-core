"""
Evolver registry for SmartMemory evolution workflows.

- Provides a pluggable interface for internal and external evolvers.
- Studio can list, validate, and select evolvers by keyword.
- External packages can register their evolvers at import time.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Type, Optional, Iterable


@dataclass(frozen=True)
class EvolverSpec:
    key: str
    dotted_path: str  # e.g., "smartmemory.evolution.working_to_episodic.WorkingToEpisodicEvolver"
    description: str = ""
    tags: tuple[str, ...] = ()


class EvolverRegistry:
    def __init__(self):
        self._specs: Dict[str, EvolverSpec] = {}
        self._cache: Dict[str, Type] = {}

    def register(self, key: str, dotted_path: str, description: str = "", tags: Optional[Iterable[str]] = None) -> None:
        if not key or not dotted_path or "." not in dotted_path:
            raise ValueError("Invalid key or dotted_path for evolver registration")
        k = key.strip()
        if k in self._specs:
            # Allow idempotent re-registration if dotted path matches
            if self._specs[k].dotted_path != dotted_path:
                raise ValueError(f"Evolver key already registered with different path: {k}")
            return
        spec = EvolverSpec(key=k, dotted_path=dotted_path, description=description or "", tags=tuple(tags or ()))
        self._specs[k] = spec

    def get(self, key: str) -> Type:
        spec = self._specs.get(key)
        if not spec:
            raise KeyError(f"Unknown evolver key: {key}")
        if spec.dotted_path in self._cache:
            return self._cache[spec.dotted_path]
        module_path, attr = spec.dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        self._cache[spec.dotted_path] = cls
        return cls

    def try_resolve(self, key_or_path: str) -> Optional[Type]:
        # Prefer registry key
        if key_or_path in self._specs:
            return self.get(key_or_path)
        # Fallback: dotted path import
        try:
            module_path, attr = key_or_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr)
        except Exception:
            return None

    def list_specs(self) -> Dict[str, EvolverSpec]:
        return dict(self._specs)


# Global registry instance
EVOLVER_REGISTRY = EvolverRegistry()


def register_builtin_evolvers() -> None:
    """Register all built-in evolvers with stable keys and tags."""
    R = EVOLVER_REGISTRY
    # Working memory promotions
    R.register(
        "working_to_episodic",
        "smartmemory.evolution.working_to_episodic.WorkingToEpisodicEvolver",
        description="Promote working buffer into episodic summaries",
        tags=("working", "episodic", "promote", "builtin"),
    )
    R.register(
        "working_to_procedural",
        "smartmemory.evolution.working_to_procedural.WorkingToProceduralEvolver",
        description="Extract repeated skill patterns into procedural macros",
        tags=("working", "procedural", "promote", "builtin"),
    )

    # Episodic promotions and maintenance
    R.register(
        "episodic_to_semantic",
        "smartmemory.evolution.episodic_to_semantic.EpisodicToSemanticEvolver",
        description="Promote stable episodic facts to semantic",
        tags=("episodic", "semantic", "promote", "builtin"),
    )
    R.register(
        "episodic_to_zettel",
        "smartmemory.evolution.episodic_to_zettel.EpisodicToZettelEvolver",
        description="Roll up episodic events into zettels",
        tags=("episodic", "zettel", "rollup", "builtin"),
    )
    R.register(
        "episodic_decay",
        "smartmemory.evolution.episodic_decay.EpisodicDecayEvolver",
        description="Archive/delete stale episodic events",
        tags=("episodic", "decay", "maintenance", "builtin"),
    )

    # Semantic maintenance
    R.register(
        "semantic_decay",
        "smartmemory.evolution.semantic_decay.SemanticDecayEvolver",
        description="Archive low-relevance semantic facts",
        tags=("semantic", "decay", "maintenance", "builtin"),
    )

    # Zettel maintenance
    R.register(
        "zettel_prune",
        "smartmemory.evolution.zettel_prune.ZettelPruneEvolver",
        description="Prune/merge low-quality or duplicate zettels",
        tags=("zettel", "prune", "maintenance", "builtin"),
    )

    # Agent-optimized suite (from service_common)
    R.register(
        "maximal_connectivity",
        "service_common.plugins.evolvers.optimized.maximal_connectivity.MaximalConnectivityEvolver",
        description="Create maximum useful connections between items",
        tags=("agent", "connectivity", "links", "experimental"),
    )
    R.register(
        "rapid_enrichment",
        "service_common.plugins.evolvers.optimized.rapid_enrichment.RapidEnrichmentEvolver",
        description="Immediately enrich items with comprehensive context",
        tags=("agent", "enrichment", "experimental"),
    )
    R.register(
        "strategic_pruning",
        "service_common.plugins.evolvers.optimized.strategic_pruning.StrategicPruningEvolver",
        description="Strategically prune redundant/outdated/low-value items",
        tags=("agent", "pruning", "experimental"),
    )
    R.register(
        "hierarchical_organization",
        "service_common.plugins.evolvers.optimized.hierarchical_organization.HierarchicalOrganizationEvolver",
        description="Create hierarchical topic/entity/temporal organization",
        tags=("agent", "hierarchy", "experimental"),
    )

    # Enhanced evolvers present in codebase but not registered

    R.register(
        "enhanced_working_to_episodic",
        "smartmemory.plugins.evolvers.enhanced.working_to_episodic.EnhancedWorkingToEpisodicEvolver",
        description="Cognitively-informed promotion from working to episodic (enhanced)",
        tags=("enhanced", "working", "episodic", "promote"),
    )
    R.register(
        "interference_based_consolidation",
        "smartmemory.plugins.evolvers.enhanced.interference_based_consolidation.InterferenceBasedConsolidationEvolver",
        description="Consolidate while mitigating interference effects (enhanced)",
        tags=("enhanced", "consolidation", "interference"),
    )
    R.register(
        "retrieval_based_strengthening",
        "smartmemory.plugins.evolvers.enhanced.retrieval_based_strengthening.RetrievalBasedStrengtheningEvolver",
        description="Strengthen memories via retrieval practice (enhanced)",
        tags=("enhanced", "retrieval", "strengthening"),
    )
    R.register(
        "exponential_decay",
        "smartmemory.plugins.evolvers.enhanced.exponential_decay.ExponentialDecayEvolver",
        description="Apply exponential forgetting curves (enhanced)",
        tags=("enhanced", "decay"),
    )

    # Decision confidence evolution
    R.register(
        "decision_confidence",
        "smartmemory.plugins.evolvers.decision_confidence.DecisionConfidenceEvolver",
        description="Evidence-based reinforcement, contradiction, decay, and retraction for decisions",
        tags=("decision", "confidence", "decay", "reinforcement", "builtin"),
    )

    # Staleness tracking
    R.register(
        "stale_memory",
        "smartmemory.plugins.evolvers.stale_memory.StaleMemoryEvolver",
        description="Mark non-code memories stale when referenced source files change",
        tags=("maintenance", "code", "staleness", "builtin"),
    )


# Auto-register builtins on import
register_builtin_evolvers()


# Convenience functions for callers

def get_evolver_by_key(key: str):
    return EVOLVER_REGISTRY.get(key)


def list_evolver_specs() -> Dict[str, EvolverSpec]:
    return EVOLVER_REGISTRY.list_specs()
