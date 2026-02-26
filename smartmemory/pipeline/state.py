"""PipelineState — immutable snapshot of data flowing through the pipeline.

Stages read from state, compute, and return a *new* state via ``dataclasses.replace()``.
Serializable via ``to_dict()`` / ``from_dict()`` for checkpointing and event-bus transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from smartmemory.pipeline.token_tracker import PipelineTokenTracker


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=False)
class PipelineState:
    """Immutable-by-convention snapshot of pipeline data.

    Stages must not mutate the state in-place — always use
    ``dataclasses.replace(state, field=value)`` to produce a new instance.
    """

    # -- Execution context --
    mode: str = "sync"
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None

    # -- Input --
    text: str = ""
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    memory_type: Optional[str] = None

    # -- Pre-processing --
    resolved_text: Optional[str] = None
    simplified_sentences: List[str] = field(default_factory=list)

    # -- Classification --
    classified_types: List[str] = field(default_factory=list)

    # -- Extraction --
    ruler_entities: List[Any] = field(default_factory=list)
    llm_entities: List[Any] = field(default_factory=list)
    llm_relations: List[Any] = field(default_factory=list)
    llm_decisions: List[Dict[str, Any]] = field(default_factory=list)  # CORE-SYS2-1b
    extraction_status: Optional[str] = None  # ruler_only | llm_enriched | llm_failed

    # -- Constraint (post-extraction) --
    entities: List[Any] = field(default_factory=list)
    relations: List[Any] = field(default_factory=list)
    rejected: List[Any] = field(default_factory=list)
    promotion_candidates: List[Any] = field(default_factory=list)

    # -- Ontology (OL-2, OL-3, OL-4) --
    ontology_registry_id: str = ""
    ontology_version: str = ""
    constraint_violations: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_entities: List[Dict[str, Any]] = field(default_factory=list)

    # -- Storage --
    item_id: Optional[str] = None
    entity_ids: Dict[str, str] = field(default_factory=dict)

    # -- Post-processing --
    links: Dict[str, Any] = field(default_factory=dict)
    enrichments: Dict[str, Any] = field(default_factory=dict)
    evolutions: Dict[str, Any] = field(default_factory=dict)

    # -- Pipeline metadata --
    stage_history: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # -- Token tracking (CFS-1) --
    # NOTE: Intentional exception to the immutability convention above.
    # The tracker is a mutable accumulator shared across all stages in a single
    # pipeline run.  Stages call record_spent/record_avoided in-place.
    token_tracker: Optional["PipelineTokenTracker"] = None

    # -- Carry-through for IngestionContext compatibility --
    _context: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON / event-bus."""
        from dataclasses import fields as dc_fields

        out: Dict[str, Any] = {}
        for f in dc_fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            elif f.name == "token_tracker" and val is not None:
                val = val.summary()
            out[f.name] = val
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineState:
        """Reconstruct from a serialized dict."""
        from dataclasses import fields as dc_fields

        known = {f.name for f in dc_fields(cls) if not f.name.startswith("_")}
        kwargs: Dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue
            # Skip token_tracker — summary dict cannot be reconstructed
            # into a live tracker; start fresh on resumed pipelines.
            if k == "token_tracker":
                continue
            # Re-parse datetime strings
            if k in ("started_at", "completed_at") and isinstance(v, str):
                try:
                    v = datetime.fromisoformat(v)
                except (ValueError, TypeError):
                    pass
            kwargs[k] = v
        return cls(**kwargs)
