"""Ground stage — entity grounding to Wikidata or Wikipedia.

When a PublicKnowledgeStore is available (ONTO-PUB-1), uses
PublicKnowledgeGrounder for Wikidata QID-based grounding.
Otherwise falls back to WikipediaGrounder for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class GroundStage:
    """Ground entities to Wikidata (preferred) or Wikipedia (fallback)."""

    def __init__(self, memory, public_knowledge_store=None):
        """Args:
            memory: SmartMemory instance (needs _graph, _grounding).
            public_knowledge_store: PublicKnowledgeStore or None. When set,
                uses PublicKnowledgeGrounder; otherwise WikipediaGrounder.
        """
        self._memory = memory
        self._public_knowledge_store = public_knowledge_store

    @property
    def name(self) -> str:
        return "ground"

    # Estimated tokens for a typical grounding call
    _AVG_GROUND_TOKENS: int = 200

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.mode == "preview":
            return state

        if not config.enrich.wikidata.enabled:
            if state.token_tracker:
                state.token_tracker.record_avoided(
                    "ground",
                    self._AVG_GROUND_TOKENS,
                    reason="stage_disabled",
                )
            return state

        entities = state.entities
        if not entities:
            return state

        try:
            from smartmemory.models.memory_item import MemoryItem

            item = MemoryItem(
                content=state.resolved_text or state.text,
                memory_type=state.memory_type or "semantic",
                metadata=dict(state.raw_metadata),
                item_id=state.item_id,
            )

            # Update entity item_ids from stored mapping
            entity_ids = state.entity_ids
            for entity in entities:
                ename = None
                if isinstance(entity, dict):
                    ename = entity.get("name") or (entity.get("metadata") or {}).get("name")
                elif hasattr(entity, "metadata") and entity.metadata:
                    ename = entity.metadata.get("name")
                if ename and ename in entity_ids:
                    if isinstance(entity, dict):
                        entity["item_id"] = entity_ids[ename]
                    else:
                        entity.item_id = entity_ids[ename]

            # Select grounder: PublicKnowledgeGrounder if store available, else WikipediaGrounder
            if self._public_knowledge_store:
                from smartmemory.grounding.public_knowledge_grounder import PublicKnowledgeGrounder
                from smartmemory.grounding.sparql_client import WDQSClient

                sparql_client = WDQSClient()
                grounder = PublicKnowledgeGrounder(self._public_knowledge_store, sparql_client=sparql_client)
            else:
                from smartmemory.plugins.grounders import WikipediaGrounder

                grounder = WikipediaGrounder()

            # Count entities before grounding to detect graph-gated skips
            entity_count = len(
                [e for e in entities if (isinstance(e, dict) and e.get("name")) or (hasattr(e, "metadata") and e.metadata and e.metadata.get("name"))]
            )

            provenance = grounder.ground(item, entities, self._memory._graph)

            # Record avoided tokens for entities that were graph-gated
            if state.token_tracker and entity_count > 0:
                api_calls_avoided = getattr(grounder, "_graph_hits", 0)
                if api_calls_avoided > 0:
                    state.token_tracker.record_avoided(
                        "ground",
                        self._AVG_GROUND_TOKENS * api_calls_avoided,
                        reason="graph_lookup",
                    )

            # Create GROUNDED_IN edges from memory item to grounding nodes
            if provenance and state.item_id:
                for grounding_id in provenance:
                    try:
                        self._memory._graph.add_edge(
                            state.item_id, grounding_id, edge_type="GROUNDED_IN", properties={}, is_global=True,
                        )
                    except Exception as e:
                        logger.debug("Failed to create memory→grounding edge: %s", e)

            ctx = dict(state._context)
            ctx["provenance_candidates"] = provenance or []

            return replace(state, _context=ctx)

        except Exception as e:
            logger.warning("Grounding failed: %s", e)
            return state

    def undo(self, state: PipelineState) -> PipelineState:
        ctx = dict(state._context)
        ctx.pop("provenance_candidates", None)
        return replace(state, _context=ctx)
