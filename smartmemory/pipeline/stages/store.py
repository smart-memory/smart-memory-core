"""Store stage — wraps CRUD.add() + StoragePipeline."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, List

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class StoreStage:
    """Persist the memory item and extracted entities to the graph."""

    # Average tokens per embedding call — used to estimate avoided tokens on cache hit.
    _AVG_EMBEDDING_TOKENS: int = 250

    def __init__(self, memory):
        """Args: memory — a SmartMemory instance."""
        self._memory = memory

    @property
    def name(self) -> str:
        return "store"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        from smartmemory.models.memory_item import MemoryItem

        if config.mode == "preview":
            return replace(state, item_id="preview_item")

        content = state.resolved_text or state.text
        metadata = dict(state.raw_metadata)
        if state.extraction_status:
            metadata["extraction_status"] = state.extraction_status
        # Inject run_id if present for run-based cleanup
        if state.raw_metadata.get("run_id"):
            metadata["run_id"] = state.raw_metadata["run_id"]
        # OL-2: Inject ontology version metadata
        if state.ontology_version:
            metadata["ontology_version"] = state.ontology_version
        if state.ontology_registry_id:
            metadata["ontology_registry_id"] = state.ontology_registry_id
        item = MemoryItem(
            content=content,
            memory_type=state.memory_type or "semantic",
            metadata=metadata,
        )

        # Build ontology_extraction payload
        entities = state.entities
        relations = state.relations
        ontology_extraction = None
        if entities or relations:
            ontology_extraction = {"entities": entities, "relations": relations}

        # Store via CRUD
        add_result = self._memory._crud.add(item, ontology_extraction=ontology_extraction)

        # Process result
        entity_ids = {}
        if isinstance(add_result, dict):
            item_id = add_result.get("memory_node_id")
            created_ids = add_result.get("entity_node_ids", []) or []
            entity_ids = self._map_entity_ids(entities, created_ids, item_id)
        else:
            item_id = add_result
            entity_ids = self._map_entity_ids(entities, [], item_id)

        item.item_id = item_id
        item.update_status("created", notes="Item ingested")

        # Process external relations via StoragePipeline
        if relations:
            self._process_relations(state, item_id, entities, relations, entity_ids)

        # Save to vector and graph with token tracking (CFS-1a)
        context = {
            "item": item,
            "entity_ids": entity_ids,
            "entities": entities,
        }
        try:
            from smartmemory.memory.ingestion.storage import StoragePipeline
            from smartmemory.memory.ingestion.observer import IngestionObserver

            storage = StoragePipeline(self._memory, IngestionObserver())
            storage.save_to_vector_and_graph(context)
        except Exception as e:
            logger.warning("Vector/graph save failed (non-fatal): %s", e)
        finally:
            # Track embedding token usage (CFS-1a) — must happen even on save failure
            # because the embedding API call may have completed before the error
            self._track_embedding_usage(state)

        return replace(state, item_id=item_id, entity_ids=entity_ids)

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, item_id=None, entity_ids={})

    @staticmethod
    def _map_entity_ids(entities: List, created_ids: List, item_id) -> dict:
        entity_ids = {}
        for i, entity in enumerate(entities):
            name = _entity_name(entity, i)
            real_id = created_ids[i] if i < len(created_ids) else f"{item_id}_entity_{i}"
            entity_ids[name] = real_id
        return entity_ids

    def _process_relations(self, state, item_id, entities, relations, entity_ids: dict | None = None):
        """Resolve extraction-time entity IDs and create semantic edges.

        Builds a map from extraction-time SHA256 entity hashes to actual graph
        node IDs, then passes resolved relations to StoragePipeline. MERGE
        semantics in the graph layer prevent duplicates if add_dual_node already
        created some of these edges.

        Args:
            entity_ids: The freshly-created entity_id map from this store run.
                        Falls back to state.entity_ids for re-processing scenarios.
        """
        entity_ids = entity_ids or state.entity_ids or {}

        # Build extraction hash → graph ID map
        extraction_id_to_graph_id = {}
        for i, entity in enumerate(entities):
            if hasattr(entity, "item_id") and entity.item_id:
                ename = _entity_name(entity, i)
                graph_id = entity_ids.get(ename)
                if graph_id:
                    extraction_id_to_graph_id[entity.item_id] = graph_id

        # Resolve relation IDs
        resolved = []
        for r in relations:
            src_hash = r.get("source_id")
            tgt_hash = r.get("target_id")
            src_id = extraction_id_to_graph_id.get(src_hash)
            tgt_id = extraction_id_to_graph_id.get(tgt_hash)
            if src_id and tgt_id:
                resolved.append({
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "relation_type": r.get("relation_type", "RELATED"),
                })
            else:
                logger.warning("Unresolvable relation: src=%s tgt=%s", src_hash, tgt_hash)

        if resolved:
            try:
                from smartmemory.memory.ingestion.storage import StoragePipeline
                from smartmemory.memory.ingestion.observer import IngestionObserver

                context = dict(state._context)
                context["entity_ids"] = entity_ids
                storage = StoragePipeline(self._memory, IngestionObserver())
                storage.process_extracted_relations(context, item_id, resolved)
            except Exception as e:
                logger.warning("Failed to process relations: %s", e)

    def _track_embedding_usage(self, state: PipelineState) -> None:
        """Track embedding token usage from the last embedding call (CFS-1a).

        Retrieves token usage from the EmbeddingService's thread-local storage
        and records it to the pipeline token tracker. Cache hits are recorded
        as avoided tokens with an estimated cost.
        """
        tracker = state.token_tracker
        if not tracker:
            return

        try:
            from smartmemory.plugins.embedding import get_last_embedding_usage

            usage = get_last_embedding_usage()
        except ImportError:
            return

        if not usage:
            return

        model = usage.get("model", "text-embedding-ada-002")

        if usage.get("cached"):
            # Cache hit — record estimated avoided tokens
            tracker.record_avoided(
                "store",
                self._AVG_EMBEDDING_TOKENS,
                model=model,
                reason="cache_hit",
            )
        else:
            # API call — record actual tokens spent (embeddings have no completion tokens)
            prompt_tokens = usage.get("prompt_tokens", 0)
            tracker.record_spent("store", prompt_tokens, 0, model=model)


def _entity_name(entity, index: int) -> str:
    """Extract display name from an entity (MemoryItem or dict)."""
    if hasattr(entity, "metadata") and isinstance(getattr(entity, "metadata", None), dict):
        name = entity.metadata.get("name")
        if name:
            return name
    if isinstance(entity, dict):
        meta = entity.get("metadata", {})
        return meta.get("name") or entity.get("name", f"entity_{index}")
    return f"entity_{index}"
