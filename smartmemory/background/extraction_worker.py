"""Tier 2 async extraction worker — runs LLM extraction on items stored by Tier 1.

Tier 1 (sync, ~4ms): spaCy + EntityRuler → stores memory, enqueues item_id.
Tier 2 (this module, ~740ms): loads item, runs LLMSingleExtractor, diffs against
ruler entities already stored, writes net-new entities + resolved relations,
then updates EntityPattern nodes for the self-learning EntityRuler loop.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)

# Pre-write quality gates for entity pattern candidates
_MIN_CONFIDENCE = 0.8
_MIN_NAME_LENGTH = 3
_BLOCKLIST: Set[str] = {"it", "this", "that", "the", "a", "an", "he", "she", "they", "we"}


def process_extract_job(
    memory: Any,
    payload: Dict[str, Any],
    redis_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run Tier 2 LLM extraction on a stored memory item.

    Args:
        memory: A SmartMemory (or SecureSmartMemory) instance scoped to the workspace.
        payload: Dict from the extract queue: {item_id, workspace_id, entity_ids}.
                 entity_ids is {name.lower(): graph_node_id} from Tier 1 StoreStage.
        redis_client: Optional Redis connection for ruler hot-reload pub/sub.

    Returns:
        {"status": "ok"|"no_text"|"item_not_found"|"llm_failed",
         "new_entities": int, "new_relations": int, "new_patterns": int}
    """
    item_id: Optional[str] = payload.get("item_id")
    workspace_id: str = payload.get("workspace_id", "")
    # ruler_entity_ids: {name.lower(): graph_node_id} populated by Tier 1 StoreStage
    ruler_entity_ids: Dict[str, str] = payload.get("entity_ids") or {}
    # Respect enable_ontology flag set by the enqueuing SmartMemory instance.
    # When False, skip EntityPattern writes and ruler hot-reload entirely.
    enable_ontology: bool = payload.get("enable_ontology", True)

    # --- 1. Load stored item ---
    item = memory.get(item_id)
    if item is None:
        logger.warning("Extract job: item_id=%s not found", item_id)
        return {"status": "item_not_found", "new_entities": 0, "new_relations": 0, "new_patterns": 0}

    content: str = ""
    if isinstance(item, MemoryItem):
        content = item.content or ""
    elif isinstance(item, dict):
        content = item.get("content", "") or ""

    if not content.strip():
        return {"status": "no_text", "new_entities": 0, "new_relations": 0, "new_patterns": 0}

    # --- 2. Run LLM extraction ---
    try:
        from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor

        extractor = LLMSingleExtractor()
        extraction = extractor.extract(content)
    except Exception as exc:
        logger.warning("Extract job: LLM extraction failed for item_id=%s: %s", item_id, exc)
        return {"status": "llm_failed", "new_entities": 0, "new_relations": 0, "new_patterns": 0}

    llm_entities: list[MemoryItem] = extraction.get("entities") or []
    llm_relations: list[dict] = extraction.get("relations") or []

    # --- 3. Build id remapping and filter valid relations ---
    from smartmemory.background.id_resolver import build_sha256_to_stored, filter_valid_relations

    sha256_to_stored = build_sha256_to_stored(llm_entities, ruler_entity_ids)

    # --- 4. Write net-new entities (those NOT already in ruler_entity_ids) ---
    net_new_ids: Set[str] = set()
    new_entity_count = 0
    ruler_names_lower = {n.lower() for n in ruler_entity_ids}

    for entity in llm_entities:
        name = (entity.metadata.get("name") or "").strip()
        if not name or name.lower() in ruler_names_lower:
            continue
        # Net-new entity — write to graph via simple add()
        try:
            result = memory._crud.add(entity)
            if isinstance(result, dict):
                net_new_node_id = result.get("memory_node_id") or entity.item_id
            else:
                net_new_node_id = result or entity.item_id
            if net_new_node_id:
                net_new_ids.add(net_new_node_id)
                # Update sha256→stored so relations can resolve this entity too
                if entity.item_id:
                    sha256_to_stored[entity.item_id] = net_new_node_id
            new_entity_count += 1
        except Exception as exc:
            logger.warning("Extract job: failed to write net-new entity '%s': %s", name, exc)

    # --- 5. Write resolved relations ---
    resolved_relations = filter_valid_relations(llm_relations, sha256_to_stored, net_new_ids)
    new_relation_count = 0
    for rel in resolved_relations:
        try:
            memory._graph.add_edge(
                rel["source_id"],
                rel["target_id"],
                edge_type=rel["relation_type"],
                properties={},
            )
            new_relation_count += 1
        except Exception as exc:
            logger.warning(
                "Extract job: failed to write relation %s->%s (%s): %s",
                rel["source_id"],
                rel["target_id"],
                rel["relation_type"],
                exc,
            )

    # --- 6+7. Update EntityPattern nodes + notify ruler hot-reload ---
    # Guarded by enable_ontology: when False (e.g. SmartMemory(enable_ontology=False)),
    # skip all ontology graph writes and Redis pub/sub entirely.
    new_pattern_count = 0
    if enable_ontology:
        try:
            from smartmemory.graph.ontology_graph import OntologyGraph

            ontology_graph = OntologyGraph(workspace_id=workspace_id)
            for entity in llm_entities:
                name = (entity.metadata.get("name") or "").strip()
                entity_type = entity.metadata.get("entity_type") or entity.memory_type or "concept"
                confidence = float(entity.metadata.get("confidence", 0.0))

                # Pre-write quality gates (confidence, length, blocklist)
                if not name or len(name) < _MIN_NAME_LENGTH:
                    continue
                if name.lower() in _BLOCKLIST:
                    continue
                if confidence < _MIN_CONFIDENCE:
                    continue

                added = ontology_graph.add_entity_pattern(
                    name=name,
                    label=entity_type,
                    confidence=confidence,
                    workspace_id=workspace_id,
                    is_global=False,
                    source="llm_discovery",
                )
                if added:
                    new_pattern_count += 1
        except Exception as exc:
            logger.warning("Extract job: failed to update entity patterns for item_id=%s: %s", item_id, exc)

        if new_pattern_count > 0 and redis_client is not None:
            try:
                from smartmemory.ontology.pattern_manager import PatternManager

                channel = f"{PatternManager.RELOAD_CHANNEL_PREFIX}:{workspace_id}"
                redis_client.publish(channel, "reload")
                logger.debug("Extract job: published ruler reload on %s", channel)
            except Exception as exc:
                logger.warning("Extract job: failed to publish ruler reload: %s", exc)
    else:
        logger.debug("Extract job: ontology disabled; skipping EntityPattern writes for item_id=%s", item_id)

    logger.info(
        "Extract job ok: item_id=%s new_entities=%d new_relations=%d new_patterns=%d",
        item_id,
        new_entity_count,
        new_relation_count,
        new_pattern_count,
    )
    return {
        "status": "ok",
        "new_entities": new_entity_count,
        "new_relations": new_relation_count,
        "new_patterns": new_pattern_count,
    }
