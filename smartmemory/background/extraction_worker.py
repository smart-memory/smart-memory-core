"""Tier 2 async extraction worker — runs LLM extraction on items stored by Tier 1.

Tier 1 (sync, ~4ms): spaCy + EntityRuler → stores memory, enqueues item_id.
Tier 2 (this module, ~740ms): loads item, runs LLMSingleExtractor, diffs against
ruler entities already stored, writes net-new entities + resolved relations,
then updates EntityPattern nodes for the self-learning EntityRuler loop.
"""
from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any, Dict, Optional, Set

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)

# Pre-write quality gates for entity pattern candidates
_MIN_CONFIDENCE = 0.8
_MIN_NAME_LENGTH = 3
_BLOCKLIST: Set[str] = {"it", "this", "that", "the", "a", "an", "he", "she", "they", "we"}

# CORE-EXT-1c: Relation label tracking quality gates
_RELATION_LABEL_BLOCKLIST: Set[str] = {"is", "has", "does", "was", "are", "were", "been", "be"}


def _get_entity_type(entity_id: str | None, entities: list) -> str | None:
    """Look up entity_type by item_id from the LLM entity list."""
    if not entity_id:
        return None
    for e in entities:
        if hasattr(e, "item_id") and e.item_id == entity_id:
            return e.metadata.get("entity_type") or e.memory_type
    return None


def _get_content_from_item(item: Any) -> str:
    """Extract normalized text content from a stored item."""
    if isinstance(item, MemoryItem):
        return item.content or ""
    if isinstance(item, dict):
        return item.get("content", "") or ""
    return ""


def _run_llm_extraction(content: str) -> Dict[str, Any]:
    """Run the Tier 2 extractor without performing any graph writes."""
    try:
        from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor

        extractor = LLMSingleExtractor()
        extraction = extractor.extract(content)
    except Exception as exc:
        logger.warning("Extract job: LLM extraction failed: %s", exc)
        return {"status": "llm_failed", "extraction": None}

    return {"status": "ok", "extraction": extraction}


def process_extract_job(
    memory: Any,
    payload: Dict[str, Any],
    redis_client: Optional[Any] = None,
    item_override: Optional[Any] = None,
    extraction_override: Optional[Dict[str, Any]] = None,
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
    item = item_override if item_override is not None else memory.get(item_id)
    if item is None:
        logger.warning("Extract job: item_id=%s not found", item_id)
        return {"status": "item_not_found", "new_entities": 0, "new_relations": 0, "new_patterns": 0}

    content = _get_content_from_item(item)

    if not content.strip():
        return {"status": "no_text", "new_entities": 0, "new_relations": 0, "new_patterns": 0}

    # --- 2. Run LLM extraction ---
    if extraction_override is None:
        llm_result = _run_llm_extraction(content)
        if llm_result["status"] != "ok":
            logger.warning("Extract job: LLM extraction failed for item_id=%s", item_id)
            return {"status": "llm_failed", "new_entities": 0, "new_relations": 0, "new_patterns": 0}
        extraction = llm_result["extraction"] or {}
    else:
        extraction = extraction_override

    if extraction is None:
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

    backend = getattr(getattr(memory, "_graph", None), "backend", None)
    write_context = nullcontext()
    if backend is not None:
        transaction_context = getattr(backend, "transaction_context", None)
        if callable(transaction_context):
            write_context = transaction_context()

    new_relation_count = 0

    with write_context:
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

        # --- 5. Write resolved relations with ONTO-PUB-3 metadata ---
        resolved_relations = filter_valid_relations(llm_relations, sha256_to_stored, net_new_ids)

        # CORE-EXT-1c: Create per-job ontology graph + normalizer with workspace overlays.
        # ontology_graph is also reused by Steps 6, 7, and 7b below.
        from smartmemory.relations.normalizer import RelationNormalizer

        ontology_graph = None
        job_normalizer = RelationNormalizer()  # alias-only baseline
        if enable_ontology:
            try:
                from smartmemory.graph.ontology_graph import OntologyGraph

                ontology_graph = OntologyGraph(workspace_id=workspace_id)
            except Exception:
                pass
            if ontology_graph is not None:
                try:
                    from smartmemory.relations.overlays import load_workspace_overlays

                    ws_aliases, _ = load_workspace_overlays(ontology_graph)
                    if ws_aliases:
                        job_normalizer = RelationNormalizer(workspace_aliases=ws_aliases)
                except Exception:
                    pass  # graceful — seed aliases still work

        for rel in resolved_relations:
            try:
                raw_pred = rel.get("raw_predicate") or rel.get("relation_type", "related_to")
                canonical, norm_conf = job_normalizer.normalize(raw_pred)
                edge_type = canonical if norm_conf > 0 else "related_to"
                props = {
                    "created_from_triple": True,
                    "source_item": item_id,
                    "canonical_type": canonical,
                    "raw_predicate": raw_pred,
                    "normalization_confidence": norm_conf,
                }
                memory._graph.add_edge(
                    rel["source_id"],
                    rel["target_id"],
                    edge_type=edge_type,
                    properties=props,
                )
                new_relation_count += 1
            except Exception as exc:
                logger.warning(
                    "Extract job: failed to write relation %s->%s (%s): %s",
                    rel["source_id"],
                    rel["target_id"],
                    rel.get("relation_type", "?"),
                    exc,
                )

        # --- 6+7. Update EntityPattern nodes + notify ruler hot-reload ---
        # Guarded by enable_ontology: when False (e.g. SmartMemory(enable_ontology=False)),
        # skip all ontology graph writes and Redis pub/sub entirely.
        new_pattern_count = 0
        if enable_ontology and ontology_graph is not None:
            try:
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

        # --- 7b. Track novel relation labels (CORE-EXT-1c) ---
        new_relation_label_count = 0
        if enable_ontology and ontology_graph is not None:
            try:
                from smartmemory.relations.normalizer import _normalize_key

                for rel in llm_relations:
                    raw_pred = rel.get("raw_predicate") or rel.get("relation_type", "related_to")
                    _, norm_conf = job_normalizer.normalize(raw_pred)
                    if norm_conf > 0.0:
                        continue  # known type, skip

                    key = _normalize_key(raw_pred)
                    if not key or len(key) < 3:
                        continue
                    if key.lower() in _RELATION_LABEL_BLOCKLIST:
                        continue

                    src_type = _get_entity_type(rel.get("source_id"), llm_entities)
                    tgt_type = _get_entity_type(rel.get("target_id"), llm_entities)

                    added = ontology_graph.add_novel_relation_label(
                        name=key,
                        raw_examples=[raw_pred],
                        source_types=[src_type] if src_type else [],
                        target_types=[tgt_type] if tgt_type else [],
                        workspace_id=workspace_id,
                    )
                    if added:
                        new_relation_label_count += 1
            except Exception as exc:
                logger.warning("Extract job: failed to track novel relation labels: %s", exc)

    logger.info(
        "Extract job ok: item_id=%s new_entities=%d new_relations=%d new_patterns=%d new_relation_labels=%d",
        item_id,
        new_entity_count,
        new_relation_count,
        new_pattern_count,
        new_relation_label_count,
    )
    return {
        "status": "ok",
        "new_entities": new_entity_count,
        "new_relations": new_relation_count,
        "new_patterns": new_pattern_count,
        "new_relation_labels": new_relation_label_count,
    }
