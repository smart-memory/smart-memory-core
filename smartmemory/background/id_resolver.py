"""Id resolution between LLMSingleExtractor's SHA-256 id space and FalkorDB stored ids.

LLMSingleExtractor._process_entities() assigns SHA-256 item_ids (hashlib.sha256 of
"name|entity_type", first 16 chars) to MemoryItem objects. StoreStage._map_entity_ids()
assigns FalkorDB graph node ids, stored in PipelineState.entity_ids as {name -> graph_id}.

Relations from the LLM call reference SHA-256 ids which must be remapped to graph ids
before writing edges. Both id spaces are needed: sha256_to_stored for ruler entities
present in both spaces, net_new_ids for entities added only in this Tier 2 run.
"""
from __future__ import annotations

from smartmemory.models.memory_item import MemoryItem


def build_sha256_to_stored(
    llm_entities: list[MemoryItem],
    ruler_entity_ids: dict[str, str],
) -> dict[str, str]:
    """Map SHA-256 ids (LLM extractor space) to stored FalkorDB ids (Tier 1 space).

    Args:
        llm_entities: List[MemoryItem] from LLMSingleExtractor.extract()["entities"].
                      Accessed via entity.item_id (SHA-256) and entity.metadata["name"].
        ruler_entity_ids: state.entity_ids from the job payload — {name.lower(): graph_node_id}.
                          Populated by StoreStage._map_entity_ids() during Tier 1 ingest.

    Returns:
        {sha256_id: graph_node_id} for entities present in both id spaces.
    """
    remap: dict[str, str] = {}
    for entity in llm_entities:
        name = entity.metadata.get("name", "").lower()
        sha256_id = entity.item_id
        if name and sha256_id and name in ruler_entity_ids:
            stored_id = ruler_entity_ids[name]
            if stored_id:
                remap[sha256_id] = stored_id
    return remap


def filter_valid_relations(
    llm_relations: list[dict],
    sha256_to_stored: dict[str, str],
    net_new_ids: set[str],
) -> list[dict]:
    """Return relations where both endpoints resolve to known graph nodes.

    An endpoint is valid if it appears in sha256_to_stored (ruler entity remapped
    to a stored id) or in net_new_ids (entity written in this Tier 2 run).

    Args:
        llm_relations: List of relation dicts from LLMSingleExtractor.extract()["relations"].
                       Each dict: {source_id: sha256, target_id: sha256, relation_type: str}.
        sha256_to_stored: Output of build_sha256_to_stored().
        net_new_ids: Set of item_ids for net-new entities written in this Tier 2 run.

    Returns:
        List of relation dicts with source_id/target_id remapped to graph node ids.
    """
    valid_ids = set(sha256_to_stored.values()) | net_new_ids
    resolved = []
    for rel in llm_relations:
        source_id = sha256_to_stored.get(rel["source_id"], rel["source_id"])
        target_id = sha256_to_stored.get(rel["target_id"], rel["target_id"])
        if source_id in valid_ids and target_id in valid_ids:
            resolved.append({
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": rel["relation_type"],
            })
    return resolved
