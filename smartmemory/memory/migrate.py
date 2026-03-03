"""
Memory migration/conversion utilities for agentic memory system.
Supports moving or transforming items between semantic, episodic, procedural, and working memory types.
These are intended for use by background processes, enrichment daemons, or agentic workflows outside SmartMemory.
"""
from datetime import datetime, UTC
from typing import Optional

from smartmemory.models.memory_item import MemoryItem
from smartmemory.smart_memory import SmartMemory


def migrate_item(
        item_id: str,
        source: str,
        target: str,
        smart_memory: SmartMemory,
        remove_from_source: bool = False,
        transform_fn: Optional[callable] = None
) -> Optional[str]:
    """
    Move or copy an item from one memory type to another.
    Optionally transform the item before adding to the target store.
    """
    item = smart_memory.get(item_id, memory_type=source)
    if not item:
        return None
    if transform_fn:
        item = transform_fn(item)
    new_id = smart_memory.add(item, memory_type=target)
    if remove_from_source:
        # Remove from source store if requested
        store = getattr(smart_memory, source, None)
        if store and hasattr(store, 'remove'):
            store.remove(item_id)
    return new_id


def promote_working_to_semantic(item_id: str, smart_memory: SmartMemory, remove_from_source=True):
    return migrate_item(item_id, 'working', 'semantic', smart_memory, remove_from_source)


def promote_working_to_episodic(item_id: str, smart_memory: SmartMemory, remove_from_source=True):
    return migrate_item(item_id, 'working', 'episodic', smart_memory, remove_from_source)


def promote_episodic_to_semantic(item_id: str, smart_memory: SmartMemory, remove_from_source=False):
    return migrate_item(item_id, 'episodic', 'semantic', smart_memory, remove_from_source)


def promote_semantic_to_procedural(item_id: str, smart_memory: SmartMemory, remove_from_source=False, transform_fn=None):
    return migrate_item(item_id, 'semantic', 'procedural', smart_memory, remove_from_source, transform_fn)


def promote_any(item_id: str, source: str, target: str, smart_memory: SmartMemory, remove_from_source=False, transform_fn=None):
    return migrate_item(item_id, source, target, smart_memory, remove_from_source, transform_fn)


# Example transform function for extracting a procedure from a note
def extract_procedure_from_note(note: MemoryItem) -> MemoryItem:
    # This is a stub; in practice, use NLP or agent to extract step-by-step info
    return MemoryItem(content=note.content, metadata={**note.metadata, 'type': 'procedure'})


# --- Default transformation functions for migration semantics ---
def working_to_semantic(item: MemoryItem) -> MemoryItem:
    # Promote to semantic: enrich with type and timestamp if missing
    meta = dict(item.metadata)
    meta.setdefault("type", "semantic")
    meta.setdefault("promoted_at", datetime.now(UTC).isoformat())
    return MemoryItem(content=item.content, metadata=meta)


def working_to_episodic(item: MemoryItem) -> MemoryItem:
    # Promote to episodic: add type and created_at
    meta = dict(item.metadata)
    meta["type"] = "episode"
    meta["created_at"] = meta.get("created_at", datetime.now(UTC).isoformat())
    return MemoryItem(content=item.content, metadata=meta)


def episodic_to_semantic(item: MemoryItem) -> MemoryItem:
    # Extract main fact/summary from episode for semantic memory
    fact = item.metadata.get("summary", item.content)
    meta = {k: v for k, v in item.metadata.items() if k not in ["created_at", "context", "type"]}
    meta["type"] = "semantic"
    meta["extracted_from_episode"] = item.item_id
    return MemoryItem(content=fact, metadata=meta)


def semantic_to_procedural(item: MemoryItem) -> MemoryItem:
    # Extract procedure from note (stub: just relabels type)
    meta = dict(item.metadata)
    meta["type"] = "procedure"
    # In practice, parse steps from content
    return MemoryItem(content=item.content, metadata=meta)


def semantic_to_working(item: MemoryItem) -> MemoryItem:
    # Place a note into working memory (copy, possibly trim metadata)
    meta = {k: v for k, v in item.metadata.items() if k != "tags"}
    meta["type"] = "working"
    return MemoryItem(content=item.content, metadata=meta)


def procedural_to_semantic(item: MemoryItem) -> MemoryItem:
    # Convert a procedure to a descriptive fact
    meta = dict(item.metadata)
    meta["type"] = "semantic"
    meta["from_procedure"] = True
    return MemoryItem(content="Procedure: " + item.content, metadata=meta)


def procedural_to_working(item: MemoryItem) -> MemoryItem:
    # Place a procedure into working memory
    meta = dict(item.metadata)
    meta["type"] = "working"
    return MemoryItem(content=item.content, metadata=meta)


def episodic_to_working(item: MemoryItem) -> MemoryItem:
    # Place an episode into working memory (e.g., for immediate context)
    meta = dict(item.metadata)
    meta["type"] = "working"
    return MemoryItem(content=item.content, metadata=meta)

# You can now use these as transform_fn in migrate_item or promote_any
