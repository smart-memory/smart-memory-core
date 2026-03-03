"""
CanonicalMemoryStore: Read-only, append-only store for raw canonical MemoryItems with bitemporal metadata.
Supports soft-delete (prune) and deep retrieval for audit or advanced reasoning.
All derived memory types should reference this canonical layer.
"""
from datetime import datetime, UTC
from typing import List, Optional

from smartmemory.memory.base import MemoryBase
from smartmemory.models.memory_item import MemoryItem


class CanonicalMemoryStore(MemoryBase):
    """
    In-memory canonical store. Uses a dict for storage and does not use a backend store.
    """

    def __init__(self, *args, **kwargs):
        self._items = {}  # item_id -> MemoryItem
        self._pruned = set()  # soft-deleted item_ids
        super().__init__(*args, **kwargs)

    def add(self, item: MemoryItem, **kwargs):
        # Only allow append, never overwrite
        if item.item_id not in self._items:
            self._items[item.item_id] = item

    def get(self, item_id: str, deep: bool = False) -> Optional[MemoryItem]:
        # By default, pruned items are hidden unless deep retrieval is requested
        if item_id in self._pruned and not deep:
            return None
        return self._items.get(item_id)

    def all(self, include_pruned: bool = False) -> List[MemoryItem]:
        # Return all non-pruned items by default
        if include_pruned:
            return list(self._items.values())
        return [item for iid, item in self._items.items() if iid not in self._pruned]

    def prune(self, item_id: str, reason: str = None):
        # Soft-delete: mark as pruned, add timestamp and reason
        if item_id in self._items:
            self._pruned.add(item_id)
            item = self._items[item_id]
            meta = dict(item.metadata)
            meta["pruned_at"] = datetime.now(UTC).isoformat()
            if reason:
                meta["prune_reason"] = reason
            self._items[item_id] = MemoryItem(content=item.content, metadata=meta)

    def is_pruned(self, item_id: str) -> bool:
        return item_id in self._pruned

    def deep_retrieve(self, item_id: str) -> Optional[MemoryItem]:
        # Always returns the item, even if pruned
        return self._items.get(item_id)

    def clear(self):
        self._items.clear()
        self._pruned.clear()

    # Prevent modification or deletion of existing items
    def update(self, *args, **kwargs):
        raise NotImplementedError("CanonicalMemoryStore is read-only. Use derived stores for updates.")

    def delete(self, *args, **kwargs):
        raise NotImplementedError("CanonicalMemoryStore is append-only. Use prune() for soft-delete.")
