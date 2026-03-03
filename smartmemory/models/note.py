from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


@dataclass
class Note(MemoryBaseModel):
    """A note node in the memory graph."""
    title: str = ""
    item_id: Optional[str] = None
    content: str = ""
    created_at: Optional[datetime] = field(default_factory=datetime.now)

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "Note":
        return cls(
            item_id=item.item_id,
            content=item.content,
            title=item.metadata.get("title", ""),
            created_at=item.metadata.get("created_at", datetime.now(UTC)),
        )

    def to_memory_item(self) -> MemoryItem:
        meta = {"title": self.title, "created_at": self.created_at}
        kwargs = dict(content=self.content, type="zettel", metadata=meta)
        if self.item_id:
            kwargs["item_id"] = self.item_id
        return MemoryItem(**kwargs)
