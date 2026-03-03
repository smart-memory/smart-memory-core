from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


@dataclass
class Procedure(MemoryBaseModel):
    """A procedure node in the memory graph."""
    name: str = ""
    item_id: Optional[str] = None
    description: str = ""
    created_at: Optional[datetime] = field(default_factory=datetime.now)

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> "Procedure":
        return cls(
            item_id=item.item_id,
            name=item.content or item.metadata.get("name", ""),
            description=item.metadata.get("description", ""),
            created_at=item.metadata.get("created_at", datetime.now(UTC)),
        )

    def to_memory_item(self) -> MemoryItem:
        kwargs = dict(
            content=self.name,
            type="procedure",
            metadata={"description": self.description, "created_at": self.created_at},
        )
        if self.item_id:
            kwargs["item_id"] = self.item_id
        return MemoryItem(**kwargs)
