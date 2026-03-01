"""PublicKnowledgeStore ABC and DI accessor.

The store provides pluggable access to canonical public entities (Wikidata).
Two implementations: SQLitePublicKnowledgeStore (local/bundled) and
FalkorDBPublicKnowledgeStore (service mode).

The ContextVar follows the same DI pattern as _cache_ctx in utils/cache.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar

from smartmemory.grounding.models import PublicEntity

_public_knowledge_ctx: ContextVar["PublicKnowledgeStore | None"] = ContextVar(
    "_public_knowledge_ctx", default=None
)


def get_public_knowledge_store() -> "PublicKnowledgeStore | None":
    """Retrieve the current PublicKnowledgeStore from context."""
    return _public_knowledge_ctx.get()


class PublicKnowledgeStore(ABC):
    """Abstract base for public knowledge entity storage.

    Implementations must be thread-safe. The version property must be
    monotonically increasing and increment on every absorb() call.
    """

    @abstractmethod
    def lookup_by_alias(self, surface_form: str) -> list[PublicEntity]:
        """Find entities matching a surface form (case-insensitive)."""
        ...

    @abstractmethod
    def lookup_by_qid(self, qid: str) -> PublicEntity | None:
        """Look up a specific entity by its Wikidata QID."""
        ...

    @abstractmethod
    def absorb(self, entity: PublicEntity) -> None:
        """Add or update an entity in the store. Increments version."""
        ...

    @abstractmethod
    def get_ruler_patterns(self) -> dict[str, str]:
        """Return surface_form → entity_type mapping for EntityRuler.

        When a surface form maps to multiple entities, select deterministically:
        highest confidence, tie-break by lowest QID.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of entities in the store."""
        ...

    @abstractmethod
    def snapshot_version(self) -> str:
        """Return the version string of the loaded snapshot."""
        ...

    @property
    @abstractmethod
    def version(self) -> int:
        """Monotonically increasing version counter. Incremented on absorb()."""
        ...
