import hashlib
import uuid
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.status import StatusLoggerMixin
from smartmemory.utils import flatten_dict, unflatten_dict


# --- Memory Types ---
# Valid memory types for the system
MEMORY_TYPES = {
    "semantic",  # Facts, knowledge, concepts
    "episodic",  # Events, experiences, interactions
    "procedural",  # Skills, how-to, procedures
    "working",  # Short-term, active context
    "zettel",  # Zettelkasten notes
    "reasoning",  # Chain-of-thought traces (System 2)
    "opinion",  # Beliefs with confidence scores (Phase 3)
    "observation",  # Synthesized entity summaries (Phase 3)
    "decision",  # Conclusions, inferences, choices with provenance (Phase 4)
    "code",  # Source code entities (classes, functions, routes) from code indexing
}


# --- MemoryItem ---
@dataclass(init=False)
class MemoryItem(MemoryBaseModel, StatusLoggerMixin):
    """
    Base class for an item stored in any memory store. Includes bi-temporal data.
    Supports multiple external references via metadata['external_refs'] (list of dicts with 'ref' and 'type').
    For backward compatibility, also supports single external_ref/external_type fields.
    """

    # Dataclass fields - all public, immutability enforced via __setattr__
    memory_type: str = "semantic"  # Default type is now 'semantic'; canonical removed
    item_id: str = dc_field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None

    group_id: Optional[str] = None
    valid_start_time: Optional[datetime] = None
    valid_end_time: Optional[datetime] = None
    transaction_time: datetime = dc_field(default_factory=lambda: datetime.now(timezone.utc))

    entities: Optional[list] = None
    relations: Optional[list] = None
    metadata: dict = dc_field(default_factory=dict)  # Arbitrary metadata

    # Track which fields are immutable after first set
    _immutable_fields: set = dc_field(default_factory=lambda: {"content", "embedding"}, repr=False, init=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Enforce immutability for specific fields after initial set.

        Fields in _immutable_fields (content, embedding) can only be set once.
        Attempting to modify them after initial set raises ValueError.
        """
        # Allow setting _immutable_fields itself during init
        if name == "_immutable_fields":
            object.__setattr__(self, name, value)
            return

        # Check if field is immutable and already set
        if hasattr(self, "_immutable_fields") and name in self._immutable_fields:
            if hasattr(self, name):
                current_value = object.__getattribute__(self, name)
                # Allow setting if current value is None/empty or same value (idempotent)
                if current_value and current_value != value:
                    raise ValueError(f"{name} has already been set and cannot be modified.")

        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, name, value)

    def to_dict(self):
        """Override to_dict to include memory_type property."""
        data = super().to_dict()
        data["memory_type"] = self.memory_type
        # Explicitly include embedding if present
        if self.embedding is not None:
            data["embedding"] = self.embedding
        # Remove internal tracking field
        data.pop("_immutable_fields", None)
        return data

    def __init__(
        self,
        *,
        content: Optional[str] = None,
        type: Optional[str] = None,
        memory_type: Optional[str] = None,
        item_id: Optional[str] = None,
        group_id: Optional[str] = None,
        valid_start_time: Optional[datetime] = None,
        valid_end_time: Optional[datetime] = None,
        transaction_time: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
        entities: Optional[list] = None,
        relations: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize immutability tracking first
        object.__setattr__(self, "_immutable_fields", {"content", "embedding"})

        # Initialize core fields with defaults
        self.memory_type = memory_type or type or "semantic"
        self.content = content if content is not None else ""
        self.item_id = item_id or str(uuid.uuid4())
        self.group_id = group_id
        self.valid_start_time = valid_start_time
        self.valid_end_time = valid_end_time

        # Normalize transaction_time
        tx = transaction_time
        if isinstance(tx, str):
            try:
                tx = datetime.fromisoformat(tx)
            except Exception:
                tx = None
        if tx is None:
            tx = datetime.now(timezone.utc)
        elif tx.tzinfo is None:
            tx = tx.replace(tzinfo=timezone.utc)
        self.transaction_time = tx

        # Set embedding
        self.embedding = embedding

        self.entities = entities
        self.relations = relations
        # Metadata and group_id coherence
        meta: Dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}
        if self.group_id is None and "group_id" in meta:
            self.group_id = meta.get("group_id")
        if self.group_id is not None:
            meta["group_id"] = self.group_id
        # Absorb known extras into metadata to avoid constructor errors
        for k, v in kwargs.items():
            if k not in meta:
                meta[k] = v
        self.metadata = meta

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def text_to_dummy_embedding(text, dim=32):
        """Generate a simple deterministic embedding from text (demo only)."""
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[:dim]]

    @staticmethod
    def compute_content_hash(content):
        if content is None:
            return None
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def from_content(cls, content, **kwargs):
        if "item_id" not in kwargs or kwargs["item_id"] is None:
            kwargs["item_id"] = cls.compute_content_hash(content)
        kwargs["content"] = content
        return MemoryItem(**kwargs)

    @classmethod
    def from_entity_edge(cls, edge):
        """Create a MemoryItem from a Graphiti EntityEdge, preserving all relevant info."""
        return MemoryItem(
            content=getattr(edge, "fact", None),
            item_id=getattr(edge, "item_id", None),
            metadata={
                "source_node": getattr(edge, "source_node", None),
                "target_node": getattr(edge, "target_node", None),
                "label": getattr(edge, "label", None),
                "score": getattr(edge, "score", None),
                "valid_at": getattr(edge, "valid_at", None),
                "embedding": getattr(edge, "embedding", None),
                "entity_edge": edge,
                "node_uuid": getattr(edge, "item_id", None),
            },
        )

    @classmethod
    def from_node(cls, node: dict) -> "MemoryItem":
        """
        Create a MemoryItem from a node dict (e.g., from the graph or vector store).

        Now simplified since field names match public API (no protected fields).
        Separates known fields from metadata, handles type conversion.
        """
        from smartmemory.utils.serialization import MemoryItemSerializer

        return MemoryItemSerializer.from_storage(cls, node)

    def to_vector_store(self) -> dict:
        """
        Convert this MemoryItem to a vector store item dict following the canonical mapping.

        Returns a dict with the following structure:
        {
            "id": str,              # item_id
            "content": str,         # content field
            "vector": List[float],  # embedding field
            "metadata": {           # all metadata fields
                "name": str,
                "type": str,
                "links": List[str],
                "memory_types": List[str],
                # All flattened metadata using double underscores
            }
        }
        """
        result = {
            "id": self.item_id,
            "content": self.content,
            "vector": self.embedding,  # Use embedding field for vector
        }

        # Prepare metadata dict
        metadata = {}

        # Add standard fields to metadata
        if isinstance(self.metadata, dict) and "name" in self.metadata and self.metadata.get("name") is not None:
            metadata["name"] = self.metadata.get("name")

        # Canonical memory type
        metadata["memory_type"] = self.memory_type

        if isinstance(self.metadata, dict) and "links" in self.metadata and self.metadata.get("links") is not None:
            metadata["links"] = self.metadata.get("links")

        if (
            isinstance(self.metadata, dict)
            and "memory_types" in self.metadata
            and self.metadata.get("memory_types") is not None
        ):
            metadata["memory_types"] = self.metadata.get("memory_types")
        else:
            # Default to [memory_type] if memory_types not specified
            metadata["memory_types"] = [self.memory_type]

        # Add all other metadata with flattening
        if hasattr(self, "metadata") and self.metadata:
            # Flatten nested dictionaries using double underscores
            flat_metadata = flatten_dict(self.metadata)
            metadata.update(flat_metadata)

        result["metadata"] = metadata
        return result

    @classmethod
    def from_vector_store(cls, vector_item: dict) -> "MemoryItem":
        """
        Create a MemoryItem from a vector store item dict.

        Expected structure:
        {
            "id": str,              # maps to item_id
            "content": str,         # maps to content
            "vector": List[float],  # maps to embedding
            "metadata": {           # contains all other fields
                "name": str,
                "memory_type": str,
                "links": List[str],
                "memory_types": List[str],
                # Plus any flattened metadata fields
            }
        }
        """
        # Extract core fields
        item_id = vector_item.get("id")
        content = vector_item.get("content")
        embedding = vector_item.get("vector")

        # Extract metadata or initialize empty dict
        metadata = vector_item.get("metadata") or {}.copy()

        # Extract standard fields from metadata
        name = metadata.pop("name", None)
        mem_type = metadata.pop("memory_type", None) or "semantic"
        links = metadata.pop("links", None)
        memory_types = metadata.pop("memory_types", [mem_type])  # Default to [memory_type] if not present

        # Unflatten remaining metadata (handles double underscore notation)
        unflattened_metadata = unflatten_dict(metadata)

        # Merge name/links/memory_types back into metadata for dataclass constructor
        if name is not None:
            unflattened_metadata["name"] = name
        if links is not None:
            unflattened_metadata["links"] = links
        if memory_types is not None:
            unflattened_metadata["memory_types"] = memory_types
        # Create MemoryItem with all fields
        return cls(
            item_id=item_id, content=content, embedding=embedding, memory_type=mem_type, metadata=unflattened_metadata
        )

    def to_node(self) -> dict:
        """
        Convert this MemoryItem to a node dict using the standard serializer.
        Delegates to MemoryItemSerializer.to_storage for consistency.
        """
        from smartmemory.utils.serialization import MemoryItemSerializer

        return MemoryItemSerializer.to_storage(self)

    def to_serializable_dict(self) -> dict:
        """
        Return a dictionary suitable for JSON serialization (API responses).
        Handles numpy array conversion for embeddings.
        """
        d = self.to_dict()
        if "embedding" in d and d["embedding"] is not None:
            # Handle numpy arrays if present
            if hasattr(d["embedding"], "tolist"):
                d["embedding"] = d["embedding"].tolist()
        return d

    @property
    def display_text(self) -> str:
        """Human-readable summary — guards content=None for code items.

        For code memory items (``memory_type="code"``) that have no ``content``,
        returns ``"<name> (<file_path>)"`` from the node's metadata.  Falls back
        to just ``name`` if ``file_path`` is absent.

        For all other memory types, returns ``content`` (or empty string when
        content is None).
        """
        if self.memory_type == "code" and not self.content:
            name = (self.metadata or {}).get("name", "")
            file_path = (self.metadata or {}).get("file_path", "")
            if name and file_path:
                return f"{name} ({file_path})"
            if name:
                return name
        return self.content or ""

    def __repr__(self) -> str:
        if self.valid_start_time is not None:
            valid_time_str = f"{self.valid_start_time.isoformat()}"
        else:
            valid_time_str = "None"
        if self.valid_end_time:
            valid_time_str += f" -> {self.valid_end_time.isoformat()}"
        else:
            valid_time_str += " -> present"
        tx_time_str = self.transaction_time.isoformat() if getattr(self, "transaction_time", None) else "None"
        return f"MemoryItem(id={self.item_id}, valid_time='{valid_time_str}', transaction_time={tx_time_str}, content='{str(self.content)[:30]}...')"
