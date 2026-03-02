import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from smartmemory.models.base import MemoryBaseModel


@dataclass
class Node(MemoryBaseModel):
    """Canonical node representation for smart-memory-core graph operations.
    
    Provides a unified interface for all node types across the system.
    """
    id: str
    name: str
    type: Optional[str] = None
    label: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    # Temporal attributes for versioning
    created_at: Optional[datetime] = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = field(default=None)

    def __post_init__(self):
        """Ensure consistent node state."""
        if not self.label:
            self.label = self.type or "Node"
        if self.updated_at is None:
            self.updated_at = self.created_at

    @classmethod
    def from_memory_item(cls, memory_item) -> 'Node':
        """Create a Node from a MemoryItem."""
        return cls(
            id=memory_item.item_id,
            name=getattr(memory_item, 'title', '') or memory_item.content[:50],
            type="memory",
            label="MemoryItem",
            properties={
                'content': memory_item.content,
                'metadata': memory_item.metadata,
                'memory_type': getattr(memory_item, 'memory_type', 'semantic')
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'label': self.label,
            'properties': self.properties,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class Edge(MemoryBaseModel):
    """Canonical edge representation for smart-memory-core graph operations.
    
    Represents relationships between nodes with rich metadata.
    """
    id: str
    name: str
    type: Optional[str] = None
    label: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    # Source and target node references
    source_id: Optional[str] = None
    target_id: Optional[str] = None

    # Relationship strength/confidence
    weight: float = field(default=1.0)
    confidence: float = field(default=1.0)

    # Temporal attributes
    created_at: Optional[datetime] = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = field(default=None)

    def __post_init__(self):
        """Ensure consistent edge state."""
        if not self.label:
            self.label = self.type or self.name
        if self.updated_at is None:
            self.updated_at = self.created_at
        if not self.id and self.source_id and self.target_id:
            self.id = f"{self.source_id}_{self.name}_{self.target_id}"

    @classmethod
    def from_graph_edge(cls, source_id: str, target_id: str, edge_type: str, properties: Dict[str, Any] = None) -> 'Edge':
        """Create an Edge from graph backend edge data."""
        props = properties or {}
        return cls(
            id=f"{source_id}_{edge_type}_{target_id}",
            name=edge_type,
            type=edge_type,
            source_id=source_id,
            target_id=target_id,
            properties=props,
            weight=props.get('weight', 1.0),
            confidence=props.get('confidence', 1.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'label': self.label,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'weight': self.weight,
            'confidence': self.confidence,
            'properties': self.properties,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class Triple(MemoryBaseModel):
    """Represents a semantic triple with optional rich node/edge references.
    
    Can work in two modes:
    1. String-based (legacy): subject/predicate/object as strings
    2. Rich-typed: subject_node/predicate_edge/object_node as full objects
    """
    # Legacy string-based fields (for backward compatibility)
    subject: str
    predicate: str
    object: str
    properties: Dict[str, Any] = field(default_factory=dict)

    # Rich typed fields (optional, for enhanced functionality)
    subject_node: Optional['Node'] = field(default=None)
    predicate_edge: Optional['Edge'] = field(default=None)
    object_node: Optional['Node'] = field(default=None)

    def __post_init__(self):
        """Auto-populate rich types from strings if not provided."""
        if self.subject_node is None and self.subject:
            self.subject_node = Node(
                id=self.subject,
                name=self.subject,
                type="auto_generated",
                label=None,
                properties={}
            )

        if self.object_node is None and self.object:
            self.object_node = Node(
                id=self.object,
                name=self.object,
                type="auto_generated",
                label=None,
                properties={}
            )

        if self.predicate_edge is None and self.predicate:
            self.predicate_edge = Edge(
                id=f"{self.subject}_{self.predicate}_{self.object}",
                name=self.predicate,
                type=self.predicate,
                label=None,
                properties=self.properties.copy()
            )

    @classmethod
    def from_nodes_and_edge(cls, subject_node: 'Node', predicate_edge: 'Edge', object_node: 'Node') -> 'Triple':
        """Create a Triple from rich Node and Edge objects."""
        return cls(
            subject=subject_node.id,
            predicate=predicate_edge.name,
            object=object_node.id,
            properties=predicate_edge.properties.copy(),
            subject_node=subject_node,
            predicate_edge=predicate_edge,
            object_node=object_node
        )

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to simple (subject, predicate, object) tuple."""
        return (self.subject, self.predicate, self.object)

    def get_nodes(self) -> list['Node']:
        """Get all nodes involved in this triple."""
        nodes = []
        if self.subject_node:
            nodes.append(self.subject_node)
        if self.object_node:
            nodes.append(self.object_node)
        return nodes

    def get_edge(self) -> Optional['Edge']:
        """Get the edge representing the predicate."""
        return self.predicate_edge


@dataclass
class MemoryPath(MemoryBaseModel):
    """
    Represents a multi-hop path (sequence of triples) in the memory graph.
    Only stores triples; nodes and edges can be derived from triples if needed.
    """
    path_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    triples: list[Triple] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    path_type: str = field(default="memory_path")

    def get_nodes(self) -> list[str]:
        nodes = []
        seen = set()
        for triple in self.triples:
            for node in (triple.subject, triple.object):
                if node not in seen:
                    nodes.append(node)
                    seen.add(node)
        return nodes

    def get_edges(self) -> list[str]:
        return [triple.predicate for triple in self.triples]

    def length(self) -> int:
        return len(self.triples)

    def is_empty(self) -> bool:
        return len(self.triples) == 0

    def contains_node(self, item_id: str) -> bool:
        return any(item_id == t.subject or item_id == t.object for t in self.triples)

    def contains_edge(self, predicate: str) -> bool:
        return any(predicate == t.predicate for t in self.triples)

    def contains_triple(self, triple: Triple) -> bool:
        return any(triple == t for t in self.triples)

    def as_tuples(self) -> list[tuple[str, str, str]]:
        return [(t.subject, t.predicate, t.object) for t in self.triples]

    def reverse(self) -> "MemoryPath":
        return MemoryPath(
            path_id=str(uuid.uuid4()),
            triples=list(reversed(self.triples)),
            created_at=datetime.now(timezone.utc),
            metadata=self.metadata.copy(),
            path_type=self.path_type,
        )

    def subpath(self, start: int, end: int) -> "MemoryPath":
        return MemoryPath(
            path_id=str(uuid.uuid4()),
            triples=self.triples[start:end],
            created_at=datetime.now(timezone.utc),
            metadata=self.metadata.copy(),
            path_type=self.path_type,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, MemoryPath):
            return False
        return self.triples == other.triples

    def shares_node_with(self, other: "MemoryPath") -> bool:
        return bool(set(self.get_nodes()) & set(other.get_nodes()))

    def __repr__(self):
        return f"MemoryPath(path_id={self.path_id}, triples={self.triples})"
