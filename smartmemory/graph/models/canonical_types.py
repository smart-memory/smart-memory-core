"""
Canonical graph types migration utilities for smart-memory-core.

This module provides utilities to migrate existing graph operations to use
the canonical Node, Edge, and Triple types defined in memory_path.py.
"""

from typing import Dict, Any, List, Optional

from smartmemory.graph.core.memory_path import Node, Edge, Triple
from smartmemory.models.memory_item import MemoryItem


class GraphTypeMigrator:
    """Utility class to help migrate existing graph operations to canonical types."""

    @staticmethod
    def convert_backend_node_to_canonical(backend_node: Dict[str, Any]) -> Node:
        """Convert a backend node dictionary to canonical Node type."""
        # Handle nested properties structure (common in Neo4j/FalkorDB)
        props = backend_node.get("properties") or {} if "properties" in backend_node else backend_node

        return Node(
            id=props.get('item_id') or props.get('id') or str(props.get('_id', '')),
            name=props.get('name') or props.get('title') or props.get('content', '')[:50],
            type=props.get('type') or props.get('memory_type'),
            label=props.get('label'),
            properties=props
        )

    @staticmethod
    def convert_backend_edge_to_canonical(edge_data: Dict[str, Any]) -> Edge:
        """Convert backend edge data to canonical Edge type."""
        return Edge.from_graph_edge(
            source_id=edge_data.get('source') or edge_data.get('source_id'),
            target_id=edge_data.get('target') or edge_data.get('target_id'),
            edge_type=edge_data.get('type') or edge_data.get('edge_type') or edge_data.get('relationship'),
            properties=edge_data.get('properties') or {}
        )

    @staticmethod
    def convert_backend_triple_to_canonical(triple_tuple: tuple) -> Triple:
        """Convert a (subject, predicate, object) tuple to canonical Triple."""
        subject, predicate, obj = triple_tuple
        return Triple(
            subject=str(subject),
            predicate=str(predicate),
            object=str(obj)
        )

    @staticmethod
    def enhance_triple_with_nodes(triple: Triple, graph) -> Triple:
        """Enhance a string-based Triple with rich Node/Edge objects from graph."""
        try:
            # Fetch actual nodes from graph
            subject_node_data = graph.get_node(triple.subject)
            object_node_data = graph.get_node(triple.object)

            if subject_node_data:
                triple.subject_node = GraphTypeMigrator.convert_memory_item_to_node(subject_node_data)

            if object_node_data:
                triple.object_node = GraphTypeMigrator.convert_memory_item_to_node(object_node_data)

            # Create enhanced edge
            triple.predicate_edge = Edge(
                id=f"{triple.subject}_{triple.predicate}_{triple.object}",
                name=triple.predicate,
                type=triple.predicate,
                source_id=triple.subject,
                target_id=triple.object,
                properties=triple.properties
            )

        except Exception:
            # If enhancement fails, keep the basic string-based triple
            pass

        return triple

    @staticmethod
    def convert_memory_item_to_node(memory_item: MemoryItem) -> Node:
        """Convert a MemoryItem to canonical Node type."""
        return Node.from_memory_item(memory_item)


class CanonicalGraphAdapter:
    """Adapter to provide canonical type interfaces for existing graph operations."""

    def __init__(self, graph):
        """Initialize with existing SmartGraph instance."""
        self.graph = graph
        self.migrator = GraphTypeMigrator()

    def add_canonical_node(self, node: Node, **kwargs) -> Node:
        """Add a canonical Node to the graph."""
        result = self.graph.add_node(
            item_id=node.id,
            properties=node.to_dict(),
            **kwargs
        )
        return node

    def add_canonical_edge(self, edge: Edge, **kwargs) -> Edge:
        """Add a canonical Edge to the graph."""
        self.graph.add_edge(
            source_id=edge.source_id,
            target_id=edge.target_id,
            edge_type=edge.type or edge.name,
            properties=edge.to_dict(),
            **kwargs
        )
        return edge

    def add_canonical_triple(self, triple: Triple, **kwargs) -> Triple:
        """Add a canonical Triple to the graph."""
        # Ensure nodes exist
        if triple.subject_node:
            self.add_canonical_node(triple.subject_node, **kwargs)
        else:
            # Create basic node
            self.graph.add_node(triple.subject, {})

        if triple.object_node:
            self.add_canonical_node(triple.object_node, **kwargs)
        else:
            # Create basic node
            self.graph.add_node(triple.object, {})

        # Add edge
        if triple.predicate_edge:
            self.add_canonical_edge(triple.predicate_edge, **kwargs)
        else:
            # Use basic edge creation
            self.graph.add_edge(
                triple.subject,
                triple.object,
                triple.predicate,
                triple.properties,
                **kwargs
            )

        return triple

    def get_canonical_node(self, node_id: str) -> Optional[Node]:
        """Get a node as canonical Node type."""
        memory_item = self.graph.get_node(node_id)
        if memory_item:
            return self.migrator.convert_memory_item_to_node(memory_item)
        return None

    def get_canonical_edges_for_node(self, node_id: str) -> List[Edge]:
        """Get all edges for a node as canonical Edge types."""
        try:
            backend_edges = self.graph.get_edges_for_node(node_id)
            return [
                self.migrator.convert_backend_edge_to_canonical(edge_data)
                for edge_data in backend_edges
            ]
        except Exception:
            return []

    def get_canonical_triples_for_node(self, node_id: str) -> List[Triple]:
        """Get all triples involving a node as canonical Triple types."""
        edges = self.get_canonical_edges_for_node(node_id)
        triples = []

        for edge in edges:
            if edge.source_id == node_id:
                triple = Triple(
                    subject=edge.source_id,
                    predicate=edge.name,
                    object=edge.target_id,
                    properties=edge.properties
                )
            else:
                triple = Triple(
                    subject=edge.target_id,
                    predicate=edge.name,
                    object=edge.source_id,
                    properties=edge.properties
                )

            # Enhance with rich types
            triple = self.migrator.enhance_triple_with_nodes(triple, self.graph)
            triples.append(triple)

        return triples


def create_migration_helpers():
    """Create helper functions for common migration patterns."""

    def migrate_procedural_memory_relations():
        """Helper to migrate procedural memory Triple usage."""
        # This would be used in procedural.py
        pass

    def migrate_linking_component():
        """Helper to migrate linking component Triple usage."""
        # This would be used in linking.py
        pass

    def migrate_smartgraph_operations():
        """Helper to migrate SmartGraph Triple operations."""
        # This would be used in smartgraph.py
        pass

    return {
        'procedural': migrate_procedural_memory_relations,
        'linking': migrate_linking_component,
        'smartgraph': migrate_smartgraph_operations
    }
