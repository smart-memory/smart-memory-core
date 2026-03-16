"""
Runtime schema validation for graph operations.

Provides validation for node and edge schemas to ensure data integrity
and consistency across different memory types.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"  # Fail on any validation error
    WARNING = "warning"  # Log warnings but continue
    DISABLED = "disabled"  # No validation


@dataclass
class NodeSchema:
    """Schema definition for graph nodes."""
    node_type: str
    required_fields: Set[str]
    optional_fields: Set[str]
    field_types: Dict[str, type]
    validation_rules: Dict[str, callable] = None

    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = {}


@dataclass
class EdgeSchema:
    """Schema definition for graph edges."""
    edge_type: str
    source_node_types: Set[str]
    target_node_types: Set[str]
    required_properties: Set[str]
    optional_properties: Set[str]
    property_types: Dict[str, type]


class GraphSchemaValidator:
    """Runtime schema validator for graph operations."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.WARNING):
        self.validation_level = validation_level
        self.node_schemas: Dict[str, NodeSchema] = {}
        self.edge_schemas: Dict[str, EdgeSchema] = {}
        self._register_default_schemas()

    def _register_default_schemas(self):
        """Register default schemas for common memory types."""

        # Episodic memory node schema
        self.register_node_schema(NodeSchema(
            node_type="episode",
            required_fields={"item_id", "content"},
            optional_fields={"title", "description", "timestamp", "participants", "location", "outcome", "tags"},
            field_types={
                "item_id": str,
                "content": str,
                "title": str,
                "description": str,
                "timestamp": (str, type(None)),
                "participants": list,
                "location": str,
                "outcome": str,
                "tags": list
            }
        ))

        # Procedural memory node schema
        self.register_node_schema(NodeSchema(
            node_type="procedure",
            required_fields={"item_id", "name"},
            optional_fields={"description", "procedure_body", "steps", "reference_time"},
            field_types={
                "item_id": str,
                "name": str,
                "description": str,
                "procedure_body": str,
                "steps": list,
                "reference_time": (str, type(None))
            }
        ))

        # Semantic memory node schema
        self.register_node_schema(NodeSchema(
            node_type="semantic",
            required_fields={"item_id", "content"},
            optional_fields={"title", "description", "concepts", "relations", "confidence"},
            field_types={
                "item_id": str,
                "content": str,
                "title": str,
                "description": str,
                "concepts": (list, type(None)),
                "relations": (list, type(None)),
                "confidence": float
            }
        ))

        # Entity node schema
        self.register_node_schema(NodeSchema(
            node_type="Entity",
            required_fields={"item_id"},
            optional_fields={"name", "type", "description", "properties", "confidence"},
            field_types={
                "item_id": str,
                "name": (str, type(None)),
                "type": (str, type(None)),
                "description": (str, type(None)),
                "properties": (dict, type(None)),
                "confidence": (float, type(None))
            }
        ))

        # Lowercase alias for entity nodes (used by ingestion flow memory_type="entity")
        self.register_node_schema(NodeSchema(
            node_type="entity",
            required_fields={"item_id"},
            optional_fields={"name", "type", "description", "properties", "confidence"},
            field_types={
                "item_id": str,
                "name": (str, type(None)),
                "type": (str, type(None)),
                "description": (str, type(None)),
                "properties": (dict, type(None)),
                "confidence": (float, type(None))
            }
        ))

        # User preference node schema
        self.register_node_schema(NodeSchema(
            node_type="user_preference",
            required_fields={"item_id", "preference_type"},
            optional_fields={"value", "confidence", "context", "timestamp"},
            field_types={
                "item_id": str,
                "preference_type": str,
                "value": (str, type(None)),
                "confidence": (float, type(None)),
                "context": (str, type(None)),
                "timestamp": (str, type(None))
            }
        ))

        # REASONING TRACE NODE SCHEMA (System 2 Memory)
        self.register_node_schema(NodeSchema(
            node_type="reasoning",
            required_fields={"item_id", "content"},
            optional_fields={"trace_id", "steps", "task_context", "evaluation", "session_id", "has_explicit_markup", "artifact_ids"},
            field_types={
                "item_id": str,
                "content": str,
                "trace_id": (str, type(None)),
                "steps": (list, type(None)),
                "task_context": (dict, type(None)),
                "evaluation": (dict, type(None)),
                "session_id": (str, type(None)),
                "has_explicit_markup": (bool, type(None)),
                "artifact_ids": (list, type(None))
            }
        ))

        # DECISION NODE SCHEMA (Decision Memory - Phase 4)
        self.register_node_schema(NodeSchema(
            node_type="decision",
            required_fields={"item_id", "content"},
            optional_fields={
                "decision_id", "decision_type", "confidence", "evidence_ids", "contradicting_ids",
                "reinforcement_count", "contradiction_count", "source_type", "source_trace_id",
                "source_session_id", "context_snapshot", "status", "superseded_by",
                "domain", "tags", "pending_requirements", "created_at", "updated_at",
            },
            field_types={
                "item_id": str,
                "content": str,
                "decision_id": (str, type(None)),
                "decision_type": (str, type(None)),
                "confidence": (float, type(None)),
                "evidence_ids": (list, type(None)),
                "contradicting_ids": (list, type(None)),
                "reinforcement_count": (int, type(None)),
                "contradiction_count": (int, type(None)),
                "source_type": (str, type(None)),
                "source_trace_id": (str, type(None)),
                "source_session_id": (str, type(None)),
                "context_snapshot": (dict, type(None)),
                "status": (str, type(None)),
                "superseded_by": (str, type(None)),
                "domain": (str, type(None)),
                "tags": (list, type(None)),
                "created_at": (str, type(None)),
                "updated_at": (str, type(None)),
            }
        ))

        # ZETTELKASTEN NODE SCHEMAS - CRITICAL MISSING SCHEMAS

        # Zettel (note) node schema
        self.register_node_schema(NodeSchema(
            node_type="zettel",
            required_fields={"item_id", "content"},
            optional_fields={"title", "tags", "concepts", "wikilinks", "created_at", "updated_at", "author", "summary"},
            field_types={
                "item_id": str,
                "content": str,
                "title": (str, type(None)),
                "tags": (list, type(None)),
                "concepts": (list, type(None)),
                "wikilinks": (list, type(None)),
                "created_at": (str, type(None)),
                "updated_at": (str, type(None)),
                "author": (str, type(None)),
                "summary": (str, type(None))
            }
        ))

        # Tag node schema
        self.register_node_schema(NodeSchema(
            node_type="tag",
            required_fields={"item_id", "name"},
            optional_fields={"description", "color", "usage_count"},
            field_types={
                "item_id": str,
                "name": str,
                "description": (str, type(None)),
                "color": (str, type(None)),
                "usage_count": (int, type(None))
            }
        ))

        # Concept node schema
        self.register_node_schema(NodeSchema(
            node_type="concept",
            required_fields={"item_id", "content"},
            optional_fields={"name", "type", "description", "confidence", "frequency"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "type": (str, type(None)),
                "description": (str, type(None)),
                "confidence": (float, type(None)),
                "frequency": (int, type(None))
            }
        ))

        # Test content node schema (for performance testing)
        self.register_node_schema(NodeSchema(
            node_type="test_content",
            required_fields={"item_id", "content"},
            optional_fields={"type", "index", "metadata"},
            field_types={
                "item_id": str,
                "content": str,
                "type": (str, type(None)),
                "index": (int, type(None)),
                "metadata": (dict, type(None))
            }
        ))

        # User input node schema
        self.register_node_schema(NodeSchema(
            node_type="user_input",
            required_fields={"item_id", "content"},
            optional_fields={"timestamp", "session_context", "type"},
            field_types={
                "item_id": str,
                "content": str,
                "timestamp": str,
                
                "session_context": str,
                "type": str
            }
        ))

        # Assistant response node schema
        self.register_node_schema(NodeSchema(
            node_type="assistant_response",
            required_fields={"item_id", "content"},
            optional_fields={"timestamp", "session_context", "type"},
            field_types={
                "item_id": str,
                "content": str,
                "timestamp": str,
                
                "session_context": str,
                "type": str
            }
        ))

        # User preference node schema
        self.register_node_schema(NodeSchema(
            node_type="user_preference",
            required_fields={"item_id", "content"},
            optional_fields={"timestamp", "type", "preference_type"},
            field_types={
                "item_id": str,
                "content": str,
                "timestamp": str,
                
                "type": str,
                "preference_type": str
            }
        ))

        # Ontology node schemas for LLM-based entity extraction
        self.register_node_schema(NodeSchema(
            node_type="person",
            required_fields={"item_id", "content"},
            optional_fields={"name", "email", "phone", "age", "occupation", "skills", "interests", "location", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "email": (str, type(None)),
                "phone": (str, type(None)),
                "age": (int, type(None)),
                "occupation": (str, type(None)),
                "skills": (list, type(None)),
                "interests": (list, type(None)),
                "location": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        self.register_node_schema(NodeSchema(
            node_type="organization",
            required_fields={"item_id", "content"},
            optional_fields={"name", "industry", "size", "founded", "headquarters", "website", "type", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "industry": (str, type(None)),
                "size": (str, type(None)),
                "founded": (str, type(None)),
                "headquarters": (str, type(None)),
                "website": (str, type(None)),
                "type": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        self.register_node_schema(NodeSchema(
            node_type="location",
            required_fields={"item_id", "content"},
            optional_fields={"name", "address", "coordinates", "location_type", "capacity", "timezone", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "address": (str, type(None)),
                "coordinates": (str, type(None)),
                "location_type": (str, type(None)),
                "capacity": (int, type(None)),
                "timezone": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        self.register_node_schema(NodeSchema(
            node_type="concept",
            required_fields={"item_id", "content"},
            optional_fields={"name", "description", "category", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "description": (str, type(None)),
                "category": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        self.register_node_schema(NodeSchema(
            node_type="event",
            required_fields={"item_id", "content"},
            optional_fields={"name", "date", "location", "participants", "outcome", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "date": (str, type(None)),
                "location": (str, type(None)),
                "participants": (list, type(None)),
                "outcome": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        self.register_node_schema(NodeSchema(
            node_type="tool",
            required_fields={"item_id", "content"},
            optional_fields={"name", "version", "description", "category", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "version": (str, type(None)),
                "description": (str, type(None)),
                "category": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        self.register_node_schema(NodeSchema(
            node_type="skill",
            required_fields={"item_id", "content"},
            optional_fields={"name", "description", "category", "confidence", "source"},
            field_types={
                "item_id": str,
                "content": str,
                "name": (str, type(None)),
                "description": (str, type(None)),
                "category": (str, type(None)),
                "confidence": (float, type(None)),
                "source": (str, type(None)),
                
            }
        ))

        # Common edge schemas
        self.register_edge_schema(EdgeSchema(
            edge_type="HAS_ACTION",
            source_node_types={"episode"},
            target_node_types={"action"},
            required_properties=set(),
            optional_properties={"order", "timestamp"},
            property_types={"order": int, "timestamp": str}
        ))

        self.register_edge_schema(EdgeSchema(
            edge_type="HAS_STEP",
            source_node_types={"procedure"},
            target_node_types={"step"},
            required_properties=set(),
            optional_properties={"order", "condition"},
            property_types={"order": int, "condition": str}
        ))

        self.register_edge_schema(EdgeSchema(
            edge_type="RELATED",
            source_node_types={"episode", "procedure", "semantic"},
            target_node_types={"episode", "procedure", "semantic"},
            required_properties=set(),
            optional_properties={"strength", "type", "description"},
            property_types={"strength": float, "type": str, "description": str}
        ))

        self.register_edge_schema(EdgeSchema(
            edge_type="CO_RETRIEVED",
            source_node_types={"semantic", "episodic", "procedural", "zettel"},
            target_node_types={"semantic", "episodic", "procedural", "zettel"},
            required_properties=set(),
            optional_properties={"co_retrieval_count", "weight", "first_co_retrieved_at", "last_co_retrieved_at"},
            property_types={
                "co_retrieval_count": int,
                "weight": float,
                "first_co_retrieved_at": str,
                "last_co_retrieved_at": str,
            }
        ))

        # ZETTELKASTEN EDGE SCHEMAS - CRITICAL MISSING SCHEMAS

        # Links between notes (wikilinks)
        self.register_edge_schema(EdgeSchema(
            edge_type="LINKS_TO",
            source_node_types={"zettel"},
            target_node_types={"zettel"},
            required_properties=set(),
            optional_properties={"direction", "auto_created", "link_type", "strength"},
            property_types={"direction": str, "auto_created": bool, "link_type": str, "strength": float}
        ))

        # Backward links (for bidirectional linking)
        self.register_edge_schema(EdgeSchema(
            edge_type="LINKS_TO_BACK",
            source_node_types={"zettel"},
            target_node_types={"zettel"},
            required_properties=set(),
            optional_properties={"direction", "auto_created", "link_type", "strength"},
            property_types={"direction": str, "auto_created": bool, "link_type": str, "strength": float}
        ))

        # Note tagged with tag
        self.register_edge_schema(EdgeSchema(
            edge_type="TAGGED_WITH",
            source_node_types={"zettel"},
            target_node_types={"tag"},
            required_properties=set(),
            optional_properties={"confidence", "auto_created"},
            property_types={"confidence": float, "auto_created": bool}
        ))

        # Note mentions concept
        self.register_edge_schema(EdgeSchema(
            edge_type="MENTIONS",
            source_node_types={"zettel"},
            target_node_types={"concept"},
            required_properties=set(),
            optional_properties={"confidence", "frequency", "context"},
            property_types={"confidence": float, "frequency": int, "context": str}
        ))

        # Generic relation for dynamic connections
        self.register_edge_schema(EdgeSchema(
            edge_type="RELATES_TO",
            source_node_types={"zettel", "tag", "concept"},
            target_node_types={"zettel", "tag", "concept"},
            required_properties=set(),
            optional_properties={"relation_type", "strength", "confidence", "auto_created"},
            property_types={"relation_type": str, "strength": float, "confidence": float, "auto_created": bool}
        ))

        # REASONING TRACE EDGE SCHEMAS (System 2 Memory)

        # Reasoning trace causes/produces an artifact
        self.register_edge_schema(EdgeSchema(
            edge_type="CAUSES",
            source_node_types={"reasoning", "decision", "episodic"},
            target_node_types={"semantic", "episodic", "procedural", "zettel", "entity", "Entity", "decision"},
            required_properties=set(),
            optional_properties={"confidence", "timestamp", "trace_id", "mechanism", "lag_time_hours"},
            property_types={
                "confidence": float, "timestamp": str, "trace_id": str,
                "mechanism": str, "lag_time_hours": float,
            }
        ))

        # Inverse: artifact was caused by reasoning trace
        self.register_edge_schema(EdgeSchema(
            edge_type="CAUSED_BY",
            source_node_types={"semantic", "episodic", "procedural", "zettel", "entity", "Entity", "decision"},
            target_node_types={"reasoning", "decision"},
            required_properties=set(),
            optional_properties={"confidence", "timestamp", "trace_id"},
            property_types={"confidence": float, "timestamp": str, "trace_id": str}
        ))

        # DECISION MEMORY EDGE SCHEMAS (Phase 4)

        # Reasoning trace produced a decision
        self.register_edge_schema(EdgeSchema(
            edge_type="PRODUCED",
            source_node_types={"reasoning"},
            target_node_types={"decision", "semantic", "episodic", "procedural", "zettel"},
            required_properties=set(),
            optional_properties={"confidence", "step_index", "timestamp"},
            property_types={"confidence": float, "step_index": int, "timestamp": str}
        ))

        # Decision derived from evidence
        self.register_edge_schema(EdgeSchema(
            edge_type="DERIVED_FROM",
            source_node_types={"decision"},
            target_node_types={"semantic", "episodic", "reasoning", "decision", "opinion"},
            required_properties=set(),
            optional_properties={"weight", "role"},
            property_types={"weight": float, "role": str}
        ))

        # New decision supersedes old decision
        self.register_edge_schema(EdgeSchema(
            edge_type="SUPERSEDES",
            source_node_types={"decision"},
            target_node_types={"decision"},
            required_properties={"timestamp"},
            optional_properties={"reason"},
            property_types={"reason": str, "timestamp": str}
        ))

        # Decisions or facts in conflict
        self.register_edge_schema(EdgeSchema(
            edge_type="CONTRADICTS",
            source_node_types={"decision", "semantic", "opinion"},
            target_node_types={"decision", "semantic", "opinion"},
            required_properties=set(),
            optional_properties={"detected_at", "resolution"},
            property_types={"detected_at": str, "resolution": str}
        ))

        # Soft influence between decisions
        self.register_edge_schema(EdgeSchema(
            edge_type="INFLUENCES",
            source_node_types={"decision", "semantic", "episodic"},
            target_node_types={"decision", "semantic"},
            required_properties=set(),
            optional_properties={"weight"},
            property_types={"weight": float}
        ))

        # INFERENCE PROVENANCE (Wave 2 - Graph Quality)
        self.register_edge_schema(EdgeSchema(
            edge_type="INFERRED_FROM",
            source_node_types={"semantic", "episodic", "decision", "entity", "Entity"},
            target_node_types={"semantic", "episodic", "decision", "entity", "Entity"},
            required_properties=set(),
            optional_properties={"rule_name", "confidence", "inferred_at"},
            property_types={"rule_name": str, "confidence": float, "inferred_at": str}
        ))

        # RESIDUATION REQUIREMENT TRACKING (Wave 2 - Symbolic Reasoning)
        self.register_edge_schema(EdgeSchema(
            edge_type="REQUIRES",
            source_node_types={"decision"},
            target_node_types={"semantic", "episodic", "decision", "entity", "Entity"},
            required_properties=set(),
            optional_properties={"requirement_type", "description", "resolved"},
            property_types={"requirement_type": str, "description": str, "resolved": bool}
        ))

    def register_node_schema(self, schema: NodeSchema):
        """Register a node schema."""
        self.node_schemas[schema.node_type] = schema
        logger.debug(f"Registered node schema for type: {schema.node_type}")

    def register_edge_schema(self, schema: EdgeSchema):
        """Register an edge schema."""
        self.edge_schemas[schema.edge_type] = schema
        logger.debug(f"Registered edge schema for type: {schema.edge_type}")

    def validate_node(self, node_data: Dict[str, Any], node_type: str = None) -> bool:
        """Validate a node against its schema."""
        if self.validation_level == ValidationLevel.DISABLED:
            return True

        # Determine node type
        if node_type is None:
            node_type = node_data.get("memory_type") or node_data.get("type") or "unknown"

        # Normalize node type and handle common aliases used across the codebase
        # We lower-case for lookup and map well-known synonyms to registered schema keys.
        original_node_type = node_type
        node_type_lc = str(node_type).lower() if node_type is not None else "unknown"
        alias_map = {
            # Dual-node architecture and ingestion aliases
            "entity_node": "entity",
            # Memory type naming variations
            "episodic": "episode",
            "procedural": "procedure",
        }
        node_type = alias_map.get(node_type_lc, node_type_lc)

        # System node types created by internal infrastructure (e.g. VersionTracker)
        # are not user-facing and don't need schema validation.
        _SYSTEM_NODE_TYPES = {"version"}
        if node_type in _SYSTEM_NODE_TYPES:
            return True

        # Check if we have a schema for this node type
        if node_type not in self.node_schemas:
            message = f"No schema registered for node type: {original_node_type}"
            return self._handle_validation_error(message)

        schema = self.node_schemas[node_type]
        errors = []

        # Check required fields
        for field in schema.required_fields:
            if field not in node_data:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, value in node_data.items():
            if field in schema.field_types:
                expected_type = schema.field_types[field]
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not isinstance(value, expected_type):
                        errors.append(f"Field {field} has incorrect type. Expected {expected_type}, got {type(value)}")
                else:
                    # Single expected type
                    if not isinstance(value, expected_type):
                        errors.append(f"Field {field} has incorrect type. Expected {expected_type}, got {type(value)}")

        # Apply custom validation rules
        for field, rule in schema.validation_rules.items():
            if field in node_data:
                try:
                    if not rule(node_data[field]):
                        errors.append(f"Field {field} failed validation rule")
                except Exception as e:
                    errors.append(f"Validation rule for {field} threw exception: {e}")

        # Handle validation results
        if errors:
            message = f"Node validation failed for type {node_type}: {'; '.join(errors)}"
            return self._handle_validation_error(message)

        return True

    def validate_edge(self, source_type: str, target_type: str, edge_type: str,
                      properties: Dict[str, Any] = None) -> bool:
        """Validate an edge against its schema."""
        if self.validation_level == ValidationLevel.DISABLED:
            return True

        if properties is None:
            properties = {}

        # Check if we have a schema for this edge type
        if edge_type not in self.edge_schemas:
            message = f"No schema registered for edge type: {edge_type}"
            return self._handle_validation_error(message)

        schema = self.edge_schemas[edge_type]
        errors = []

        # Check source node type
        if source_type not in schema.source_node_types:
            errors.append(f"Invalid source node type {source_type} for edge {edge_type}. "
                          f"Expected one of: {schema.source_node_types}")

        # Check target node type
        if target_type not in schema.target_node_types:
            errors.append(f"Invalid target node type {target_type} for edge {edge_type}. "
                          f"Expected one of: {schema.target_node_types}")

        # Check required properties
        for prop in schema.required_properties:
            if prop not in properties:
                errors.append(f"Missing required property: {prop}")

        # Check property types
        for prop, value in properties.items():
            if prop in schema.property_types:
                expected_type = schema.property_types[prop]
                try:
                    # Handle complex types that might cause isinstance issues
                    if not self._check_type_compatibility(value, expected_type):
                        errors.append(f"Property {prop} has incorrect type. "
                                      f"Expected {expected_type}, got {type(value)}")
                except Exception as e:
                    # If type checking fails due to unhashable types, skip validation
                    # This prevents "unhashable type: 'dict'" errors
                    logger.debug(f"Skipping type validation for property {prop} due to: {e}")
                    continue

        # Handle validation results
        if errors:
            message = f"Edge validation failed for type {edge_type}: {'; '.join(errors)}"
            return self._handle_validation_error(message)

        return True

    def _check_type_compatibility(self, value, expected_type) -> bool:
        """Safely check type compatibility, handling complex types and unhashable dicts."""
        try:
            # Handle tuple types (e.g., (str, type(None)))
            if isinstance(expected_type, tuple):
                return any(isinstance(value, t) for t in expected_type)

            # Handle basic isinstance check
            return isinstance(value, expected_type)
        except (TypeError, ValueError):
            # If isinstance fails (e.g., with subscripted generics), do basic type check
            if expected_type == dict or (hasattr(expected_type, '__origin__') and expected_type.__origin__ == dict):
                return isinstance(value, dict)
            elif expected_type == list or (hasattr(expected_type, '__origin__') and expected_type.__origin__ == list):
                return isinstance(value, list)
            else:
                # For other complex types, assume compatibility
                return True

    def _handle_validation_error(self, message: str) -> bool:
        """Handle validation errors based on validation level."""
        if self.validation_level == ValidationLevel.STRICT:
            logger.error(message)
            raise ValueError(message)
        elif self.validation_level == ValidationLevel.WARNING:
            logger.warning(message)
            return False
        else:  # DISABLED
            return True

    def get_registered_schemas(self) -> Dict[str, Any]:
        """Get all registered schemas."""
        return {
            "node_schemas": {k: {
                "node_type": v.node_type,
                "required_fields": list(v.required_fields),
                "optional_fields": list(v.optional_fields),
                "field_types": {field: str(type_) for field, type_ in v.field_types.items()}
            } for k, v in self.node_schemas.items()},
            "edge_schemas": {k: {
                "edge_type": v.edge_type,
                "source_node_types": list(v.source_node_types),
                "target_node_types": list(v.target_node_types),
                "required_properties": list(v.required_properties),
                "optional_properties": list(v.optional_properties),
                "property_types": {prop: str(type_) for prop, type_ in v.property_types.items()}
            } for k, v in self.edge_schemas.items()}
        }


# Global validator instance
_global_validator = None


def get_validator() -> GraphSchemaValidator:
    """Get the global schema validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = GraphSchemaValidator()
    return _global_validator


def set_validation_level(level: ValidationLevel):
    """Set the global validation level."""
    validator = get_validator()
    validator.validation_level = level
    logger.info(f"Set validation level to: {level.value}")
