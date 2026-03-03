import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, UTC
from typing import Dict, Set, List, Any, Optional


@dataclass
class PropertyConstraint:
    """Constraint on an entity type property (OL-3)."""

    required: bool = False
    type: str = "string"  # string | number | date | boolean | enum
    cardinality: str = "one"  # one | many
    kind: str = "soft"  # soft (warn) | hard (reject)
    enum_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "required": self.required,
            "type": self.type,
            "cardinality": self.cardinality,
            "kind": self.kind,
        }
        if self.enum_values:
            d["enum_values"] = self.enum_values
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertyConstraint":
        valid_keys = {"required", "type", "cardinality", "kind", "enum_values"}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class EntityTypeDefinition:
    """Definition of an entity type in the ontology."""

    name: str
    description: str
    properties: Dict[str, str]  # property_name -> property_type
    required_properties: Set[str]
    parent_types: Set[str]  # inheritance hierarchy
    aliases: Set[str]  # alternative names
    examples: List[str]  # example entities of this type
    created_by: str  # "human", "llm", "inferred"
    created_at: datetime
    confidence: float = 1.0
    property_constraints: Dict[str, "PropertyConstraint"] = field(default_factory=dict)


@dataclass
class RelationshipTypeDefinition:
    """Definition of a relationship type in the ontology."""

    name: str
    description: str
    source_types: Set[str]  # allowed source entity types
    target_types: Set[str]  # allowed target entity types
    properties: Dict[str, str]  # relationship properties
    bidirectional: bool = False
    aliases: Set[str] = None
    examples: List[Dict[str, str]] = None  # example relationships
    created_by: str = "human"
    created_at: datetime = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()
        if self.examples is None:
            self.examples = []
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


@dataclass
class OntologyRule:
    """Validation or enrichment rule for the ontology."""

    id: str
    name: str
    description: str
    rule_type: str  # "validation", "enrichment", "inference"
    conditions: Dict[str, Any]  # rule conditions
    actions: Dict[str, Any]  # rule actions
    enabled: bool = True
    created_by: str = "human"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


@dataclass
class OntologySubscription:
    """Tracks subscription of an overlay ontology to a base ontology."""

    base_registry_id: str
    pinned_version: Optional[str] = None  # None = follow latest
    hidden_types: Set[str] = field(default_factory=set)
    subscribed_at: datetime = field(default_factory=datetime.now)
    subscribed_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_registry_id": self.base_registry_id,
            "pinned_version": self.pinned_version,
            "hidden_types": sorted(self.hidden_types),
            "subscribed_at": self.subscribed_at.isoformat(),
            "subscribed_by": self.subscribed_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologySubscription":
        return cls(
            base_registry_id=data["base_registry_id"],
            pinned_version=data.get("pinned_version"),
            hidden_types=set(data.get("hidden_types", [])),
            subscribed_at=datetime.fromisoformat(data["subscribed_at"]) if "subscribed_at" in data else datetime.now(UTC),
            subscribed_by=data.get("subscribed_by", ""),
        )


@dataclass
class LayerDiff:
    """Diff between base and overlay ontology layers."""

    base_only: List[str]  # types in base but not overlay
    overlay_only: List[str]  # types in overlay but not base
    overridden: List[str]  # types in both (overlay wins)
    hidden: List[str]  # base types suppressed via subscription


class Ontology:
    """Complete ontology definition with entities, relationships, and rules."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.version = version
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

        self.entity_types: Dict[str, EntityTypeDefinition] = {}
        self.relationship_types: Dict[str, RelationshipTypeDefinition] = {}
        self.rules: Dict[str, OntologyRule] = {}

        # Metadata
        self.description = ""
        self.domain = ""  # e.g., "general", "medical", "legal", "technical"
        self.language = "en"
        self.created_by = "system"
        self.tenant_id: str = ""  # tenant scope for multi-tenancy filtering
        self.is_template: bool = False
        self.source_template: str = ""

        # Layer support
        self.subscription: Optional[OntologySubscription] = None
        self.is_base_layer: bool = False

    def add_entity_type(self, entity_type: EntityTypeDefinition) -> None:
        """Add an entity type to the ontology."""
        self.entity_types[entity_type.name.lower()] = entity_type
        self.updated_at = datetime.now(UTC)

    def add_relationship_type(self, rel_type: RelationshipTypeDefinition) -> None:
        """Add a relationship type to the ontology."""
        self.relationship_types[rel_type.name.lower()] = rel_type
        self.updated_at = datetime.now(UTC)

    def add_rule(self, rule: OntologyRule) -> None:
        """Add a rule to the ontology."""
        self.rules[rule.id] = rule
        self.updated_at = datetime.now(UTC)

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """Get entity type by name (case-insensitive)."""
        return self.entity_types.get(name.lower())

    def get_relationship_type(self, name: str) -> Optional[RelationshipTypeDefinition]:
        """Get relationship type by name (case-insensitive)."""
        return self.relationship_types.get(name.lower())

    def validate_entity(self, entity_type: str, properties: Dict[str, Any]) -> List[str]:
        """Validate an entity against the ontology. Returns list of validation errors."""
        errors = []

        entity_def = self.get_entity_type(entity_type)
        if not entity_def:
            return [f"Unknown entity type: {entity_type}"]

        # Check required properties
        for req_prop in entity_def.required_properties:
            if req_prop not in properties:
                errors.append(f"Missing required property '{req_prop}' for entity type '{entity_type}'")

        # Check property types (basic validation)
        for prop_name, prop_value in properties.items():
            if prop_name in entity_def.properties:
                expected_type = entity_def.properties[prop_name]
                # Add type validation logic here if needed

        return errors

    def validate_relationship(self, rel_type: str, source_type: str, target_type: str) -> List[str]:
        """Validate a relationship against the ontology. Returns list of validation errors."""
        errors = []

        rel_def = self.get_relationship_type(rel_type)
        if not rel_def:
            return [f"Unknown relationship type: {rel_type}"]

        # Check source type constraints
        if rel_def.source_types and source_type.lower() not in {t.lower() for t in rel_def.source_types}:
            errors.append(
                f"Invalid source type '{source_type}' for relationship '{rel_type}'. Allowed: {rel_def.source_types}"
            )

        # Check target type constraints
        if rel_def.target_types and target_type.lower() not in {t.lower() for t in rel_def.target_types}:
            errors.append(
                f"Invalid target type '{target_type}' for relationship '{rel_type}'. Allowed: {rel_def.target_types}"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ontology to dictionary."""
        # Custom serialization to handle sets properly
        entity_types_dict = {}
        for name, entity_type in self.entity_types.items():
            entity_dict = asdict(entity_type)
            entity_dict["required_properties"] = list(entity_type.required_properties)
            entity_dict["parent_types"] = list(entity_type.parent_types)
            entity_dict["aliases"] = list(entity_type.aliases)
            entity_dict["created_at"] = entity_type.created_at.isoformat()
            if entity_type.property_constraints:
                entity_dict["property_constraints"] = {
                    k: v.to_dict() for k, v in entity_type.property_constraints.items()
                }
            entity_types_dict[name] = entity_dict

        relationship_types_dict = {}
        for name, rel_type in self.relationship_types.items():
            rel_dict = asdict(rel_type)
            rel_dict["source_types"] = list(rel_type.source_types)
            rel_dict["target_types"] = list(rel_type.target_types)
            rel_dict["aliases"] = list(rel_type.aliases)
            rel_dict["created_at"] = rel_type.created_at.isoformat()
            relationship_types_dict[name] = rel_dict

        rules_dict = {}
        for rule_id, rule in self.rules.items():
            rule_dict = asdict(rule)
            rule_dict["created_at"] = rule.created_at.isoformat()
            rules_dict[rule_id] = rule_dict

        result = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description,
            "domain": self.domain,
            "language": self.language,
            "created_by": self.created_by,
            "tenant_id": self.tenant_id,
            "is_template": self.is_template,
            "source_template": self.source_template,
            "entity_types": entity_types_dict,
            "relationship_types": relationship_types_dict,
            "rules": rules_dict,
            "is_base_layer": self.is_base_layer,
        }
        if self.subscription is not None:
            result["subscription"] = self.subscription.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ontology":
        """Deserialize ontology from dictionary."""
        ontology = cls(data["name"], data["version"])
        ontology.id = data["id"]
        ontology.created_at = datetime.fromisoformat(data["created_at"])
        ontology.updated_at = datetime.fromisoformat(data["updated_at"])
        ontology.description = data.get("description", "")
        ontology.domain = data.get("domain", "")
        ontology.language = data.get("language", "en")
        ontology.created_by = data.get("created_by", "system")
        ontology.tenant_id = data.get("tenant_id", "")
        ontology.is_template = data.get("is_template", False)
        ontology.source_template = data.get("source_template", "")
        ontology.is_base_layer = data.get("is_base_layer", False)
        if "subscription" in data and data["subscription"] is not None:
            ontology.subscription = OntologySubscription.from_dict(data["subscription"])

        # Load entity types
        for name, entity_data in (data.get("entity_types") or {}).items():
            entity_data["created_at"] = datetime.fromisoformat(entity_data["created_at"])
            entity_data["required_properties"] = set(entity_data["required_properties"])
            entity_data["parent_types"] = set(entity_data["parent_types"])
            entity_data["aliases"] = set(entity_data["aliases"])
            raw_constraints = entity_data.pop("property_constraints", None) or {}
            entity_data["property_constraints"] = {
                k: PropertyConstraint.from_dict(v) for k, v in raw_constraints.items()
            }
            entity_type = EntityTypeDefinition(**entity_data)
            ontology.entity_types[name] = entity_type

        # Load relationship types
        for name, rel_data in (data.get("relationship_types") or {}).items():
            rel_data["created_at"] = datetime.fromisoformat(rel_data["created_at"])
            rel_data["source_types"] = set(rel_data.get("source_types", []))
            rel_data["target_types"] = set(rel_data.get("target_types", []))
            rel_data["aliases"] = set(rel_data.get("aliases", []))
            rel_type = RelationshipTypeDefinition(**rel_data)
            ontology.relationship_types[name] = rel_type

        # Load rules
        for rule_id, rule_data in (data.get("rules") or {}).items():
            rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
            rule = OntologyRule(**rule_data)
            ontology.rules[rule_id] = rule

        return ontology


@dataclass
class TemplateInfo:
    """Metadata about an ontology template for catalog browsing."""

    id: str
    name: str
    domain: str
    description: str
    version: str
    entity_count: int
    relationship_count: int
    is_builtin: bool
    created_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemplatePreview(TemplateInfo):
    """Extended template info with entity/relationship names for preview."""

    entity_types: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    rules_count: int = 0
