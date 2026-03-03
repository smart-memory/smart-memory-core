#!/usr/bin/env python3
"""
Self-Learning Ontology Example

Demonstrates SmartMemory's self-learning ontology system:
1. Creating and managing ontology schemas
2. Defining entity types with properties and constraints
3. LLM-discovered entity type promotion
4. Layered ontologies (base + overlay)
5. Validation and governance

The ontology system allows SmartMemory to learn new entity types from
data patterns while maintaining schema governance and validation.

Key Features:
- Entity type definitions with properties, aliases, and examples
- Relationship type definitions with domain/range constraints
- Promotion workflow for LLM-discovered types
- Layered architecture for base schemas + workspace customizations
- Validation rules and governance controls

Run this example:
    PYTHONPATH=. python examples/self_learning_ontology_example.py
"""

import logging
from datetime import datetime, UTC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from smartmemory.ontology.models import (
        Ontology,
        EntityTypeDefinition,
        RelationshipTypeDefinition,
        OntologyRule,
        OntologySubscription,
        LayerDiff,
    )
    from smartmemory.ontology.layered import LayeredOntology
    from smartmemory.ontology.promotion import (
        PromotionCandidate,
        PromotionResult,
        PromotionEvaluator,
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("This example requires the full SmartMemory system to be available.")
    exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_create_ontology():
    """Demonstrate creating an ontology with entity and relationship types."""
    print_section("1. Creating an Ontology Schema")
    
    # Create a new ontology
    ontology = Ontology(name="TechDomain", version="1.0.0")
    ontology.description = "Technical domain ontology for software development"
    ontology.domain = "technology"
    
    print(f"📝 Created ontology: {ontology.name} v{ontology.version}")
    print(f"   ID: {ontology.id}")
    print(f"   Domain: {ontology.domain}")
    
    # Add entity types
    person_type = EntityTypeDefinition(
        name="Person",
        description="A human individual",
        properties={
            "name": "string",
            "email": "string",
            "role": "string",
            "skills": "list[string]",
        },
        required_properties={"name"},
        parent_types=set(),
        aliases={"Individual", "User"},
        examples=["John Smith", "Jane Doe"],
        created_by="human",
        created_at=datetime.now(UTC),
        confidence=1.0,
    )
    ontology.add_entity_type(person_type)
    
    org_type = EntityTypeDefinition(
        name="Organization",
        description="A company or organization",
        properties={
            "name": "string",
            "industry": "string",
            "size": "integer",
            "location": "string",
        },
        required_properties={"name"},
        parent_types=set(),
        aliases={"Company", "Business"},
        examples=["Google", "OpenAI"],
        created_by="human",
        created_at=datetime.now(UTC),
        confidence=1.0,
    )
    ontology.add_entity_type(org_type)
    
    tech_type = EntityTypeDefinition(
        name="Technology",
        description="A programming language, framework, or tool",
        properties={
            "name": "string",
            "category": "string",  # language, framework, library, tool
            "version": "string",
        },
        required_properties={"name"},
        parent_types=set(),
        aliases={"Tech", "Tool", "Framework"},
        examples=["Python", "React", "Docker"],
        created_by="human",
        created_at=datetime.now(UTC),
        confidence=1.0,
    )
    ontology.add_entity_type(tech_type)
    
    print(f"\n✅ Added {len(ontology.entity_types)} entity types:")
    for name, entity_type in ontology.entity_types.items():
        print(f"   • {entity_type.name}: {entity_type.description}")
        print(f"     Properties: {list(entity_type.properties.keys())}")
        print(f"     Aliases: {entity_type.aliases}")
    
    # Add relationship types
    works_at = RelationshipTypeDefinition(
        name="works_at",
        description="Person works at an organization",
        source_types={"Person"},
        target_types={"Organization"},
        properties={"role": "string", "start_date": "date"},
        bidirectional=False,
        aliases={"employed_at", "employed_by"},
    )
    ontology.add_relationship_type(works_at)
    
    uses = RelationshipTypeDefinition(
        name="uses",
        description="Person or organization uses a technology",
        source_types={"Person", "Organization"},
        target_types={"Technology"},
        properties={"proficiency": "string"},
        bidirectional=False,
        aliases={"utilizes", "uses_technology"},
    )
    ontology.add_relationship_type(uses)
    
    print(f"\n✅ Added {len(ontology.relationship_types)} relationship types:")
    for name, rel_type in ontology.relationship_types.items():
        print(f"   • {rel_type.name}: {rel_type.description}")
        print(f"     {rel_type.source_types} → {rel_type.target_types}")
    
    return ontology


def demo_validation():
    """Demonstrate ontology validation."""
    print_section("2. Ontology Validation")
    
    ontology = Ontology(name="ValidationDemo", version="1.0.0")
    
    # Add a simple entity type with required properties
    person_type = EntityTypeDefinition(
        name="Person",
        description="A human individual",
        properties={"name": "string", "email": "string"},
        required_properties={"name"},
        parent_types=set(),
        aliases=set(),
        examples=[],
        created_by="human",
        created_at=datetime.now(UTC),
    )
    ontology.add_entity_type(person_type)
    
    # Valid entity
    valid_entity = {"name": "John Smith", "email": "john@example.com"}
    errors = ontology.validate_entity("Person", valid_entity)
    print(f"✅ Valid entity: {valid_entity}")
    print(f"   Errors: {errors if errors else 'None'}")
    
    # Invalid entity (missing required 'name')
    invalid_entity = {"email": "john@example.com"}
    errors = ontology.validate_entity("Person", invalid_entity)
    print(f"\n❌ Invalid entity: {invalid_entity}")
    print(f"   Errors: {errors}")
    
    # Unknown entity type
    errors = ontology.validate_entity("Robot", {"name": "R2D2"})
    print("\n❓ Unknown type: 'Robot'")
    print(f"   Errors: {errors}")


def demo_layered_ontology():
    """Demonstrate layered ontologies (base + overlay)."""
    print_section("3. Layered Ontologies (Base + Overlay)")
    
    # Create a base ontology (shared across workspaces)
    base = Ontology(name="BaseOntology", version="1.0.0")
    base.is_base_layer = True
    
    base.add_entity_type(EntityTypeDefinition(
        name="Person",
        description="A human individual (base)",
        properties={"name": "string"},
        required_properties={"name"},
        parent_types=set(),
        aliases=set(),
        examples=[],
        created_by="human",
        created_at=datetime.now(UTC),
    ))
    base.add_entity_type(EntityTypeDefinition(
        name="Organization",
        description="An organization (base)",
        properties={"name": "string"},
        required_properties={"name"},
        parent_types=set(),
        aliases=set(),
        examples=[],
        created_by="human",
        created_at=datetime.now(UTC),
    ))
    base.add_entity_type(EntityTypeDefinition(
        name="Location",
        description="A location (base)",
        properties={"name": "string"},
        required_properties={"name"},
        parent_types=set(),
        aliases=set(),
        examples=[],
        created_by="human",
        created_at=datetime.now(UTC),
    ))
    
    print(f"📋 Base ontology: {len(base.entity_types)} entity types")
    print(f"   Types: {list(base.entity_types.keys())}")
    
    # Create an overlay ontology (workspace-specific customizations)
    overlay = Ontology(name="WorkspaceOverlay", version="1.0.0")
    overlay.subscription = OntologySubscription(
        base_registry_id=base.id,
        pinned_version=None,  # Follow latest base version
        hidden_types={"location"},  # Hide Location from this workspace
    )
    
    # Override Person with custom properties
    overlay.add_entity_type(EntityTypeDefinition(
        name="Person",
        description="A human individual (customized for tech domain)",
        properties={"name": "string", "github_username": "string", "skills": "list"},
        required_properties={"name"},
        parent_types=set(),
        aliases={"Developer", "Engineer"},
        examples=["Linus Torvalds", "Guido van Rossum"],
        created_by="human",
        created_at=datetime.now(UTC),
    ))
    
    # Add new type not in base
    overlay.add_entity_type(EntityTypeDefinition(
        name="Repository",
        description="A code repository",
        properties={"name": "string", "url": "string", "stars": "integer"},
        required_properties={"name"},
        parent_types=set(),
        aliases={"Repo", "Project"},
        examples=["linux", "tensorflow"],
        created_by="human",
        created_at=datetime.now(UTC),
    ))
    
    print(f"\n📋 Overlay ontology: {len(overlay.entity_types)} entity types")
    print(f"   Types: {list(overlay.entity_types.keys())}")
    
    # Create layered view
    layered = LayeredOntology(
        overlay=overlay,
        base=base,
        hidden_types=overlay.subscription.hidden_types,
    )
    
    print("\n🔗 Layered view (merged):")
    print(f"   Total entity types: {len(layered.entity_types)}")
    print(f"   Types: {list(layered.entity_types.keys())}")
    
    # Show provenance
    print("\n📊 Type provenance:")
    for type_name, provenance in layered.get_provenance_map().items():
        icon = {"local": "🆕", "base": "📦", "override": "🔄", "hidden": "👁️‍🗨️"}.get(provenance, "❓")
        print(f"   {icon} {type_name}: {provenance}")
    
    # Compute diff
    diff = layered.compute_diff()
    print("\n📋 Layer Diff:")
    print(f"   Base only: {diff.base_only}")
    print(f"   Overlay only: {diff.overlay_only}")
    print(f"   Overridden: {diff.overridden}")
    print(f"   Hidden: {diff.hidden}")
    
    return layered


def demo_promotion_candidates():
    """Demonstrate the entity type promotion workflow."""
    print_section("4. Self-Learning: Entity Type Promotion")
    
    print("""
🧠 Self-Learning Workflow:

1. LLM extracts entities with novel types during ingestion
   e.g., "React" → type: "Framework" (not in ontology)

2. Promotion candidates are collected with statistics:
   - Frequency: How often this type appears
   - Confidence: LLM confidence in the type assignment
   - Consistency: How consistently this name gets this type

3. PromotionEvaluator applies gates:
   - min_name_length (filter short/invalid names)
   - Common word blocklist (filter "the", "a", etc.)
   - min_confidence threshold
   - min_frequency threshold
   - min_type_consistency ratio
   - Optional: LLM reasoning validation

4. If all gates pass, type is promoted to the ontology
    """)
    
    # Demonstrate promotion candidate structure
    candidate = PromotionCandidate(
        entity_name="React",
        entity_type="Framework",
        confidence=0.92,
        source_memory_id="mem_abc123",
    )
    
    print("📝 Promotion Candidate:")
    print(f"   Entity: {candidate.entity_name}")
    print(f"   Type: {candidate.entity_type}")
    print(f"   Confidence: {candidate.confidence}")
    print(f"   Source: {candidate.source_memory_id}")
    
    # Show gate outcomes (simulated)
    print("\n🚦 Promotion Gates (simulated):")
    print(f"   ✅ Gate 1: Name length ({len(candidate.entity_name)} >= 3)")
    print("   ✅ Gate 2: Not a common word")
    print(f"   ✅ Gate 3: Confidence ({candidate.confidence:.2f} >= 0.70)")
    print("   ✅ Gate 4: Frequency (5 >= 2)")
    print("   ✅ Gate 5: Type consistency (0.85 >= 0.80)")
    print("   🔄 Gate 6: Reasoning validation (optional)")
    
    # Simulated result
    result = PromotionResult(
        promoted=True,
        reason="All gates passed",
    )
    print(f"\n{'✅ PROMOTED' if result.promoted else '❌ REJECTED'}: {result.reason}")


def demo_ontology_rules():
    """Demonstrate ontology rules."""
    print_section("5. Ontology Rules")
    
    ontology = Ontology(name="RulesDemo", version="1.0.0")
    
    # Add validation rule
    validation_rule = OntologyRule(
        id="rule_email_format",
        name="Email Format Validation",
        description="Ensure email properties contain valid email format",
        rule_type="validation",
        conditions={"property": "email", "pattern": r"^[\w.-]+@[\w.-]+\.\w+$"},
        actions={"reject_if_invalid": True},
    )
    ontology.add_rule(validation_rule)
    
    # Add enrichment rule
    enrichment_rule = OntologyRule(
        id="rule_auto_tag",
        name="Auto-Tag Technology",
        description="Automatically tag technology entities by category",
        rule_type="enrichment",
        conditions={"entity_type": "Technology"},
        actions={"add_tags": ["tech", "software"]},
    )
    ontology.add_rule(enrichment_rule)
    
    # Add inference rule
    inference_rule = OntologyRule(
        id="rule_infer_team",
        name="Infer Team Membership",
        description="If Person works_at Org and uses Tech, they may be on a team",
        rule_type="inference",
        conditions={
            "pattern": "(Person)-[:works_at]->(Org)<-[:uses]-(Person)-[:uses]->(Tech)"
        },
        actions={"create_edge": {"type": "teammate", "confidence": 0.6}},
    )
    ontology.add_rule(inference_rule)
    
    print("📜 Ontology Rules:")
    for rule_id, rule in ontology.rules.items():
        print(f"\n   [{rule.rule_type.upper()}] {rule.name}")
        print(f"   Description: {rule.description}")
        print(f"   Conditions: {rule.conditions}")
        print(f"   Actions: {rule.actions}")


def demo_serialization():
    """Demonstrate ontology serialization."""
    print_section("6. Ontology Serialization")
    
    # Create ontology
    original = Ontology(name="SerializeDemo", version="1.0.0")
    original.description = "Demo ontology for serialization"
    original.domain = "demo"
    
    original.add_entity_type(EntityTypeDefinition(
        name="Example",
        description="An example entity",
        properties={"name": "string"},
        required_properties={"name"},
        parent_types=set(),
        aliases={"Sample"},
        examples=["test"],
        created_by="human",
        created_at=datetime.now(UTC),
    ))
    
    # Serialize to dict
    data = original.to_dict()
    print("📦 Serialized ontology:")
    print(f"   Keys: {list(data.keys())}")
    print(f"   Entity types: {list(data['entity_types'].keys())}")
    
    # Deserialize from dict
    restored = Ontology.from_dict(data)
    print("\n📥 Restored ontology:")
    print(f"   Name: {restored.name}")
    print(f"   Version: {restored.version}")
    print(f"   Entity types: {list(restored.entity_types.keys())}")
    print(f"   Match: {'✅' if restored.name == original.name else '❌'}")


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         SmartMemory Self-Learning Ontology System - Demo            ║
║                                                                      ║
║  Learn new entity types from data while maintaining governance      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_create_ontology()
        demo_validation()
        demo_layered_ontology()
        demo_promotion_candidates()
        demo_ontology_rules()
        demo_serialization()
        
        print_section("Demo Complete!")
        print("""
🎉 You've seen the core features of SmartMemory's Self-Learning Ontology!

Key concepts:
  • EntityTypeDefinition: Schema for entity types with properties
  • RelationshipTypeDefinition: Schema for relationships with constraints
  • LayeredOntology: Base + overlay architecture for customization
  • PromotionEvaluator: Gate-based approval for LLM-discovered types
  • OntologyRule: Validation, enrichment, and inference rules

Self-learning workflow:
  1. LLM discovers novel entity types during extraction
  2. Candidates are evaluated against promotion gates
  3. Approved types are added to the workspace ontology
  4. Schema evolves while maintaining governance

Use cases:
  • Domain-specific knowledge graphs (legal, medical, finance)
  • Multi-tenant ontologies with shared base + custom overlays
  • Progressive schema discovery from unstructured data
        """)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
