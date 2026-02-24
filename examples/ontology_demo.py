#!/usr/bin/env python3
"""
Demonstration of the Rich Ontology System

This script demonstrates SmartMemory's ontology-aware extraction capabilities:
1. Typed entity nodes (Person, Organization, Concept, etc.)
2. Semantic relationships (WORKS_AT, MANAGES, USES, etc.)
3. LLM-based extraction with ontology schema guidance

This showcases how ontology-aware systems can provide richer semantic understanding
compared to generic entity extraction approaches.
"""

import os
import sys

from smartmemory.plugins.extractors import LLMExtractor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartmemory.smart_memory import SmartMemory
from smartmemory.models.memory_item import MemoryItem


def demonstrate_old_vs_new():
    """Demonstrate the ontology-aware extraction system."""

    print("🧠 SmartMemory Ontology Demonstration")
    print("=" * 60)
    print()

    # Sample text with rich semantic content
    sample_text = """
    John Smith works at Google Inc as a Senior Software Engineer. He manages Sarah Johnson 
    and collaborates with Mike Chen on machine learning projects. John lives in San Francisco 
    and is an expert in Python programming. He uses VS Code for development and knows about 
    artificial intelligence concepts. Sarah Johnson reports to John Smith and specializes 
    in React development. Google Inc is located in Mountain View and uses advanced AI tools.
    """

    print("📝 Sample Text:")
    print("-" * 20)
    print(sample_text.strip())
    print()

    # Initialize SmartMemory with ontology-aware extraction
    memory = SmartMemory()

    print("🎯 ONTOLOGY-AWARE EXTRACTION:")
    print("-" * 35)
    print("This demonstration shows how ontology-guided extraction provides:")
    print("  • Typed entity nodes (not generic 'Entity' objects)")
    print("  • Semantic relationships (not generic 'RELATED' links)")
    print("  • Structured understanding of domain concepts")
    print()

    # Create memory item and ingest with ontology extraction
    item = MemoryItem(
        content=sample_text,
        metadata={"source": "ontology_demo", "user_id": "demo_user"}
    )

    # Ingest the item (will use ontology extractor by default)
    result_id = memory.ingest(item)
    print(f"✅ Ingested item: {result_id}")
    print()

    # Show extracted entities by type
    print("🏷️  EXTRACTED TYPED ENTITIES:")
    print("-" * 30)

    # Get all nodes from the graph to analyze what was extracted
    try:
        nodes = memory._graph.get_all_nodes()
        entities_by_type = {}

        for node in nodes:
            node_type = node.get('type', 'unknown')
            if node_type not in entities_by_type:
                entities_by_type[node_type] = []
            entities_by_type[node_type].append(node)

        for node_type, entities in entities_by_type.items():
            if node_type != 'unknown':
                print(f"📋 {node_type.upper()}:")
                for entity in entities:
                    name = entity.get('name', entity.get('content', 'Unknown'))
                    properties = entity.get('properties') or {}
                    print(f"   • {name}")
                    if properties:
                        for key, value in properties.items():
                            if key not in ['created_at', 'updated_at', 'confidence', 'source'] and value:
                                print(f"     - {key}: {value}")
                print()

    except Exception as e:
        print(f"⚠️  Could not retrieve nodes: {e}")
        print("   This might be due to backend connectivity issues")
        print()

    # Show semantic relationships
    print("🔗 SEMANTIC RELATIONSHIPS:")
    print("-" * 25)

    try:
        edges = memory._graph.get_all_edges()
        relationship_types = {}

        for edge in edges:
            rel_type = edge.get('type', edge.get('relation_type', 'unknown'))
            if rel_type not in relationship_types:
                relationship_types[rel_type] = []
            relationship_types[rel_type].append(edge)

        for rel_type, relationships in relationship_types.items():
            if rel_type != 'unknown' and rel_type != 'RELATED':
                print(f"🎯 {rel_type}:")
                for rel in relationships:
                    source = rel.get('source_id', 'Unknown')
                    target = rel.get('target_id', 'Unknown')
                    print(f"   • {source} → {target}")
                print()

        # Show relationship type distribution
        related_count = len(relationship_types.get('RELATED', []))
        semantic_count = sum(len(rels) for rel_type, rels in relationship_types.items()
                             if rel_type not in ['unknown', 'RELATED'])

        print("📊 RELATIONSHIP QUALITY:")
        print(f"   Semantic relationships: {semantic_count}")
        print(f"   Generic 'RELATED': {related_count}")
        if semantic_count + related_count > 0:
            print(f"   Semantic ratio: {semantic_count / (semantic_count + related_count) * 100:.1f}%")
        print()

    except Exception as e:
        print(f"⚠️  Could not retrieve relationships: {e}")
        print()

    # Show the LLM ontology extractor in action
    print("🔬 LLM ONTOLOGY EXTRACTOR ANALYSIS:")
    print("-" * 40)

    extractor = LLMExtractor()
    extraction_result = extractor.extract_entities_and_relations(sample_text, "demo_user")

    print(f"📈 Entities extracted: {len(extraction_result['entities'])}")
    for entity in extraction_result['entities']:
        print(f"   • {entity.name} ({entity.node_type.value})")

    print(f"\n🔗 Semantic relations found: {len(extraction_result['relations'])}")
    for relation in extraction_result['relations']:
        print(f"   • {relation['source_text']} --{relation['relation_type']}--> {relation['target_text']}")

    print()
    print("🎉 BENEFITS OF ONTOLOGY-AWARE EXTRACTION:")
    print("-" * 44)
    print("✅ LLM understands context and semantics")
    print("✅ Proper entity typing enables specialized reasoning")
    print("✅ Semantic relationships provide meaningful connections")
    print("✅ Improved query capabilities through typed entities")
    print("✅ Knowledge graph with rich semantic structure")
    print("✅ Enhanced agent reasoning with domain understanding")
    print("✅ Ontology schema guides LLM for consistent extraction")
    print()

    return memory


def demonstrate_querying_improvements(memory):
    """Show how ontology-aware extraction enables richer querying."""

    print("🔍 QUERY CAPABILITIES:")
    print("-" * 22)
    print()

    # Example queries enabled by typed entities and semantic relationships
    queries = [
        "Find all people who work at Google",
        "Who manages Sarah Johnson?",
        "What programming languages does John know?",
        "Which tools are used for development?",
        "Where is Google located?"
    ]

    print("💡 Example queries enabled by ontology-aware extraction:")
    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")

    print()
    print("Note: These queries leverage typed entities and semantic relationships")
    print("      to provide more precise and meaningful results.")
    print()


if __name__ == "__main__":
    try:
        memory = demonstrate_old_vs_new()
        demonstrate_querying_improvements(memory)

        print("🏁 Demo completed successfully!")
        print("   The ontology-aware extraction provides richer")
        print("   semantic understanding through typed entities and relationships.")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
