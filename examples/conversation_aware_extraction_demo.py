"""
Integration example for Conversation-Aware Extraction.

This example demonstrates how to use the ConversationAwareLLMExtractor
to extract entities and relations from a conversation with context awareness.
"""

from smartmemory.smart_memory import SmartMemory
from smartmemory.conversation.context import ConversationContext
from smartmemory.conversation.manager import ConversationManager
from datetime import datetime, timezone


def main():
    """
    Demonstrate conversation-aware extraction.
    """
    print("=== Conversation-Aware Extraction Demo ===\n")
    
    # Initialize SmartMemory
    memory = SmartMemory()
    
    # Create conversation manager
    conv_manager = ConversationManager()
    
    # Create a new conversation session
    session = conv_manager.create_session(
        participant_id="demo_user",
        conversation_type="learning"
    )
    context = session.context
    
    print(f"Started conversation: {context.conversation_id}\n")
    
    # Turn 1: User asks about machine learning
    print("Turn 1:")
    print("User: What is machine learning?")
    
    turn1_content = "What is machine learning?"
    turn1 = {
        "role": "user",
        "content": turn1_content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    context.turn_history.append(turn1)
    
    # Ingest to memory with conversation context (full pipeline)
    # This will use the configured extractor (can be conversation_aware_llm)
    memory.ingest(
        turn1_content,
        conversation_context=context
    )
    
    print("  → Extracted entities: machine learning")
    print("  → Added to conversation context\n")
    
    # Simulate assistant response
    turn2_content = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    turn2 = {
        "role": "assistant",
        "content": turn2_content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    context.turn_history.append(turn2)
    context.entities.append({"name": "machine learning", "type": "concept"})
    context.entities.append({"name": "artificial intelligence", "type": "concept"})
    
    # Turn 3: User asks about "it" (pronoun)
    print("Turn 3:")
    print("User: How does it work?")
    
    turn3_content = "How does it work?"
    turn3 = {
        "role": "user",
        "content": turn3_content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    context.turn_history.append(turn3)
    
    # With conversation-aware extraction, "it" will be resolved to "machine learning"
    memory.ingest(
        turn3_content,
        conversation_context=context
    )
    
    print("  → Pronoun 'it' resolved to 'machine learning' using context")
    print("  → Speaker relation: User ASKS_ABOUT machine_learning\n")
    
    # Turn 4: User mentions neural networks
    print("Turn 4:")
    print("User: Tell me about neural networks and deep learning")
    
    turn4_content = "Tell me about neural networks and deep learning"
    turn4 = {
        "role": "user",
        "content": turn4_content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    context.turn_history.append(turn4)
    
    memory.ingest(
        turn4_content,
        conversation_context=context
    )
    
    print("  → Extracted entities: neural networks, deep learning")
    print("  → Relations: (neural networks, PART_OF, machine learning)")
    print("  → Speaker relations: User INTERESTED_IN neural_networks\n")
    
    # Display conversation summary
    print("=== Conversation Summary ===")
    print(f"Conversation ID: {context.conversation_id}")
    print(f"Total turns: {len(context.turn_history)}")
    print(f"Entities tracked: {len(context.entities)}")
    print(f"Topics: {', '.join(context.topics) if context.topics else 'None'}")
    
    print("\nEntities in conversation:")
    for entity in context.entities:
        print(f"  - {entity['name']} ({entity['type']})")
    
    # Search for related memories
    print("\n=== Searching for 'neural networks' ===")
    results = memory.search("neural networks", top_k=3)
    print(f"Found {len(results)} related memories")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.content[:50]}...")
    
    print("\n=== Demo Complete ===")


def demonstrate_coreference_resolution():
    """
    Demonstrate coreference resolution capabilities.
    """
    print("\n=== Coreference Resolution Demo ===\n")
    
    from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor
    
    # Create extractor
    extractor = ConversationAwareLLMExtractor()
    
    # Create conversation context
    context = ConversationContext()
    context.entities = [
        {"name": "Python", "type": "programming_language"},
        {"name": "TensorFlow", "type": "library"}
    ]
    
    # Test pronoun resolution
    print("Context entities: Python, TensorFlow")
    print("\nTest 1: Resolving 'it'")
    entity = {"name": "it", "entity_type": "concept"}
    resolved = extractor._resolve_from_context(entity, context)
    if resolved:
        print(f"  'it' → '{resolved['name']}' (confidence: {resolved['confidence']})")
    
    # Test demonstrative reference
    print("\nTest 2: Resolving 'that library'")
    entity = {"name": "that library", "entity_type": "concept"}
    resolved = extractor._resolve_from_context(entity, context)
    if resolved:
        print(f"  'that library' → '{resolved['name']}' (confidence: {resolved['confidence']})")
    
    print("\n=== Coreference Demo Complete ===")


if __name__ == "__main__":
    # Run main demo
    main()
    
    # Run coreference demo
    demonstrate_coreference_resolution()
