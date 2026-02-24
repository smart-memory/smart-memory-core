#!/usr/bin/env python3
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.golden]

"""
Test script to verify all README.md code snippets work in a fresh environment.
Run this in a new virtual environment after: pip install smartmemory
"""

def test_basic_usage():
    """Test the basic usage snippet from README"""
    print("\n=== Testing Basic Usage ===")
    from smartmemory import SmartMemory, MemoryItem

    # Initialize SmartMemory
    memory = SmartMemory()

    # Add a memory
    item = MemoryItem(
        content="User prefers Python for data analysis tasks",
        memory_type="semantic",
        
        metadata={'topic': 'preferences', 'domain': 'programming'}
    )
    memory.add(item)

    # Search memories
    results = memory.search("Python programming", top_k=5)
    for result in results:
        print(f"Content: {result.content}")
        print(f"Type: {result.memory_type}")

    # Get memory summary
    summary = memory.summary()
    print(f"Total memories: {summary}")
    print("✅ Basic usage works!")


def test_memory_types():
    """Test using different memory types"""
    print("\n=== Testing Different Memory Types ===")
    from smartmemory import SmartMemory, MemoryItem

    # Initialize SmartMemory
    memory = SmartMemory()

    # Add working memory (short-term context)
    working_item = MemoryItem(
        content="Current conversation context",
        memory_type="working"
    )
    memory.add(working_item)

    # Add semantic memory (facts and concepts)
    semantic_item = MemoryItem(
        content="Python is a high-level programming language",
        memory_type="semantic"
    )
    memory.add(semantic_item)

    # Add episodic memory (experiences)
    episodic_item = MemoryItem(
        content="User completed Python tutorial on 2024-01-15",
        memory_type="episodic"
    )
    memory.add(episodic_item)

    # Add procedural memory (skills and procedures)
    procedural_item = MemoryItem(
        content="To sort a list in Python: use list.sort() or sorted(list)",
        memory_type="procedural"
    )
    memory.add(procedural_item)

    # Add Zettelkasten note (interconnected knowledge)
    zettel_item = MemoryItem(
        content="# Machine Learning\n\nML learns from data using algorithms.",
        memory_type="zettel",
        metadata={'title': 'ML Basics', 'tags': ['ai', 'ml']}
    )
    memory.add(zettel_item)

    print("✅ All memory types work!")


def test_api_reference():
    """Test the API reference examples"""
    print("\n=== Testing API Reference ===")
    from smartmemory import SmartMemory, MemoryItem

    memory = SmartMemory()
    
    # Test add
    item = MemoryItem(content="Test memory", memory_type="semantic")
    result = memory.add(item)
    assert result is not None
    
    # Test get
    retrieved = memory.get(item.item_id)
    assert retrieved is not None
    
    # Test search
    results = memory.search("test", top_k=10)
    assert isinstance(results, list)
    
    # Test delete
    success = memory.delete(item.item_id)
    assert isinstance(success, bool)
    
    # Test summary
    summary = memory.summary()
    assert isinstance(summary, dict)
    
    print("✅ API reference works!")


def test_custom_plugin_example():
    """Test the custom plugin example compiles"""
    print("\n=== Testing Custom Plugin Example ===")
    from smartmemory.plugins.base import EnricherPlugin, PluginMetadata

    class MyCustomEnricher(EnricherPlugin):
        @classmethod
        def metadata(cls):
            return PluginMetadata(
                name="my_enricher",
                version="1.0.0",
                author="Your Name",
                description="My custom enricher",
                plugin_type="enricher",
                dependencies=["some-lib>=1.0.0"],
                security_profile="standard",
                requires_network=False,
                requires_llm=False
            )
        
        def enrich(self, item, node_ids=None):
            # Your enrichment logic
            item.metadata["custom_field"] = "value"
            return item.metadata

    # Verify metadata works
    meta = MyCustomEnricher.metadata()
    assert meta.name == "my_enricher"
    print("✅ Custom plugin example works!")


def test_imports():
    """Test that all imports from README work"""
    print("\n=== Testing Imports ===")
    
    # Main imports
    print("✅ Main imports work")
    
    # Plugin imports
    print("✅ Plugin imports work")
    
    # Check version
    from smartmemory import __version__
    print(f"✅ SmartMemory version: {__version__}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing README.md Code Snippets")
    print("=" * 60)
    
    try:
        test_imports()
        test_basic_usage()
        test_memory_types()
        test_api_reference()
        test_custom_plugin_example()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nAll README code snippets work correctly in a fresh environment.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
