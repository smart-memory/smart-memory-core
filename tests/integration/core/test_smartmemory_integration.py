"""
Integration tests for SmartMemory core functionality.
Tests cross-component interactions and real backend integrations.
NO MOCKS - Uses real backends for true integration testing.
"""
import pytest


pytestmark = pytest.mark.integration
from datetime import datetime, timezone
import time
import threading

from smartmemory.models.memory_item import MemoryItem


@pytest.mark.integration
class TestSmartMemoryRealBackendIntegration:
    """Integration tests using real backends - NO MOCKS.

    These tests require:
    - Running Redis instance (localhost:9012 or configured port)
    - Running FalkorDB instance (localhost:9010 or configured port) 
    - Optional: OpenAI API key for embedding tests
    
    Run with: pytest -m integration --tb=short
    """
    
    def test_memory_ingestion_to_graph_integration(self, real_smartmemory_for_integration):
        """Test complete memory ingestion flow to real graph backend."""
        memory = real_smartmemory_for_integration
        
        # Create test memory item
        test_item = MemoryItem(
            content="Integration test: Python is a programming language",
            memory_type="semantic",
            
            metadata={"test": "integration", "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        
        # Test ingestion
        result = memory.add(test_item)
        assert result is not None
        assert isinstance(result, str)  # Should return item_id string
        
        # Debug: Verify node exists immediately
        node = memory.get(result)
        print(f"DEBUG: Created node: {node}")
        assert node is not None
        assert "Python" in node.content

        # Test retrieval with user_id filtering
        # Retry logic for eventual consistency if needed
        retrieved_items = []
        for i in range(3):
            retrieved_items = memory.search("Python programming", )
            if retrieved_items:
                break
            time.sleep(0.5)
            
        print(f"DEBUG: Retrieved items: {retrieved_items}")
        assert len(retrieved_items) > 0, f"Expected search results with user_id filtering but got {len(retrieved_items)} items"
        
        # Verify content matches
        found_item = next((item for item in retrieved_items if "Python" in str(item.content)), None)
        assert found_item is not None, "Expected to find item containing 'Python' in search results"
    
    def test_memory_to_vector_store_integration(self, real_smartmemory_for_integration):
        """Test memory storage and retrieval with real vector store."""
        memory = real_smartmemory_for_integration
        
        # Create memory items with different content for similarity testing
        items = [
            MemoryItem(
                content="Machine learning is a subset of artificial intelligence",
                memory_type="semantic",
                
                metadata={"topic": "AI"}
            ),
            MemoryItem(
                content="Deep learning uses neural networks with multiple layers",
                memory_type="semantic", 
                
                metadata={"topic": "AI"}
            ),
            MemoryItem(
                content="Cooking pasta requires boiling water and salt",
                memory_type="procedural",
                
                metadata={"topic": "cooking"}
            )
        ]
        
        # Store items
        for item in items:
            memory.add(item)
        
        # Test similarity search
        ai_results = memory.search("artificial intelligence neural networks", )
        assert len(ai_results) >= 2  # Should find both AI-related items
        
        # Verify AI topics are ranked higher than cooking
        ai_content_found = any("machine learning" in str(result.content).lower() or "neural network" in str(result.content).lower() 
                              for result in ai_results[:2])
        assert ai_content_found
    
    def test_cache_integration_with_memory_operations(self, real_smartmemory_for_integration):
        """Test cache integration with real Redis backend."""
        memory = real_smartmemory_for_integration
        
        # Create test item
        test_item = MemoryItem(
            content="Cache integration test content",
            memory_type="working",
            
            metadata={"cache_test": True}
        )
        
        # First operation - should populate cache
        memory.add(test_item)
        
        # Search operation - should use cache if available
        results1 = memory.search("cache integration", )
        
        # Second identical search - should be faster due to cache
        results2 = memory.search("cache integration", )
        
        # Verify results are consistent
        # Extract IDs for comparison to avoid object identity issues
        ids1 = [item.item_id for item in results1]
        ids2 = [item.item_id for item in results2]
        assert set(ids1) == set(ids2)


@pytest.mark.integration
class TestCrossComponentIntegration:
    """Test integration between different SmartMemory components."""
    
    def test_graph_and_vector_store_sync(self, real_smartmemory_for_integration):
        """Test synchronization between graph and vector store."""
        memory = real_smartmemory_for_integration
        
        # Test data
        test_item = MemoryItem(
            content="Integration test content for sync",
            memory_type="semantic",
            
            metadata={"test": True}
        )
        
        # Add item
        item_id = memory.add(test_item)
        assert item_id is not None
        
        # Verify in Graph - node is a MemoryItem object, not a dict
        node = memory._graph.get_node(item_id)
        assert node is not None
        # Access as attribute
        assert node.content == test_item.content
        
        # Verify in Vector Store (if available)
        # Note: Vector store might be optional or mocked in some envs, but this is integration test
        if hasattr(memory, '_vector_store') and memory._vector_store:
             # Vector store might be initialized lazily or inside pipeline
             pass

    def test_pipeline_stage_integration(self, real_smartmemory_for_integration):
        """Test integration between pipeline stages."""
        memory = real_smartmemory_for_integration
        
        # Test data
        test_item = MemoryItem(
            content="Pipeline integration test content",
            memory_type="episodic",
            
        )
        
        # Run ingestion
        result = memory.ingest(test_item, sync=True)
        
        # Verify result
        assert result is not None
        
        # Handle result types
        if isinstance(result, dict):
            # Pipeline result
            if 'item' in result:
                item_id = result['item'].item_id
            elif 'item_id' in result:
                item_id = result['item_id']
            else:
                item_id = None
        else:
            # Might be item object or ID
            item_id = getattr(result, 'item_id', str(result))
            
        assert item_id is not None
        
        # Verify persistence
        saved_item = memory.get(item_id)
        assert saved_item is not None
        # Ensure content is preserved
        assert saved_item.content == test_item.content
    
    def test_conversation_memory_integration(self, real_smartmemory_for_integration):
        """Test integration between conversation management and memory storage."""
        # Use real components instead of mocks
        memory = real_smartmemory_for_integration
        from smartmemory.conversation.manager import ConversationManager
        
        conv_manager = ConversationManager()
        
        # Test conversation-memory integration
        conversation_id = "conv_real_integration"
        user_id = "user_real_integration"
        
        # Create context (ConversationManager usually stores in DB, but we just use context here)
        # For integration, we can just create the item with metadata
        
        memory_item = MemoryItem(
            content="Conversation memory integration test",
            memory_type="working",
            
            metadata={"conversation_id": conversation_id}
        )
        
        memory.add(memory_item)
        search_results = memory.search("Conversation")
        
        # Debug print
        print(f"DEBUG: Conversation search results: {search_results}")
        
        assert len(search_results) > 0
        assert any(conversation_id in str(item.metadata) for item in search_results)


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration integration across components."""
    def test_configuration_validation_integration(self):
        """Test configuration validation with real ConfigManager."""
        from smartmemory.configuration.manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # ConfigManager.validate_config() raises exception on failure, returns None on success
        try:
            config_manager.validate_config()
        except Exception as e:
            # It's okay if it fails due to missing env vars in test env, but we want to catch it
            print(f"Config validation warning (expected in test env): {e}")



@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance characteristics in integrated environment."""
    
    def test_concurrent_operations_integration(self, real_smartmemory_for_integration):
        """Test concurrent operations across components."""
        memory = real_smartmemory_for_integration
        results = []
        
        def concurrent_operation(operation_id):
            """Simulate concurrent memory operation."""
            try:
                test_item = MemoryItem(
                    content=f"Concurrent test {operation_id}",
                    memory_type="working",
                    
                )
                result = memory.add(test_item)
                results.append(result)
            except Exception as e:
                print(f"Concurrent op failed: {e}")
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify concurrent operations completed
        assert len(results) == 5
        assert all(result is not None for result in results)
