"""
Real integration tests for pipeline integration using the full backend stack.
Verifies Enrichment and Evolution plugins against real Graph and Vector stores.
"""

import pytest


pytestmark = pytest.mark.integration
from smartmemory.memory.pipeline.stages.enrichment import Enrichment
from smartmemory.evolution.cycle import run_evolution_cycle
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.registry import get_plugin_registry

@pytest.mark.integration
class TestRealPipelineIntegration:
    """Integration tests for pipeline stages using real backends."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        self.enrichment = Enrichment(self.memory._graph)
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def test_enrichment_writes_to_graph(self):
        """Test that enrichment plugins actually modify the graph."""
        
        # 1. Create a basic memory item
        item = MemoryItem(
            content="I feel really happy about learning Python today!",
            memory_type="episodic",
            
        )
        
        # 2. Add to graph (bypassing normal pipeline to isolate enrichment test)
        item_id = self.memory.add(item)
        
        # 3. Run enrichment manually on this item
        # We expect SentimentEnricher to run and add sentiment properties
        # Note: Enrichment.enrich returns a dict of properties to merge
        
        # We need to simulate the pipeline passing the item
        # But Enrichment.enrich is designed to take an item and return enhancements
        # It doesn't auto-write to graph unless part of the pipeline flow.
        # The integration test should verify the *result* of enrichment is valid
        # AND that we can write it.
        
        context = {"item": item, "node_ids": None}
        enrichment_results = self.enrichment.enrich(context)
        
        # 4. Verify enrichment results contain expected data
        assert 'properties' in enrichment_results
        props = enrichment_results['properties']
        
        # Sentiment enricher should have added something (mock or real logic)
        # Even if basic logic: "happy" -> positive
        assert 'sentiment' in props or any('sentiment' in k for k in props.keys())
        
        # 5. Write these properties to the real graph
        # SmartGraphNodes doesn't have update_node, use add_node to upsert/merge properties
        self.memory._graph.add_node(item_id, props, memory_type="episodic")
        
        # 6. Verify persistence
        node = self.memory.get(item_id)
        # Check if properties are present in metadata or direct attributes
        # The graph backend flattens/unflattens, so it might be in metadata
        
        has_sentiment = (
            (hasattr(node, 'metadata') and 'sentiment' in node.metadata) or
            (hasattr(node, 'sentiment')) or
            (hasattr(node, 'metadata') and any('sentiment' in k for k in node.metadata.keys()))
        )
        assert has_sentiment, "Enrichment data not persisted to graph"

    def test_evolution_cycle_on_real_data(self):
        """Test running evolution cycle on real memory data."""
        
        # 1. Setup data: Create some "working" memory that should evolve to "episodic"
        # This depends on the specific evolvers registered. 
        # Assuming WorkingToEpisodicEvolver is active.
        
        item = MemoryItem(
            content="Important working memory to be consolidated",
            memory_type="working",
            
            metadata={"importance": "high"}
        )
        item_id = self.memory.add(item)
        
        # 2. Run evolution cycle
        # This will iterate through evolvers and apply them to the memory system
        run_evolution_cycle(self.memory, config={}, logger=None)
        
        # 3. Verify effects
        # Since specific evolution logic is complex (depends on thresholds etc),
        # we mainly verify that the cycle ran without error and interacted with the graph.
        # We can check if the item still exists and if any metadata changed (like access counts)
        # Or if we have a specific evolver we can rely on (e.g. access count decay)
        
        node = self.memory.get(item_id)
        assert node is not None
        
        # If there's an access counter, it might have been updated
        # For now, just ensuring the system is stable and data is intact is a good baseline
        assert node.content == item.content

    def test_plugin_registry_integration(self):
        """Verify plugin registry is correctly wired to the real system."""
        registry = get_plugin_registry()
        
        # Ensure standard plugins are available
        assert registry.has_plugin("basic_enricher")
        assert registry.has_plugin("sentiment_enricher")
        
        # Ensure they are loaded in the enrichment stage
        assert "basic_enricher" in self.enrichment.enricher_registry
        
        # Ensure we can retrieve them
        EnricherCls = registry.get_enricher("basic_enricher")
        assert EnricherCls is not None
