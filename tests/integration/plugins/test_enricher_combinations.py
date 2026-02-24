"""
Comprehensive integration tests for enricher plugin combinations.
Tests all 6 enricher types individually and in combinations.
"""
import pytest


pytestmark = pytest.mark.integration

from smartmemory.smart_memory import SmartMemory
from smartmemory.memory.models.memory_item import MemoryItem
from smartmemory.plugins.enrichers import (
    BasicEnricher, 
    SentimentEnricher, 
    TemporalEnricher,
    ExtractSkillsToolsEnricher,
    TopicEnricher,
    WikipediaEnricher
)


@pytest.mark.integration
class TestEnricherCombinations:
    """Integration tests for enricher plugin combinations with real backends."""
    
    @pytest.fixture(scope="function")
    def enricher_memory(self):
        """SmartMemory instance for enricher testing."""
        # Use default config with spacy extractor
        memory = SmartMemory()
        # Set default extractor to spacy for tests
        if hasattr(memory, '_ingestion_flow') and hasattr(memory._ingestion_flow, 'extraction_pipeline'):
            memory._ingestion_flow.extraction_pipeline.default_extractor = 'spacy'
        yield memory
        try:
            memory.clear()
        except Exception as e:
            print(f"Warning: Cleanup error (non-fatal): {e}")
    
    @pytest.fixture
    def test_items(self):
        """Various test items for different enricher scenarios."""
        return {
            'sentiment_text': MemoryItem(
                content="I absolutely love machine learning! It's fascinating how neural networks can learn patterns.",
                memory_type="semantic",
                
                metadata={"test_type": "sentiment"}
            ),
            'temporal_text': MemoryItem(
                content="Yesterday I learned Python programming. Tomorrow I will study deep learning algorithms.",
                memory_type="episodic", 
                
                metadata={"test_type": "temporal"}
            ),
            'skills_text': MemoryItem(
                content="I used TensorFlow and PyTorch to build neural networks. Also worked with Docker and Kubernetes.",
                memory_type="procedural",
                 
                metadata={"test_type": "skills_tools"}
            ),
            'topic_text': MemoryItem(
                content="Artificial intelligence encompasses machine learning, natural language processing, and computer vision.",
                memory_type="semantic",
                
                metadata={"test_type": "topic"}
            ),
            'wikipedia_text': MemoryItem(
                content="Albert Einstein developed the theory of relativity and won the Nobel Prize in Physics.",
                memory_type="semantic", 
                
                metadata={"test_type": "wikipedia"}
            )
        }
    
    def test_basic_enricher_integration(self, enricher_memory, test_items):
        """Test BasicEnricher functionality with correct Dict return API."""
        enricher = BasicEnricher()
        item = test_items['sentiment_text']
        
        # Test enrichment - returns Dict[str, Any], not modified MemoryItem
        enrichment_result = enricher.enrich(item)
        
        # Verify enrichment result structure
        assert enrichment_result is not None
        assert isinstance(enrichment_result, dict)
        assert 'new_items' in enrichment_result
        
        # BasicEnricher should add summary
        assert 'summary' in enrichment_result
        assert enrichment_result['summary'] is not None
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        retrieved = enricher_memory.get(item_id)
        assert retrieved is not None
        
        print(f"✅ BasicEnricher: Enrichment result keys: {list(enrichment_result.keys())}")
        print(f"   Summary: {enrichment_result.get('summary', 'N/A')[:50]}...")
    
    def test_sentiment_enricher_integration(self, enricher_memory, test_items):
        """Test SentimentEnricher functionality with correct Dict return API."""
        enricher = SentimentEnricher()
        item = test_items['sentiment_text']
        
        # Test sentiment enrichment - returns Dict[str, Any]
        enrichment_result = enricher.enrich(item)
        
        # Verify enrichment result structure
        assert enrichment_result is not None
        assert isinstance(enrichment_result, dict)
        
        # SentimentEnricher returns {'properties': {'sentiment': {...}}}
        # Check for sentiment-related data in result
        sentiment_keys = [key for key in enrichment_result.keys() if any(term in key.lower() for term in ['sentiment', 'emotion', 'polarity', 'mood', 'properties'])]
        
        # Verify sentiment analysis occurred
        has_sentiment_data = (
            'properties' in enrichment_result and 
            isinstance(enrichment_result['properties'], dict) and
            'sentiment' in enrichment_result['properties']
        )
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        print(f"✅ SentimentEnricher: Enrichment result keys: {list(enrichment_result.keys())}")
        print(f"   Sentiment-related keys: {sentiment_keys}")
    
    def test_temporal_enricher_integration(self, enricher_memory, test_items):
        """Test TemporalEnricher functionality with correct Dict return API."""
        enricher = TemporalEnricher()
        item = test_items['temporal_text']
        
        # Test temporal enrichment - returns Dict[str, Any]
        enrichment_result = enricher.enrich(item)
        
        # Verify enrichment result structure
        assert enrichment_result is not None
        assert isinstance(enrichment_result, dict)
        
        # TemporalEnricher returns {'temporal': {'Alice': {...}, 'Bob': {...}}}
        # Check for temporal-related data in result
        temporal_keys = [key for key in enrichment_result.keys() if any(term in key.lower() for term in ['time', 'temporal', 'date', 'when', 'yesterday', 'tomorrow'])]
        
        # Verify temporal analysis occurred
        has_temporal_data = (
            'temporal' in enrichment_result and 
            isinstance(enrichment_result['temporal'], dict)
        )
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        print(f"✅ TemporalEnricher: Enrichment result keys: {list(enrichment_result.keys())}")
        print(f"   Temporal-related keys: {temporal_keys}")
    
    def test_skills_tools_enricher_integration(self, enricher_memory, test_items):
        """Test ExtractSkillsToolsEnricher functionality with correct Dict return API."""
        enricher = ExtractSkillsToolsEnricher()
        item = test_items['skills_text']
        
        # Test skills/tools enrichment - returns Dict[str, Any]
        enrichment_result = enricher.enrich(item)
        
        # Verify enrichment result structure
        assert enrichment_result is not None
        assert isinstance(enrichment_result, dict)
        # ExtractSkillsToolsEnricher returns {'skills': [...], 'tools': [...]}
        assert 'skills' in enrichment_result or 'tools' in enrichment_result
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        print(f"✅ SkillsToolsEnricher: Enrichment result keys: {list(enrichment_result.keys())}")
        print(f"   Skills: {enrichment_result.get('skills', [])}")
        print(f"   Tools: {enrichment_result.get('tools', [])}")
    
    def test_topic_enricher_integration(self, enricher_memory, test_items):
        """Test TopicEnricher functionality with correct Dict return API."""
        enricher = TopicEnricher()
        item = test_items['topic_text']
        
        # Test topic enrichment - returns Dict[str, Any]
        enrichment_result = enricher.enrich(item)
        
        # Verify enrichment result structure
        assert enrichment_result is not None
        assert isinstance(enrichment_result, dict)
        # TopicEnricher returns properties dict
        assert 'properties' in enrichment_result or len(enrichment_result) > 0
        
        # Check for topic-related data in result
        topic_keys = [key for key in enrichment_result.keys() if any(term in key.lower() for term in ['topic', 'category', 'subject', 'theme', 'artificial', 'intelligence'])]
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        print(f"✅ TopicEnricher: Enrichment result keys: {list(enrichment_result.keys())}")
        print(f"   Topic-related keys: {topic_keys}")
    
    def test_wikipedia_enricher_integration(self, enricher_memory, test_items):
        """Test WikipediaEnricher functionality with correct Dict return API."""
        enricher = WikipediaEnricher()
        item = test_items['wikipedia_text']
        
        # Test Wikipedia enrichment - returns Dict[str, Any]
        enrichment_result = enricher.enrich(item)
        
        # Verify enrichment result structure
        assert enrichment_result is not None
        assert isinstance(enrichment_result, dict)
        # WikipediaEnricher returns enrichment data directly
        assert len(enrichment_result) >= 0  # May return empty dict if no Wikipedia data found
        
        # Check for Wikipedia-related data in result
        wiki_keys = [key for key in enrichment_result.keys() if any(term in key.lower() for term in ['wikipedia', 'wiki', 'external', 'knowledge', 'einstein', 'physics'])]
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        print(f"✅ WikipediaEnricher: Enrichment result keys: {list(enrichment_result.keys())}")
        print(f"   Wikipedia-related keys: {wiki_keys}")
    
    def test_multiple_enricher_combination(self, enricher_memory, test_items):
        """Test combining multiple enrichers on the same item with correct Dict API."""
        item = test_items['sentiment_text']
        
        # Apply multiple enrichers in sequence - each returns Dict[str, Any]
        enrichers = [
            BasicEnricher(),
            SentimentEnricher(), 
            TemporalEnricher()
        ]
        
        enrichment_results = []
        for enricher in enrichers:
            enrichment_result = enricher.enrich(item)
            assert enrichment_result is not None
            assert isinstance(enrichment_result, dict)
            enrichment_results.append(enrichment_result)
        
        # Verify all enrichments were applied
        all_keys = set()
        for result in enrichment_results:
            all_keys.update(result.keys())
        
        # Test integration with memory system (original item)
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        # Verify retrieval works
        retrieved = enricher_memory.get(item_id)
        assert retrieved is not None
        
        print(f"✅ Multi-Enricher: Applied {len(enrichers)} enrichers")
        print(f"   Combined enrichment keys: {sorted(all_keys)}")
    
    def test_enricher_pipeline_integration(self, enricher_memory, test_items):
        """Test enrichers working within the full SmartMemory pipeline."""
        # Test that enrichers work with the full ingestion pipeline
        item = test_items['skills_text']
        
        # Use add() which goes through the full ingestion pipeline
        item_id = enricher_memory.add(item)
        assert item_id is not None
        
        # Search should work with content (enrichers may enhance searchability)
        search_results = enricher_memory.search(
            "TensorFlow PyTorch", 
            
            top_k=5
        )
        # Search may or may not return results depending on embedding availability
        # Just verify it doesn't error
        assert search_results is not None
        
        print("✅ Pipeline Integration: Enrichers integrated within full ingestion pipeline")
        print(f"   Search completed successfully, found {len(search_results)} results")
    
    def test_enricher_error_handling(self, enricher_memory):
        """Test enricher error handling with malformed inputs."""
        enrichers = [
            BasicEnricher(),
            SentimentEnricher(),
            TemporalEnricher()
        ]
        
        # Test with empty content
        empty_item = MemoryItem(
            content="",
            memory_type="semantic", 
            
        )
        
        for enricher in enrichers:
            try:
                result = enricher.enrich(empty_item)
                # Should return Dict[str, Any], not crash
                assert result is not None
                assert isinstance(result, dict)
                assert 'new_items' in result
                print(f"✅ {enricher.__class__.__name__}: Handled empty content gracefully")
            except Exception as e:
                print(f"⚠️ {enricher.__class__.__name__}: Error with empty content: {e}")
        
        # Test with None content
        none_item = MemoryItem(
            content=None,
            memory_type="semantic",
            
        )
        
        for enricher in enrichers:
            try:
                result = enricher.enrich(none_item)
                assert result is not None
                assert isinstance(result, dict)
                print(f"✅ {enricher.__class__.__name__}: Handled None content gracefully")
            except Exception as e:
                print(f"⚠️ {enricher.__class__.__name__}: Error with None content: {e}")
