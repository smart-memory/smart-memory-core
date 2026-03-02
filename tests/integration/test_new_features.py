"""
Tests for features in SmartMemory.

Run with:
    PYTHONPATH=/Users/ruze/reg/my/SmartMemory/smart-memory-core conda run -n memory pytest tests/integration/test_new_features.py -v
"""
import pytest


pytestmark = pytest.mark.integration


class TestSemHashDeduplication:
    """Test SemHash-based deduplication."""
    
    def test_singularize_text(self):
        from smartmemory.utils.deduplication import singularize_text
        
        # Test plurals
        assert singularize_text("companies") == "company"
        assert singularize_text("apples") == "apple"
        assert singularize_text("people") == "person"
        # Non-plurals should stay the same
        assert singularize_text("apple") == "apple"
    
    def test_semhash_deduplicator_basic(self):
        from smartmemory.utils.deduplication import SemHashDeduplicator
        
        dedup = SemHashDeduplicator(similarity_threshold=0.95)
        items = ['Apple Inc.', 'apple inc', 'APPLE INC.', 'Microsoft', 'Google']
        
        result, duplicates = dedup.deduplicate(items)
        
        # Should reduce duplicates
        assert len(result) < len(items)
        # Microsoft and Google should remain
        result_lower = [r.lower() for r in result]
        assert 'microsoft' in result_lower
        assert 'google' in result_lower
    
    def test_semhash_deduplicate_entities(self):
        from smartmemory.utils.deduplication import semhash_deduplicate_entities
        
        entities = [
            {'name': 'Apple Inc.', 'type': 'org'},
            {'name': 'apple inc', 'type': 'org'},
            {'name': 'Microsoft', 'type': 'org'},
            {'name': 'Google', 'type': 'org'},
        ]
        
        result, clusters = semhash_deduplicate_entities(entities, similarity_threshold=0.95)
        
        # Should have fewer entities
        assert len(result) <= len(entities)
        # All results should have names
        assert all('name' in e for e in result)


class TestTokenTracking:
    """Test token usage tracking."""
    
    def test_token_tracker_basic(self):
        from smartmemory.utils.token_tracking import TokenTracker
        
        tracker = TokenTracker()
        
        # Track some usage
        tracker.track(prompt_tokens=100, completion_tokens=50, model='gpt-5-mini')
        tracker.track(prompt_tokens=200, completion_tokens=100, model='gpt-5-mini')
        
        usage = tracker.get_usage()
        
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 150
        assert usage.total_tokens == 450
        assert usage.call_count == 2
        assert 'gpt-5-mini' in usage.models_used
    
    def test_token_tracker_reset(self):
        from smartmemory.utils.token_tracking import TokenTracker
        
        tracker = TokenTracker()
        tracker.track(prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        
        usage = tracker.get_usage()
        assert usage.total_tokens == 0
        assert usage.call_count == 0
    
    def test_token_tracker_from_dict_response(self):
        from smartmemory.utils.token_tracking import TokenTracker
        
        tracker = TokenTracker()
        
        # Simulate OpenAI-style response
        response = {
            'usage': {
                'prompt_tokens': 150,
                'completion_tokens': 75,
                'total_tokens': 225
            },
            'model': 'gpt-4o'
        }
        
        tracker.track(response=response)
        usage = tracker.get_usage()
        
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 75
    
    def test_global_tracker(self):
        from smartmemory.utils.token_tracking import track_usage, get_usage, reset_usage
        
        reset_usage()
        track_usage(prompt_tokens=50, completion_tokens=25)
        
        usage = get_usage()
        assert usage['total_tokens'] == 75
        
        reset_usage()
    
    def test_cost_estimation(self):
        from smartmemory.utils.token_tracking import AggregatedUsage, estimate_cost
        
        usage = AggregatedUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        
        cost = estimate_cost(usage, 'gpt-5-mini')
        assert cost > 0
        assert isinstance(cost, float)


class TestChunking:
    """Test text chunking utilities."""
    
    def test_chunk_text_basic(self):
        from smartmemory.utils.chunking import chunk_text
        
        text = "This is a test. " * 100  # ~1600 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size (approximately)
        for chunk in chunks:
            assert len(chunk) <= 600  # Allow some buffer for sentence boundaries
    
    def test_chunk_text_small_text(self):
        from smartmemory.utils.chunking import chunk_text
        
        text = "Short text."
        chunks = chunk_text(text, chunk_size=500)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunked_extractor_init(self):
        from smartmemory.utils.chunking import ChunkedExtractor
        
        # Mock extractor
        class MockExtractor:
            def extract(self, text, **kwargs):
                return {'entities': [], 'relations': []}
        
        chunked = ChunkedExtractor(
            MockExtractor(),
            chunk_size=5000,
            parallel=True,
            max_workers=4
        )
        
        assert chunked.chunk_size == 5000
        assert chunked.parallel == True
        assert chunked.max_workers == 4


class TestHybridRetrieval:
    """Test hybrid BM25+Embedding retrieval."""
    
    def test_hybrid_retriever_init(self):
        from smartmemory.utils.hybrid_retrieval import HybridRetriever
        
        items = ['Apple Inc.', 'Microsoft Corporation', 'Google LLC']
        retriever = HybridRetriever(items=items)
        
        assert retriever.items == items
    
    def test_hybrid_retriever_empty(self):
        from smartmemory.utils.hybrid_retrieval import HybridRetriever
        
        retriever = HybridRetriever(items=[])
        results = retriever.retrieve("test query")
        
        assert results == []
    
    def test_entity_matcher_init(self):
        from smartmemory.utils.hybrid_retrieval import EntityMatcher
        
        entities = [
            {'name': 'Apple Inc.', 'type': 'org'},
            {'name': 'Microsoft', 'type': 'org'},
        ]
        
        matcher = EntityMatcher(entities=entities)
        assert len(matcher.entity_names) == 2


class TestClusteringImports:
    """Test that clustering modules import correctly."""
    
    def test_embedding_clusterer_import(self):
        from smartmemory.clustering.embedding import EmbeddingClusterer
        assert EmbeddingClusterer is not None
    
    def test_hybrid_deduplicator_import(self):
        from smartmemory.clustering.deduplicator import HybridDeduplicator
        assert HybridDeduplicator is not None
    
    def test_deduplicate_extraction_import(self):
        from smartmemory.memory.pipeline.stages.clustering import deduplicate_extraction
        assert callable(deduplicate_extraction)
    
    def test_graph_aggregator_import(self):
        from smartmemory.clustering import GraphAggregator
        assert GraphAggregator is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
