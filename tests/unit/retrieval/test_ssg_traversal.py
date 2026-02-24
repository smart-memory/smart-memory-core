"""
Unit tests for Similarity Graph Traversal (SSG) algorithms.
"""

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import Mock, patch
import numpy as np

from smartmemory.retrieval.ssg_traversal import SimilarityGraphTraversal
from smartmemory.models.memory_item import MemoryItem


@pytest.fixture
def mock_smart_memory():
    """Create a mock SmartMemory instance."""
    sm = Mock()
    sm._graph = Mock()
    sm._graph.nodes = Mock()
    sm._graph.get_neighbors = Mock(return_value=[])
    return sm


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    store = Mock()
    store.search = Mock(return_value=[])
    store.get = Mock(return_value=None)
    return store


@pytest.fixture
def ssg_traversal(mock_smart_memory, mock_vector_store):
    """Create SSG traversal instance with mocks."""
    with patch('smartmemory.stores.vector.vector_store.VectorStore', return_value=mock_vector_store):
        ssg = SimilarityGraphTraversal(mock_smart_memory)
        ssg.vector_store = mock_vector_store
        return ssg


class TestSimilarityGraphTraversal:
    """Test SSG traversal algorithms."""
    
    def test_initialization(self, mock_smart_memory):
        """Test SSG initialization."""
        with patch('smartmemory.stores.vector.vector_store.VectorStore'):
            ssg = SimilarityGraphTraversal(mock_smart_memory)
            
            assert ssg.sm == mock_smart_memory
            assert ssg.graph == mock_smart_memory._graph
            assert ssg.default_max_results == 15
            assert ssg.min_before_early_stop == 8
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        similarity = SimilarityGraphTraversal._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 0.001
        
        # Orthogonal vectors
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        similarity = SimilarityGraphTraversal._cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 0.001
        
        # Opposite vectors
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        similarity = SimilarityGraphTraversal._cosine_similarity(a, b)
        assert abs(similarity - (-1.0)) < 0.001
    
    def test_embed_query_success(self, ssg_traversal):
        """Test query embedding success."""
        mock_embedding = np.array([0.1, 0.2, 0.3])
        
        with patch('smartmemory.plugins.embedding.create_embeddings', return_value=mock_embedding):
            result = ssg_traversal._embed_query("test query")
            
            assert result == [0.1, 0.2, 0.3]
    
    def test_embed_query_failure(self, ssg_traversal):
        """Test query embedding failure handling."""
        with patch('smartmemory.plugins.embedding.create_embeddings', return_value=None):
            result = ssg_traversal._embed_query("test query")
            
            assert result is None
    
    def test_get_memory_item_success(self, ssg_traversal, mock_smart_memory):
        """Test retrieving memory item."""
        mock_item = MemoryItem(
            item_id="test_123",
            content="Test content",
            memory_type="semantic"
        )
        mock_smart_memory._graph.nodes.get_node.return_value = mock_item
        
        result = ssg_traversal._get_memory_item("test_123")
        
        assert result == mock_item
        mock_smart_memory._graph.nodes.get_node.assert_called_once_with("test_123")
    
    def test_get_memory_item_not_found(self, ssg_traversal, mock_smart_memory):
        """Test retrieving non-existent memory item."""
        mock_smart_memory._graph.nodes.get_node.side_effect = Exception("Not found")
        
        result = ssg_traversal._get_memory_item("nonexistent")
        
        assert result is None
    
    def test_get_neighbors_graph_only(self, ssg_traversal, mock_smart_memory, mock_vector_store):
        """Test getting neighbors from graph relationships."""
        mock_neighbor = Mock()
        mock_neighbor.item_id = "neighbor_1"
        mock_smart_memory._graph.get_neighbors.return_value = [mock_neighbor]
        mock_vector_store.get.return_value = None  # No vector data
        
        neighbors = ssg_traversal._get_neighbors("test_123", set())
        
        assert "neighbor_1" in neighbors
    
    def test_get_neighbors_vector_only(self, ssg_traversal, mock_smart_memory, mock_vector_store):
        """Test getting neighbors from vector similarity."""
        mock_smart_memory._graph.get_neighbors.side_effect = Exception("No graph neighbors")
        
        mock_vector_store.get.return_value = {
            'embedding': [0.1, 0.2, 0.3]
        }
        mock_vector_store.search.return_value = [
            {'id': 'similar_1'},
            {'id': 'similar_2'}
        ]
        
        neighbors = ssg_traversal._get_neighbors("test_123", set())
        
        assert "similar_1" in neighbors
        assert "similar_2" in neighbors
    
    def test_get_neighbors_excludes_visited(self, ssg_traversal, mock_smart_memory, mock_vector_store):
        """Test that visited nodes are excluded from neighbors."""
        mock_neighbor = Mock()
        mock_neighbor.item_id = "visited_node"
        mock_smart_memory._graph.get_neighbors.return_value = [mock_neighbor]
        
        visited = {"visited_node"}
        neighbors = ssg_traversal._get_neighbors("test_123", visited)
        
        assert "visited_node" not in neighbors
    
    def test_select_best_by_query_similarity(self, ssg_traversal, mock_vector_store):
        """Test selecting best candidate by query similarity."""
        query_embedding = [1.0, 0.0, 0.0]
        
        # Mock candidate embeddings
        mock_vector_store.get.side_effect = [
            {'embedding': [0.9, 0.1, 0.0]},  # candidate_1: high similarity
            {'embedding': [0.0, 1.0, 0.0]},  # candidate_2: low similarity
        ]
        
        candidates = ["candidate_1", "candidate_2"]
        best = ssg_traversal._select_best_by_query_similarity(candidates, query_embedding)
        
        assert best == "candidate_1"
    
    def test_should_stop_early_minimum_not_reached(self, ssg_traversal):
        """Test early stopping doesn't trigger before minimum results."""
        extracted = [Mock() for _ in range(5)]  # Less than min_before_early_stop (8)
        
        should_stop = ssg_traversal._should_stop_early(extracted, "next_candidate", [1.0, 0.0])
        
        assert should_stop is False
    
    def test_should_stop_early_triggers(self, ssg_traversal, mock_vector_store):
        """Test early stopping triggers when extracted > candidate."""
        # Create 8 extracted items
        extracted = []
        for i in range(8):
            item = Mock()
            item.item_id = f"item_{i}"
            extracted.append(item)
        
        query_embedding = [1.0, 0.0, 0.0]
        
        # Mock: best extracted has high similarity (0.9)
        mock_vector_store.get.side_effect = [
            {'embedding': [0.9, 0.1, 0.0]},  # item_0
            {'embedding': [0.9, 0.1, 0.0]},  # item_1
            {'embedding': [0.9, 0.1, 0.0]},  # item_2
            {'embedding': [0.9, 0.1, 0.0]},  # item_3
            {'embedding': [0.9, 0.1, 0.0]},  # item_4
            {'embedding': [0.9, 0.1, 0.0]},  # item_5
            {'embedding': [0.9, 0.1, 0.0]},  # item_6
            {'embedding': [0.9, 0.1, 0.0]},  # item_7
            {'embedding': [0.5, 0.5, 0.0]},  # next_candidate: low similarity
        ]
        
        should_stop = ssg_traversal._should_stop_early(extracted, "next_candidate", query_embedding)
        
        assert should_stop is True
    
    def test_query_traversal_no_anchor(self, ssg_traversal, mock_vector_store):
        """Test query_traversal when no anchor is found."""
        mock_vector_store.search.return_value = []  # No anchor
        
        with patch('smartmemory.plugins.embedding.create_embeddings', return_value=[1.0, 0.0]):
            results = ssg_traversal.query_traversal("test query")
        
        assert results == []
    
    def test_query_traversal_basic_flow(self, ssg_traversal, mock_smart_memory, mock_vector_store):
        """Test basic query_traversal flow."""
        query_embedding = [1.0, 0.0, 0.0]
        
        # Mock anchor search
        mock_vector_store.search.return_value = [{'id': 'anchor_123'}]
        
        # Mock anchor item
        anchor_item = MemoryItem(
            item_id="anchor_123",
            content="Anchor content",
            memory_type="semantic"
        )
        mock_smart_memory._graph.nodes.get_node.return_value = anchor_item
        
        # Mock no neighbors (will stop after anchor)
        mock_smart_memory._graph.get_neighbors.return_value = []
        mock_vector_store.get.return_value = None
        
        with patch('smartmemory.plugins.embedding.create_embeddings', return_value=query_embedding):
            results = ssg_traversal.query_traversal("test query", max_results=5)
        
        assert len(results) == 1
        assert results[0].item_id == "anchor_123"
    
    def test_triangulation_fulldim_no_anchor(self, ssg_traversal, mock_vector_store):
        """Test triangulation_fulldim when no anchor is found."""
        mock_vector_store.search.return_value = []
        
        with patch('smartmemory.plugins.embedding.create_embeddings', return_value=[1.0, 0.0]):
            results = ssg_traversal.triangulation_fulldim("test query")
        
        assert results == []
    
    def test_select_by_centroid_distance(self, ssg_traversal, mock_vector_store):
        """Test centroid-based candidate selection."""
        query_embedding = [1.0, 0.0, 0.0]
        
        # Mock current chunk embedding
        mock_vector_store.get.side_effect = [
            {'embedding': [0.9, 0.1, 0.0]},  # current
            {'embedding': [0.8, 0.2, 0.0]},  # candidate_1: closer centroid
            {'embedding': [0.0, 1.0, 0.0]},  # candidate_2: farther centroid
        ]
        
        candidates = ["candidate_1", "candidate_2"]
        best = ssg_traversal._select_by_centroid_distance(
            "current_123",
            candidates,
            query_embedding
        )
        
        assert best == "candidate_1"
    
    def test_cache_operations(self, ssg_traversal):
        """Test similarity cache operations."""
        # Add to cache
        ssg_traversal.similarity_cache["test_key"] = 0.95
        
        stats = ssg_traversal.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["cache_enabled"] is True
        
        # Clear cache
        ssg_traversal.clear_cache()
        stats = ssg_traversal.get_cache_stats()
        assert stats["cache_size"] == 0


class TestSSGIntegration:
    """Integration tests for SSG with real-ish data."""
    
    def test_multi_hop_traversal_simulation(self, ssg_traversal, mock_smart_memory, mock_vector_store):
        """Simulate multi-hop traversal across related chunks."""
        query_embedding = [1.0, 0.0, 0.0]
        
        # Mock anchor
        mock_vector_store.search.return_value = [{'id': 'chunk_1'}]
        
        # Mock items
        items = {
            'chunk_1': MemoryItem(item_id='chunk_1', content='First chunk', memory_type='semantic'),
            'chunk_2': MemoryItem(item_id='chunk_2', content='Second chunk', memory_type='semantic'),
            'chunk_3': MemoryItem(item_id='chunk_3', content='Third chunk', memory_type='semantic'),
        }
        
        def get_node_side_effect(item_id):
            return items.get(item_id)
        
        mock_smart_memory._graph.nodes.get_node.side_effect = get_node_side_effect
        
        # Mock neighbors: chunk_1 -> chunk_2 -> chunk_3
        def get_neighbors_side_effect(chunk_id):
            if chunk_id == 'chunk_1':
                neighbor = Mock()
                neighbor.item_id = 'chunk_2'
                return [neighbor]
            elif chunk_id == 'chunk_2':
                neighbor = Mock()
                neighbor.item_id = 'chunk_3'
                return [neighbor]
            return []
        
        mock_smart_memory._graph.get_neighbors.side_effect = get_neighbors_side_effect
        
        # Mock embeddings for similarity scoring
        def get_embedding_side_effect(item_id):
            embeddings = {
                'chunk_1': {'embedding': [0.9, 0.1, 0.0]},
                'chunk_2': {'embedding': [0.8, 0.2, 0.0]},
                'chunk_3': {'embedding': [0.7, 0.3, 0.0]},
            }
            return embeddings.get(item_id)
        
        mock_vector_store.get.side_effect = get_embedding_side_effect
        
        with patch('smartmemory.plugins.embedding.create_embeddings', return_value=query_embedding):
            results = ssg_traversal.query_traversal("test query", max_results=3)
        
        # Should traverse all 3 chunks
        assert len(results) == 3
        assert results[0].item_id == 'chunk_1'
        assert results[1].item_id == 'chunk_2'
        assert results[2].item_id == 'chunk_3'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
