"""
Unit tests for ConversationAwareLLMExtractor.
"""

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import MagicMock, patch

from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor
from smartmemory.conversation.context import ConversationContext


class TestConversationAwareLLMExtractor:
    """Test suite for conversation-aware extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ConversationAwareLLMExtractor(
            max_history_turns=3,
            resolve_coreferences=True,
            extract_speaker_relations=True
        )
    
    @pytest.fixture
    def conversation_context(self):
        """Create sample conversation context."""
        context = ConversationContext(
            conversation_id="test_conv",
            
        )
        
        # Add conversation history
        context.turn_history = [
            {"role": "user", "content": "What is machine learning?", "timestamp": "2024-01-01T10:00:00Z"},
            {"role": "assistant", "content": "Machine learning is a subset of AI...", "timestamp": "2024-01-01T10:00:05Z"},
        ]
        
        # Add known entities
        context.entities = [
            {"name": "machine learning", "type": "concept"},
            {"name": "AI", "type": "concept"}
        ]
        
        # Add topics
        context.topics = ["machine learning", "AI"]
        
        return context
    
    def test_build_context_text(self, extractor, conversation_context):
        """Test context text building from conversation history."""
        context_text = extractor._build_context_text(conversation_context)
        
        assert "Recent conversation:" in context_text
        assert "[User]: What is machine learning?" in context_text
        assert "[Assistant]: Machine learning is a subset of AI" in context_text
        assert "Known entities in conversation:" in context_text
        assert "machine learning (concept)" in context_text
        assert "Conversation topics:" in context_text
    
    def test_build_context_text_limits_history(self, extractor):
        """Test that context text limits history to max_history_turns."""
        context = ConversationContext()
        context.turn_history = [
            {"role": "user", "content": f"Turn {i}"} for i in range(10)
        ]
        
        context_text = extractor._build_context_text(context)
        
        # Should only include last 3 turns (max_history_turns=3)
        assert "Turn 7" in context_text
        assert "Turn 8" in context_text
        assert "Turn 9" in context_text
        assert "Turn 6" not in context_text
    
    def test_is_demonstrative_reference(self, extractor):
        """Test demonstrative reference detection."""
        assert extractor._is_demonstrative_reference("that algorithm")
        assert extractor._is_demonstrative_reference("this method")
        assert extractor._is_demonstrative_reference("these techniques")
        assert not extractor._is_demonstrative_reference("algorithm")
        assert not extractor._is_demonstrative_reference("it")
    
    def test_resolve_from_context_pronoun_it(self, extractor, conversation_context):
        """Test resolving 'it' pronoun to last non-person entity."""
        entity = {"name": "it", "entity_type": "concept"}
        
        resolved = extractor._resolve_from_context(entity, conversation_context)
        
        assert resolved is not None
        assert resolved['name'] == "AI"  # Last non-person entity
        assert resolved['resolved_from'] == "it"
        assert resolved['confidence'] == 0.7
    
    def test_resolve_from_context_demonstrative(self, extractor):
        """Test resolving demonstrative reference."""
        context = ConversationContext()
        context.entities = [
            {"name": "neural network", "type": "concept"},
            {"name": "deep learning", "type": "concept"}
        ]
        
        entity = {"name": "that network", "entity_type": "concept"}
        
        resolved = extractor._resolve_from_context(entity, context)
        
        assert resolved is not None
        assert resolved['name'] == "neural network"
        assert resolved['resolved_from'] == "that network"
    
    def test_resolve_coreferences(self, extractor, conversation_context):
        """Test coreference resolution for multiple entities."""
        entities = [
            {"name": "neural networks", "entity_type": "concept"},
            {"name": "it", "entity_type": "concept"},
            {"name": "deep learning", "entity_type": "concept"}
        ]
        
        resolved = extractor._resolve_coreferences(entities, conversation_context)
        
        assert len(resolved) == 3
        assert resolved[0]['name'] == "neural networks"
        assert resolved[1]['name'] == "AI"  # "it" resolved to last entity
        assert resolved[2]['name'] == "deep learning"
    
    def test_normalize_entities(self, extractor):
        """Test entity normalization."""
        raw_entities = [
            {"name": "Python", "entity_type": "PROGRAMMING_LANGUAGE", "confidence": 0.9},
            {"name": "  TensorFlow  ", "entity_type": "library"},
            {"name": "", "entity_type": "concept"},  # Should be filtered
            "invalid"  # Should be filtered
        ]
        
        normalized = extractor._normalize_entities(raw_entities)
        
        assert len(normalized) == 2
        assert normalized[0]['name'] == "Python"
        assert normalized[0]['entity_type'] == "programming_language"
        assert normalized[0]['confidence'] == 0.9
        assert normalized[1]['name'] == "TensorFlow"
        assert normalized[1]['entity_type'] == "library"
        assert normalized[1]['confidence'] == 0.8  # Default
    
    def test_normalize_relations(self, extractor):
        """Test relation normalization."""
        raw_relations = [
            {"subject": "Python", "predicate": "used_for", "object": "machine learning"},
            {"subject": "  TensorFlow  ", "predicate": "  is_a  ", "object": "  library  "},
            {"subject": "", "predicate": "invalid", "object": "relation"},  # Should be filtered
            "invalid"  # Should be filtered
        ]
        
        normalized = extractor._normalize_relations(raw_relations)
        
        assert len(normalized) == 2
        assert normalized[0]['subject'] == "Python"
        assert normalized[0]['predicate'] == "used_for"
        assert normalized[0]['object'] == "machine learning"
        assert normalized[1]['subject'] == "TensorFlow"
        assert normalized[1]['predicate'] == "is_a"
        assert normalized[1]['object'] == "library"
    
    @patch('smartmemory.plugins.extractors.conversation_aware_llm.call_llm')
    def test_extract_with_context(self, mock_call_llm, extractor, conversation_context):
        """Test extraction with conversation context."""
        # Mock LLM responses — call_llm returns (parsed_dict, raw_text) tuples
        mock_call_llm.side_effect = [
            # Entity extraction response
            (
                {
                    'entities': [
                        {'name': 'neural networks', 'entity_type': 'concept', 'confidence': 0.9}
                    ]
                },
                '{"entities": [{"name": "neural networks", "entity_type": "concept", "confidence": 0.9}]}'
            ),
            # Relation extraction response
            (
                {
                    'relations': [
                        {'subject': 'neural networks', 'predicate': 'part_of', 'object': 'machine learning'}
                    ]
                },
                '{"relations": [{"subject": "neural networks", "predicate": "part_of", "object": "machine learning"}]}'
            ),
            # Speaker relation extraction response
            (
                {
                    'speaker_relations': [
                        {'subject': 'User', 'predicate': 'ASKS_ABOUT', 'object': 'neural networks'}
                    ]
                },
                '{"speaker_relations": [{"subject": "User", "predicate": "ASKS_ABOUT", "object": "neural networks"}]}'
            ),
        ]
        
        # Mock API key
        extractor._get_api_key = MagicMock(return_value="test_key")
        
        result = extractor.extract(
            "Tell me about neural networks",
            conversation_context=conversation_context
        )
        
        assert 'entities' in result
        assert len(result['entities']) == 1
        assert result['entities'][0]['name'] == 'neural networks'
        
        assert 'relations' in result
        assert len(result['relations']) == 1
        
        assert 'speaker_relations' in result
        assert len(result['speaker_relations']) == 1
        assert result['speaker_relations'][0]['predicate'] == 'ASKS_ABOUT'
    
    @patch('smartmemory.plugins.extractors.conversation_aware_llm.call_llm')
    def test_extract_without_context_falls_back(self, mock_call_llm, extractor):
        """Test that extraction without context falls back to base LLMExtractor."""
        # Mock parent class extract method
        with patch.object(ConversationAwareLLMExtractor.__bases__[0], 'extract') as mock_parent_extract:
            mock_parent_extract.return_value = {'entities': [], 'relations': []}
            
            result = extractor.extract("Some text", conversation_context=None)
            
            # Should call parent extract (no context = fallback to base class)
            mock_parent_extract.assert_called_once_with("Some text")
    
    def test_metadata(self):
        """Test plugin metadata."""
        metadata = ConversationAwareLLMExtractor.metadata()
        
        assert metadata.name == "conversation_aware_llm"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == "extractor"
        assert "conversation" in metadata.tags
        assert "coreference" in metadata.tags
