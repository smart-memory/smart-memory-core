"""
Unit tests for extractor plugins.

Tests all extractor plugins with mocked models to verify:
- Correct output format
- Entity extraction
- Relationship extraction
- Error handling
"""

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import Mock, patch

from smartmemory.plugins.extractors.spacy import SpacyExtractor
from smartmemory.plugins.extractors.llm import LLMExtractor


class TestSpacyExtractor:
    """Test SpacyExtractor plugin."""
    
    @pytest.fixture
    def mock_spacy_model(self):
        """Create mock spaCy model."""
        mock_doc = Mock()
        mock_doc.ents = []
        
        mock_model = Mock()
        mock_model.return_value = mock_doc
        return mock_model
    
    @pytest.fixture
    def extractor(self, mock_spacy_model):
        """Create SpacyExtractor with mocked model."""
        with patch('spacy.load', return_value=mock_spacy_model):
            extractor = SpacyExtractor()
            # Ensure the mock is used
            extractor.nlp = mock_spacy_model
            return extractor
    
    def test_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor is not None
        metadata = extractor.metadata()
        assert metadata.name == "spacy"
        assert metadata.version is not None
    
    def test_extract_basic(self, extractor, mock_spacy_model):
        """Test basic extraction."""
        # Mock entities
        mock_ent1 = Mock()
        mock_ent1.text = "Apple"
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 5
        
        mock_doc = Mock()
        mock_doc.ents = [mock_ent1]
        mock_doc.sents = []
        # Mock tokens for dependency parsing check
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_spacy_model.return_value = mock_doc
        
        result = extractor.extract("Apple is a company")
        
        assert isinstance(result, dict)
        assert 'entities' in result
        assert 'relations' in result
        assert len(result['entities']) == 1
        assert result['entities'][0]['text'] == "Apple"
        assert result['entities'][0]['type'] == "organization"
    
    def test_extract_with_relationships(self, extractor, mock_spacy_model):
        """Test extraction with relationships."""
        # Mock entities
        mock_ent1 = Mock()
        mock_ent1.text = "Steve Jobs"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 10
        
        mock_ent2 = Mock()
        mock_ent2.text = "Apple"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 19
        mock_ent2.end_char = 24
        
        mock_doc = Mock()
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_doc.sents = []
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_spacy_model.return_value = mock_doc
        
        result = extractor.extract("Steve Jobs founded Apple")
        
        assert len(result['entities']) == 2
        # Relationships are extracted based on proximity
        assert isinstance(result['relations'], list)
    
    def test_extract_empty_text(self, extractor, mock_spacy_model):
        """Test extraction with empty text."""
        mock_doc = Mock()
        mock_doc.ents = []
        mock_spacy_model.return_value = mock_doc
        
        result = extractor.extract("")
        
        assert result['entities'] == []
        assert result['relations'] == []
    
    def test_entity_type_mapping(self, extractor):
        """Test entity type mapping."""
        # The method is _map_spacy_label_to_type, not _map_entity_type
        assert extractor._map_spacy_label_to_type("PERSON") == "person"
        assert extractor._map_spacy_label_to_type("ORG") == "organization"
        assert extractor._map_spacy_label_to_type("GPE") == "location"
        assert extractor._map_spacy_label_to_type("DATE") == "temporal"
        assert extractor._map_spacy_label_to_type("UNKNOWN") == "concept"


class TestLLMExtractor:
    """Test LLMExtractor plugin."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"entities": [], "relations": []}'
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def extractor(self, mock_llm_client):
        """Create LLMExtractor with mocked client."""
        with patch('litellm.completion', return_value=mock_llm_client):
            return LLMExtractor()
    
    def test_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor is not None
        metadata = extractor.metadata()
        assert metadata.name == "llm"
    
    def test_extract_basic(self, extractor, mock_llm_client):
        """Test basic LLM extraction."""
        parsed_response = {
            "entities": [
                {"name": "Python", "entity_type": "technology", "confidence": 0.9}
            ],
        }

        with patch('smartmemory.plugins.extractors.llm.call_llm') as mock_call:
            mock_call.return_value = (parsed_response, '{"entities": []}')

            result = extractor.extract("Python is a programming language")

            assert isinstance(result, dict)
            assert 'entities' in result
            assert 'relations' in result
    
    def test_extract_error_handling(self, extractor):
        """Test error handling in LLM extraction."""
        with patch('litellm.completion', side_effect=Exception("API Error")):
            result = extractor.extract("Test text")
            
            # Should return empty result on error
            assert result['entities'] == []
            assert result['relations'] == []



