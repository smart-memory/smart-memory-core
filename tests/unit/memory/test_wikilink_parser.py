"""
Unit tests for Wikilink Parser

Tests the wikilink parsing functionality for Zettelkasten system.
"""

import pytest


pytestmark = pytest.mark.unit
from smartmemory.memory.types.wikilink_parser import (
    WikilinkParser, ParsedLinks, 
    parse_wikilinks, extract_all_links
)


class TestWikilinkParser:
    """Test WikilinkParser class."""
    
    def test_extract_simple_wikilink(self):
        """Test extracting a simple wikilink."""
        parser = WikilinkParser()
        content = "This mentions [[Machine Learning]] in the text."
        
        links = parser.extract_wikilinks(content)
        
        assert len(links) == 1
        assert "Machine Learning" in links
    
    def test_extract_multiple_wikilinks(self):
        """Test extracting multiple wikilinks."""
        parser = WikilinkParser()
        content = "Topics: [[AI]], [[ML]], and [[Deep Learning]]."
        
        links = parser.extract_wikilinks(content)
        
        assert len(links) == 3
        assert "AI" in links
        assert "ML" in links
        assert "Deep Learning" in links
    
    def test_extract_wikilink_with_alias(self):
        """Test extracting wikilink with alias syntax."""
        parser = WikilinkParser()
        content = "See [[Machine Learning|ML]] for details."
        
        links = parser.extract_wikilinks(content)
        
        assert len(links) == 1
        assert "Machine Learning" in links  # Should extract title, not alias
    
    def test_extract_concepts(self):
        """Test extracting concept mentions."""
        parser = WikilinkParser()
        content = "Key concepts: ((Gradient Descent)) and ((Backpropagation))."
        
        concepts = parser.extract_concepts(content)
        
        assert len(concepts) == 2
        assert "Gradient Descent" in concepts
        assert "Backpropagation" in concepts
    
    def test_extract_hashtags(self):
        """Test extracting hashtags."""
        parser = WikilinkParser()
        content = "Tags: #machine-learning #ai #deep-learning"
        
        hashtags = parser.extract_hashtags(content)
        
        assert len(hashtags) == 3
        assert "machine-learning" in hashtags
        assert "ai" in hashtags
        assert "deep-learning" in hashtags
    
    def test_parse_all_link_types(self):
        """Test parsing all link types together."""
        parser = WikilinkParser()
        content = """
        # Note Title
        
        This note discusses [[Machine Learning]] and uses ((Algorithms)).
        Related to [[Deep Learning]] and ((Neural Networks)).
        
        Tags: #ai #ml #research
        """
        
        parsed = parser.parse(content)
        
        assert isinstance(parsed, ParsedLinks)
        assert len(parsed.wikilinks) == 2
        assert len(parsed.concepts) == 2
        assert len(parsed.hashtags) == 3
    
    def test_duplicate_removal(self):
        """Test that duplicates are removed."""
        parser = WikilinkParser()
        content = "[[ML]] and [[ML]] and [[ML]] again. #ai #ai #ai"
        
        parsed = parser.parse(content)
        
        assert len(parsed.wikilinks) == 1
        assert len(parsed.hashtags) == 1
    
    def test_title_to_id_conversion(self):
        """Test converting note titles to IDs."""
        parser = WikilinkParser()
        
        assert parser.title_to_id("Machine Learning") == "machine_learning"
        assert parser.title_to_id("Deep Learning (Advanced)") == "deep_learning_advanced"
        assert parser.title_to_id("AI & ML") == "ai_ml"
        assert parser.title_to_id("  Spaced  Out  ") == "spaced_out"
    
    def test_enrich_metadata(self):
        """Test enriching metadata with parsed links."""
        parser = WikilinkParser()
        
        metadata = {
            'title': 'Test Note',
            'tags': ['existing-tag'],
            'concepts': ['Existing Concept']
        }
        
        parsed = ParsedLinks(
            wikilinks=['Note A', 'Note B'],
            concepts=['New Concept'],
            hashtags=['new-tag', 'another-tag']
        )
        
        enriched = parser.enrich_metadata(metadata, parsed)
        
        assert 'existing-tag' in enriched['tags']
        assert 'new-tag' in enriched['tags']
        assert 'another-tag' in enriched['tags']
        assert 'Existing Concept' in enriched['concepts']
        assert 'New Concept' in enriched['concepts']
        assert enriched['wikilinks'] == ['Note A', 'Note B']
    
    def test_empty_content(self):
        """Test parsing empty content."""
        parser = WikilinkParser()
        parsed = parser.parse("")
        
        assert len(parsed.wikilinks) == 0
        assert len(parsed.concepts) == 0
        assert len(parsed.hashtags) == 0
    
    def test_no_links_in_content(self):
        """Test content with no links."""
        parser = WikilinkParser()
        content = "This is just plain text with no special syntax."
        
        parsed = parser.parse(content)
        
        assert len(parsed.wikilinks) == 0
        assert len(parsed.concepts) == 0
        assert len(parsed.hashtags) == 0
    
    def test_malformed_wikilinks(self):
        """Test handling of malformed wikilinks."""
        parser = WikilinkParser()
        
        # Incomplete brackets
        content1 = "This has [[incomplete bracket"
        parsed1 = parser.parse(content1)
        assert len(parsed1.wikilinks) == 0
        
        # Empty wikilink
        content2 = "This has [[]] empty"
        parsed2 = parser.parse(content2)
        assert len(parsed2.wikilinks) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parse_wikilinks_function(self):
        """Test parse_wikilinks convenience function."""
        content = "[[Note A]] and ((Concept)) #tag"
        parsed = parse_wikilinks(content)
        
        assert isinstance(parsed, ParsedLinks)
        assert len(parsed.wikilinks) == 1
        assert len(parsed.concepts) == 1
        assert len(parsed.hashtags) == 1
    
    def test_extract_all_links_function(self):
        """Test extract_all_links convenience function."""
        content = "[[Note A]] and ((Concept)) #tag"
        links = extract_all_links(content)
        
        assert isinstance(links, dict)
        assert 'wikilinks' in links
        assert 'concepts' in links
        assert 'hashtags' in links
        assert links['wikilinks'] == ['Note A']
        assert links['concepts'] == ['Concept']
        assert links['hashtags'] == ['tag']


class TestParsedLinks:
    """Test ParsedLinks dataclass."""
    
    def test_to_dict(self):
        """Test converting ParsedLinks to dictionary."""
        parsed = ParsedLinks(
            wikilinks=['A', 'B'],
            concepts=['C', 'D'],
            hashtags=['e', 'f']
        )
        
        result = parsed.to_dict()
        
        assert isinstance(result, dict)
        assert result['wikilinks'] == ['A', 'B']
        assert result['concepts'] == ['C', 'D']
        assert result['hashtags'] == ['e', 'f']


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_markdown_document(self):
        """Test parsing a complete markdown document."""
        parser = WikilinkParser()
        
        content = """# Machine Learning Overview

## Introduction
Machine learning is a subset of [[Artificial Intelligence]] that focuses
on learning from data. It uses ((Statistical Methods)) and ((Algorithms))
to improve performance.

## Key Concepts
- [[Supervised Learning]]
- [[Unsupervised Learning]]
- [[Reinforcement Learning]]

## Applications
ML is used in [[Computer Vision]], [[Natural Language Processing]], and
[[Robotics]]. Core techniques include ((Neural Networks)) and ((Decision Trees)).

Tags: #machine-learning #ai #data-science #algorithms
        """
        
        parsed = parser.parse(content)
        
        # Should extract all wikilinks
        assert 'Artificial Intelligence' in parsed.wikilinks
        assert 'Supervised Learning' in parsed.wikilinks
        assert 'Computer Vision' in parsed.wikilinks
        
        # Should extract all concepts
        assert 'Statistical Methods' in parsed.concepts
        assert 'Neural Networks' in parsed.concepts
        
        # Should extract all hashtags
        assert 'machine-learning' in parsed.hashtags
        assert 'ai' in parsed.hashtags
    
    def test_nested_brackets(self):
        """Test handling of nested or adjacent brackets."""
        parser = WikilinkParser()
        
        # Adjacent wikilinks
        content = "[[Note A]][[Note B]]"
        parsed = parser.parse(content)
        assert len(parsed.wikilinks) == 2
    
    def test_multiline_content(self):
        """Test parsing multiline content."""
        parser = WikilinkParser()
        
        content = """Line 1 with [[Link A]]
        Line 2 with [[Link B]]
        Line 3 with ((Concept))
        Line 4 with #tag"""
        
        parsed = parser.parse(content)
        
        assert len(parsed.wikilinks) == 2
        assert len(parsed.concepts) == 1
        assert len(parsed.hashtags) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
