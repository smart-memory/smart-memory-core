"""
Example: Creating a Custom Extractor Plugin

This example demonstrates how to create a custom extractor plugin that
extracts entities and relationships from text.
"""

from typing import Dict, Any
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata


class RegexExtractor(ExtractorPlugin):
    """
    Simple regex-based extractor for emails, URLs, and phone numbers.
    
    This extractor uses regular expressions to identify common patterns
    in text and extract them as entities.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery and registration."""
        return PluginMetadata(
            name="regex_extractor",
            version="1.0.0",
            author="Your Name",
            description="Regex-based extraction for emails, URLs, and phone numbers",
            plugin_type="extractor",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["regex", "email", "url", "phone"]
        )
    
    def __init__(self):
        """Initialize the regex extractor."""
        import re
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text using regex patterns.
        
        Args:
            text: The text to extract from
        
        Returns:
            Dictionary with 'entities' and 'relations' keys
        """
        entities = []
        
        # Extract emails
        for match in self.patterns['email'].finditer(text):
            entities.append({
                'name': match.group(),
                'type': 'email',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract URLs
        for match in self.patterns['url'].finditer(text):
            entities.append({
                'name': match.group(),
                'type': 'url',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract phone numbers
        for match in self.patterns['phone'].finditer(text):
            entities.append({
                'name': match.group(),
                'type': 'phone',
                'start': match.start(),
                'end': match.end()
            })
        
        return {
            'entities': entities,
            'relations': []  # No relationships extracted
        }


class CustomNERExtractor(ExtractorPlugin):
    """
    Custom Named Entity Recognition extractor with domain-specific entities.
    
    This extractor identifies domain-specific entities like product names,
    company names, and technical terms.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="custom_ner_extractor",
            version="1.0.0",
            author="Your Name",
            description="Domain-specific NER extractor",
            plugin_type="extractor",
            tags=["ner", "domain-specific"]
        )
    
    def __init__(self):
        """Initialize with domain-specific entity lists."""
        # Example: Tech company names
        self.known_companies = {
            'OpenAI', 'Google', 'Microsoft', 'Amazon', 'Meta',
            'Apple', 'Tesla', 'SpaceX', 'Anthropic'
        }
        
        # Example: Programming languages
        self.known_languages = {
            'Python', 'JavaScript', 'TypeScript', 'Java', 'C++',
            'Rust', 'Go', 'Ruby', 'Swift', 'Kotlin'
        }
        
        # Example: Technical terms
        self.known_tech_terms = {
            'AI', 'ML', 'NLP', 'API', 'GPU', 'CPU',
            'LLM', 'RAG', 'Vector Database', 'Embedding'
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract domain-specific entities from text.
        
        Args:
            text: The text to extract from
        
        Returns:
            Dictionary with 'entities' and 'relations' keys
        """
        entities = []
        triples = []
        
        # Find company mentions
        for company in self.known_companies:
            if company in text:
                entities.append({
                    'name': company,
                    'type': 'organization'
                })
        
        # Find programming language mentions
        for lang in self.known_languages:
            if lang in text:
                entities.append({
                    'name': lang,
                    'type': 'programming_language'
                })
        
        # Find technical term mentions
        for term in self.known_tech_terms:
            if term in text:
                entities.append({
                    'name': term,
                    'type': 'technical_term'
                })
        
        # Create simple co-occurrence relationships
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                triples.append((
                    entities[i]['name'],
                    'mentioned_with',
                    entities[i + 1]['name']
                ))
        
        return {
            'entities': entities,
            'relations': triples  # Return as relations (list of tuples)
        }


# Example usage
if __name__ == "__main__":
    # Test regex extractor
    regex_extractor = RegexExtractor()
    
    text1 = "Contact me at john.doe@example.com or visit https://example.com. Call 555-123-4567."
    result1 = regex_extractor.extract(text1)
    
    print("Regex Extractor Results:")
    print(f"  Text: {text1}")
    print(f"  Entities found: {len(result1['entities'])}")
    for entity in result1['entities']:
        print(f"    - {entity['type']}: {entity['name']}")
    print()
    
    # Test custom NER extractor
    ner_extractor = CustomNERExtractor()
    
    text2 = "OpenAI developed GPT-4 using Python and advanced NLP techniques. Google also works on LLM research."
    result2 = ner_extractor.extract(text2)
    
    print("Custom NER Extractor Results:")
    print(f"  Text: {text2}")
    print(f"  Entities found: {len(result2['entities'])}")
    for entity in result2['entities']:
        print(f"    - {entity['type']}: {entity['name']}")
    print(f"  Relations found: {len(result2['relations'])}")
    for triple in result2['relations']:
        print(f"    - {triple[0]} --[{triple[1]}]--> {triple[2]}")
