"""
Wikilink Parser for Zettelkasten System

Parses markdown-style wikilinks and automatically creates bidirectional connections.

Supported syntax:
- [[Note Title]] - Wikilink to another note
- ((Concept)) - Concept mention
- #tag - Hashtag
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedLinks:
    """Container for parsed links from content."""
    wikilinks: List[str]  # [[Note Title]]
    concepts: List[str]   # ((Concept))
    hashtags: List[str]   # #tag
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary format."""
        return {
            'wikilinks': self.wikilinks,
            'concepts': self.concepts,
            'hashtags': self.hashtags
        }


class WikilinkParser:
    """
    Parser for Zettelkasten wikilink syntax.
    
    Extracts structured links from markdown content for automatic
    bidirectional linking.
    """
    
    # Regex patterns for different link types
    WIKILINK_PATTERN = r'\[\[([^\]]+)\]\]'  # [[Note Title]]
    CONCEPT_PATTERN = r'\(\(([^\)]+)\)\)'   # ((Concept))
    HASHTAG_PATTERN = r'#([a-zA-Z0-9_-]+)'  # #tag
    
    def __init__(self):
        """Initialize the parser with compiled regex patterns."""
        self.wikilink_re = re.compile(self.WIKILINK_PATTERN)
        self.concept_re = re.compile(self.CONCEPT_PATTERN)
        self.hashtag_re = re.compile(self.HASHTAG_PATTERN)
    
    def parse(self, content: str) -> ParsedLinks:
        """
        Parse all link types from content.
        
        Args:
            content: Markdown content to parse
            
        Returns:
            ParsedLinks object containing all extracted links
        """
        wikilinks = self.extract_wikilinks(content)
        concepts = self.extract_concepts(content)
        hashtags = self.extract_hashtags(content)
        
        return ParsedLinks(
            wikilinks=wikilinks,
            concepts=concepts,
            hashtags=hashtags
        )
    
    def extract_wikilinks(self, content: str) -> List[str]:
        """
        Extract wikilinks from content.
        
        Examples:
            [[Machine Learning]] -> "Machine Learning"
            [[Deep Learning|DL]] -> "Deep Learning"
        
        Args:
            content: Text to parse
            
        Returns:
            List of note titles referenced
        """
        matches = self.wikilink_re.findall(content)
        
        # Handle alias syntax: [[Note Title|Alias]]
        links = []
        for match in matches:
            # Split on | to get actual title (before alias)
            title = match.split('|')[0].strip()
            if title:
                links.append(title)
        
        return list(set(links))  # Remove duplicates
    
    def extract_concepts(self, content: str) -> List[str]:
        """
        Extract concept mentions from content.
        
        Examples:
            ((Gradient Descent)) -> "Gradient Descent"
            ((Neural Network)) -> "Neural Network"
        
        Args:
            content: Text to parse
            
        Returns:
            List of concepts mentioned
        """
        matches = self.concept_re.findall(content)
        return list(set(match.strip() for match in matches if match.strip()))
    
    def extract_hashtags(self, content: str) -> List[str]:
        """
        Extract hashtags from content.
        
        Examples:
            #machine-learning -> "machine-learning"
            #AI -> "AI"
        
        Args:
            content: Text to parse
            
        Returns:
            List of hashtags (without # prefix)
        """
        matches = self.hashtag_re.findall(content)
        return list(set(match.strip() for match in matches if match.strip()))
    
    def title_to_id(self, title: str) -> str:
        """
        Convert a note title to a valid note ID.
        
        Examples:
            "Machine Learning" -> "machine_learning"
            "Deep Learning (Advanced)" -> "deep_learning_advanced"
        
        Args:
            title: Note title
            
        Returns:
            Valid note ID
        """
        # Convert to lowercase
        note_id = title.lower()
        
        # Replace spaces and special chars with underscores
        note_id = re.sub(r'[^\w\s-]', '', note_id)
        note_id = re.sub(r'[-\s]+', '_', note_id)
        
        # Remove leading/trailing underscores
        note_id = note_id.strip('_')
        
        return note_id
    
    def enrich_metadata(self, metadata: Dict[str, Any], parsed_links: ParsedLinks) -> Dict[str, Any]:
        """
        Enrich metadata with parsed links.
        
        Merges parsed hashtags and concepts into existing metadata,
        avoiding duplicates.
        
        Args:
            metadata: Existing metadata dictionary
            parsed_links: Parsed links from content
            
        Returns:
            Enriched metadata dictionary
        """
        enriched = metadata.copy()
        
        # Add hashtags to tags
        existing_tags = set(enriched.get('tags', []))
        new_tags = existing_tags.union(set(parsed_links.hashtags))
        enriched['tags'] = list(new_tags)
        
        # Add concepts
        existing_concepts = set(enriched.get('concepts', []))
        new_concepts = existing_concepts.union(set(parsed_links.concepts))
        enriched['concepts'] = list(new_concepts)
        
        # Store wikilinks for reference
        enriched['wikilinks'] = parsed_links.wikilinks
        
        return enriched


class WikilinkResolver:
    """
    Resolves wikilinks to actual note IDs and creates bidirectional connections.
    
    Works with ZettelMemory to automatically link notes based on wikilink syntax.
    """
    
    def __init__(self, zettel_memory):
        """
        Initialize resolver with a ZettelMemory instance.
        
        Args:
            zettel_memory: ZettelMemory instance to resolve links against
        """
        self.zettel_memory = zettel_memory
        self.parser = WikilinkParser()
        self.title_to_id_map = {}  # Cache for title -> ID mapping
    
    def resolve_and_link(self, note_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse content, resolve wikilinks, and create bidirectional connections.
        
        Args:
            note_id: ID of the note being processed
            content: Note content with wikilinks
            metadata: Note metadata
            
        Returns:
            Enriched metadata with parsed links
        """
        # Parse links from content
        parsed_links = self.parser.parse(content)
        
        # Enrich metadata
        enriched_metadata = self.parser.enrich_metadata(metadata, parsed_links)
        
        # Create bidirectional links for wikilinks
        for title in parsed_links.wikilinks:
            target_id = self._resolve_title(title)
            if target_id:
                try:
                    self.zettel_memory.create_bidirectional_link(
                        note_id, target_id, 'LINKS_TO'
                    )
                    logger.info(f"Created wikilink: {note_id} <-> {target_id}")
                except Exception as e:
                    logger.warning(f"Failed to create wikilink {note_id} -> {target_id}: {e}")
        
        return enriched_metadata
    
    def _resolve_title(self, title: str) -> str:
        """
        Resolve a note title to its ID.
        
        Tries multiple strategies:
        1. Check cache
        2. Convert title to ID format and check if exists
        3. Search for note with matching title in metadata
        
        Args:
            title: Note title to resolve
            
        Returns:
            Note ID if found, None otherwise
        """
        # Check cache
        if title in self.title_to_id_map:
            return self.title_to_id_map[title]
        
        # Try converting title to ID
        potential_id = self.parser.title_to_id(title)
        if self._note_exists(potential_id):
            self.title_to_id_map[title] = potential_id
            return potential_id
        
        # Search for note with matching title
        note_id = self._search_by_title(title)
        if note_id:
            self.title_to_id_map[title] = note_id
            return note_id
        
        logger.warning(f"Could not resolve wikilink: [[{title}]]")
        return None
    
    def _note_exists(self, note_id: str) -> bool:
        """Check if a note with given ID exists."""
        try:
            note = self.zettel_memory.get(note_id)
            return note is not None
        except Exception:
            return False
    
    def _search_by_title(self, title: str) -> str:
        """
        Search for a note by title in metadata.
        
        Args:
            title: Title to search for
            
        Returns:
            Note ID if found, None otherwise
        """
        try:
            # Get all notes and search for matching title
            # This is a fallback - ideally notes should use consistent ID format
            all_notes = self.zettel_memory.structure._get_all_notes()
            
            for note in all_notes:
                note_title = note.metadata.get('title', '')
                if note_title.lower() == title.lower():
                    return note.item_id
            
            return None
        except Exception as e:
            logger.error(f"Error searching for title '{title}': {e}")
            return None
    
    def update_title_cache(self, note_id: str, title: str):
        """
        Manually update the title -> ID cache.
        
        Useful when adding new notes to ensure wikilinks resolve correctly.
        
        Args:
            note_id: Note ID
            title: Note title
        """
        self.title_to_id_map[title] = note_id


# Convenience functions for direct use

def parse_wikilinks(content: str) -> ParsedLinks:
    """
    Parse wikilinks from content.
    
    Convenience function for one-off parsing.
    
    Args:
        content: Content to parse
        
    Returns:
        ParsedLinks object
    """
    parser = WikilinkParser()
    return parser.parse(content)


def extract_all_links(content: str) -> Dict[str, List[str]]:
    """
    Extract all links from content as a dictionary.
    
    Args:
        content: Content to parse
        
    Returns:
        Dictionary with 'wikilinks', 'concepts', and 'hashtags' keys
    """
    parsed = parse_wikilinks(content)
    return parsed.to_dict()
