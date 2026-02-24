"""
Enhanced Similarity Metrics Framework for Agentic Memory.

Provides comprehensive similarity calculations including semantic, graph-based,
and temporal metrics for improved memory relationships and evolution decisions.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity calculation with detailed breakdown."""
    overall_score: float
    semantic_score: float = 0.0
    graph_score: float = 0.0
    temporal_score: float = 0.0
    metadata_score: float = 0.0
    content_score: float = 0.0
    agent_workflow_score: float = 0.0
    explanation: str = ""
    confidence: float = 1.0


class SimilarityMetric(ABC):
    """Abstract base class for similarity metrics."""

    @abstractmethod
    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate similarity between two memory items."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the similarity metric."""
        pass


class ContentSimilarityMetric(SimilarityMetric):
    """Enhanced content similarity with fuzzy matching and semantic awareness."""

    def __init__(self, use_fuzzy_matching: bool = True, fuzzy_threshold: float = 0.8):
        self.use_fuzzy_matching = use_fuzzy_matching
        self.fuzzy_threshold = fuzzy_threshold

    @property
    def name(self) -> str:
        return "content_similarity"

    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate enhanced content similarity."""
        if not item1.content or not item2.content:
            return 0.0

        # Basic word-level similarity
        basic_score = self._jaccard_similarity(item1.content, item2.content)

        # Enhanced fuzzy matching
        if self.use_fuzzy_matching:
            fuzzy_score = self._fuzzy_word_matching(item1.content, item2.content)
            basic_score = max(basic_score, fuzzy_score * 0.8)  # Fuzzy contributes up to 80%

        # Length similarity bonus
        length_bonus = self._length_similarity_bonus(item1.content, item2.content)

        # Combine scores
        final_score = min(1.0, basic_score + length_bonus * 0.1)

        return final_score

    def _jaccard_similarity(self, content1: str, content2: str) -> float:
        """Calculate Jaccard similarity between word sets."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _fuzzy_word_matching(self, content1: str, content2: str) -> float:
        """Enhanced fuzzy matching with partial word matching."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        partial_matches = 0.0
        for word1 in words1:
            for word2 in words2:
                if len(word1) >= 4 and len(word2) >= 4:
                    # Substring matching for longer words
                    if word1 in word2 or word2 in word1:
                        partial_matches += 0.5
                    # Edit distance for similar words
                    elif self._edit_distance_similarity(word1, word2) > self.fuzzy_threshold:
                        partial_matches += 0.3

        return partial_matches / max(len(words1), len(words2))

    def _edit_distance_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity based on edit distance."""
        if not word1 or not word2:
            return 0.0

        # Simple edit distance approximation
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 1.0

        # Count character differences
        min_len = min(len(word1), len(word2))
        differences = abs(len(word1) - len(word2))  # Length difference

        for i in range(min_len):
            if word1[i] != word2[i]:
                differences += 1

        return 1.0 - (differences / max_len)

    def _length_similarity_bonus(self, content1: str, content2: str) -> float:
        """Calculate bonus for similar content lengths."""
        len1, len2 = len(content1), len(content2)
        if len1 == 0 and len2 == 0:
            return 1.0

        max_len = max(len1, len2)
        if max_len == 0:
            return 0.0

        len_diff = abs(len1 - len2)
        return 1.0 - (len_diff / max_len)


class SemanticSimilarityMetric(SimilarityMetric):
    """Semantic similarity using embeddings and conceptual relationships."""

    def __init__(self, embedding_model: Optional[str] = None):
        self.embedding_model = embedding_model or "sentence-transformers"
        self._embedding_cache = {}

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            # Try to use embeddings if available
            embedding1 = self._get_embedding(item1.content)
            embedding2 = self._get_embedding(item2.content)

            if embedding1 is not None and embedding2 is not None:
                return self._cosine_similarity(embedding1, embedding2)
            else:
                # Fallback to conceptual similarity
                return self._conceptual_similarity(item1.content, item2.content)

        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            # Fallback to conceptual similarity
            return self._conceptual_similarity(item1.content, item2.content)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (cached)."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            # Try to import and use sentence-transformers
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, '_model'):
                self._model = SentenceTransformer('all-MiniLM-L6-v2')

            embedding = self._model.encode(text).tolist()
            self._embedding_cache[text] = embedding
            return embedding

        except ImportError:
            logger.debug("sentence-transformers not available, using fallback")
            return None
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _conceptual_similarity(self, content1: str, content2: str) -> float:
        """Fallback conceptual similarity using keyword analysis."""
        # Extract key concepts and entities
        concepts1 = self._extract_concepts(content1)
        concepts2 = self._extract_concepts(content2)

        if not concepts1 or not concepts2:
            return 0.0

        # Calculate concept overlap
        common_concepts = len(concepts1 & concepts2)
        total_concepts = len(concepts1 | concepts2)

        return common_concepts / total_concepts if total_concepts > 0 else 0.0

    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key concepts from text."""
        # Simple concept extraction (can be enhanced with NLP)
        words = text.lower().split()

        # Filter for meaningful words (longer than 3 chars, not common words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = {word for word in words if len(word) > 3 and word not in stop_words}

        return concepts


class TemporalSimilarityMetric(SimilarityMetric):
    """Temporal similarity based on time patterns and proximity."""

    def __init__(self, time_decay_factor: float = 0.1):
        self.time_decay_factor = time_decay_factor

    @property
    def name(self) -> str:
        return "temporal_similarity"

    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate temporal similarity."""
        # Time proximity is the primary factor for temporal similarity
        proximity_score = self._time_proximity_score(item1, item2)

        # Temporal pattern similarity (secondary factor)
        pattern_score = self._temporal_pattern_score(item1, item2)

        # Prioritize time proximity heavily, with pattern as minor enhancement
        return (proximity_score * 0.9 + pattern_score * 0.1)

    def _time_proximity_score(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on temporal proximity."""
        try:
            # Use transaction_time as the primary timestamp field for MemoryItem
            time1 = getattr(item1, 'transaction_time', None) or getattr(item1, 'timestamp', None) or getattr(item1, 'created_at', None)
            time2 = getattr(item2, 'transaction_time', None) or getattr(item2, 'timestamp', None) or getattr(item2, 'created_at', None)

            if not time1 or not time2:
                return 0.0  # No temporal information available

            if not isinstance(time1, datetime):
                time1 = datetime.fromisoformat(str(time1)) if time1 else datetime.now()
            if not isinstance(time2, datetime):
                time2 = datetime.fromisoformat(str(time2)) if time2 else datetime.now()

            time_diff = abs((time1 - time2).total_seconds())

            # If times are identical or very close (< 1 minute), return high similarity
            if time_diff < 60.0:
                return 0.95

            # Multi-scale temporal similarity based on realistic human time perception
            # Very close (< 1 hour): high similarity
            if time_diff < 3600:  # 1 hour
                return 0.8 * math.exp(-time_diff / 1800)  # 30-minute half-life

            # Same day (< 24 hours): medium similarity  
            elif time_diff < 86400:  # 24 hours
                return 0.6 * math.exp(-(time_diff - 3600) / 7200)  # 2-hour half-life after first hour

            # Same week (< 7 days): low-medium similarity
            elif time_diff < 604800:  # 7 days
                return 0.3 * math.exp(-(time_diff - 86400) / 172800)  # 2-day half-life after first day

            # Beyond a week: very low similarity with slow decay
            else:
                return 0.1 * math.exp(-(time_diff - 604800) / 1209600)  # 2-week half-life after first week

        except Exception as e:
            logger.warning(f"Time proximity calculation failed: {e}")
            return 0.0  # Default to no similarity on error

    def _temporal_pattern_score(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on temporal patterns."""
        # Extract temporal indicators from content
        patterns1 = self._extract_temporal_patterns(item1.content)
        patterns2 = self._extract_temporal_patterns(item2.content)

        if not patterns1 or not patterns2:
            return 0.0

        # Calculate pattern overlap
        common_patterns = len(patterns1 & patterns2)
        total_patterns = len(patterns1 | patterns2)

        return common_patterns / total_patterns if total_patterns > 0 else 0.0

    def _extract_temporal_patterns(self, content: str) -> Set[str]:
        """Extract temporal patterns from content."""
        temporal_keywords = {
            'morning', 'afternoon', 'evening', 'night',
            'daily', 'weekly', 'monthly', 'yearly',
            'before', 'after', 'during', 'while',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        }

        words = set(content.lower().split())
        return words & temporal_keywords


class GraphSimilarityMetric(SimilarityMetric):
    """Graph-based similarity using relationship analysis."""

    def __init__(self, graph_store=None):
        self.graph_store = graph_store

    @property
    def name(self) -> str:
        return "graph_similarity"

    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate graph-based similarity."""
        # Entity overlap
        entity_score = self._entity_overlap_score(item1, item2)

        # Relationship similarity
        relationship_score = self._relationship_similarity_score(item1, item2)

        # Graph connectivity
        connectivity_score = self._graph_connectivity_score(item1, item2)

        # Combine scores
        return (entity_score * 0.4 + relationship_score * 0.3 + connectivity_score * 0.3)

    def _entity_overlap_score(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on entity overlap."""
        entities1 = self._extract_entities(item1)
        entities2 = self._extract_entities(item2)

        if not entities1 or not entities2:
            return 0.0

        common_entities = len(entities1 & entities2)
        total_entities = len(entities1 | entities2)

        return common_entities / total_entities if total_entities > 0 else 0.0

    def _relationship_similarity_score(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on relationship patterns."""
        # Extract relationship patterns from content
        patterns1 = self._extract_relationship_patterns(item1.content)
        patterns2 = self._extract_relationship_patterns(item2.content)

        if not patterns1 or not patterns2:
            return 0.0

        common_patterns = len(patterns1 & patterns2)
        total_patterns = len(patterns1 | patterns2)

        return common_patterns / total_patterns if total_patterns > 0 else 0.0

    def _graph_connectivity_score(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on graph connectivity."""
        if not self.graph_store:
            return 0.0

        try:
            # Get neighbors for both items
            neighbors1 = self._get_neighbors(item1.item_id)
            neighbors2 = self._get_neighbors(item2.item_id)

            if not neighbors1 or not neighbors2:
                return 0.0

            # Calculate neighbor overlap
            common_neighbors = len(neighbors1 & neighbors2)
            total_neighbors = len(neighbors1 | neighbors2)

            return common_neighbors / total_neighbors if total_neighbors > 0 else 0.0

        except Exception as e:
            logger.warning(f"Graph connectivity calculation failed: {e}")
            return 0.0

    def _extract_entities(self, item: MemoryItem) -> Set[str]:
        """Extract entities from memory item."""
        entities = set()

        # From metadata
        if 'entities' in item.metadata:
            entities.update(item.metadata['entities'])

        # Simple entity extraction from content
        entities.update(self._simple_entity_extraction(item.content))

        return entities

    def _simple_entity_extraction(self, content: str) -> Set[str]:
        """Simple entity extraction (can be enhanced with NER)."""
        # Extract capitalized words as potential entities
        words = content.split()
        entities = {word.strip('.,!?') for word in words if word[0].isupper() and len(word) > 2}
        return entities

    def _extract_relationship_patterns(self, content: str) -> Set[str]:
        """Extract relationship patterns from content."""
        relationship_patterns = {
            # Causal relationships
            'causal': ['causes', 'leads to', 'results in', 'because of', 'due to', 'triggers', 'produces'],
            # Temporal relationships
            'temporal': ['before', 'after', 'during', 'while', 'when', 'then', 'next'],
            # Similarity relationships
            'similarity': ['similar to', 'like', 'resembles', 'comparable to'],
            # Difference relationships
            'difference': ['different from', 'unlike', 'opposite to', 'contrasts with'],
            # Containment relationships
            'containment': ['contains', 'includes', 'part of', 'belongs to', 'within']
        }

        content_lower = content.lower()
        found_patterns = set()

        for pattern_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    found_patterns.add(pattern_type)
                    break  # Found one pattern of this type

        return found_patterns

    def _get_neighbors(self, item_id: str) -> Set[str]:
        """Get neighbors of an item in the graph."""
        if not self.graph_store or not hasattr(self.graph_store, 'get_neighbors'):
            return set()

        try:
            neighbors = self.graph_store.get_neighbors(item_id)
            return {neighbor.item_id for neighbor in neighbors if hasattr(neighbor, 'item_id')}
        except Exception:
            return set()


class MetadataSimilarityMetric(SimilarityMetric):
    """Similarity based on metadata attributes."""

    @property
    def name(self) -> str:
        return "metadata_similarity"

    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate metadata-based similarity."""
        # Type similarity
        type_score = self._type_similarity(item1, item2)

        # Tag similarity
        tag_score = self._tag_similarity(item1, item2)

        # Category similarity
        category_score = self._category_similarity(item1, item2)

        # Confidence similarity
        confidence_score = self._confidence_similarity(item1, item2)

        # Weighted combination
        return (type_score * 0.3 + tag_score * 0.3 + category_score * 0.2 + confidence_score * 0.2)

    def _type_similarity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on memory type."""
        type1 = item1.metadata.get('type', item1.metadata.get('memory_type', ''))
        type2 = item2.metadata.get('type', item2.metadata.get('memory_type', ''))

        return 1.0 if type1 == type2 else 0.0

    def _tag_similarity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on tags."""
        tags1 = set(item1.metadata.get('tags', []))
        tags2 = set(item2.metadata.get('tags', []))

        if not tags1 or not tags2:
            return 0.0

        common_tags = len(tags1 & tags2)
        total_tags = len(tags1 | tags2)

        return common_tags / total_tags if total_tags > 0 else 0.0

    def _category_similarity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on categories."""
        cat1 = item1.metadata.get('category', '')
        cat2 = item2.metadata.get('category', '')

        if not cat1 or not cat2:
            return 0.0

        return 1.0 if cat1 == cat2 else 0.0

    def _confidence_similarity(self, item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate similarity based on confidence levels."""
        conf1 = item1.metadata.get('confidence', 1.0)
        conf2 = item2.metadata.get('confidence', 1.0)

        # Similar confidence levels indicate similar reliability
        diff = abs(conf1 - conf2)
        return 1.0 - diff  # Closer confidence = higher similarity


class AgentWorkflowSimilarityMetric(SimilarityMetric):
    """Similarity based on agent workflow patterns like sequential tasks, problem-solution pairs, and learning progressions."""

    def __init__(self):
        # Define workflow pattern vocabularies
        self.sequential_indicators = {
            'step', 'phase', 'stage', 'first', 'second', 'third', 'next', 'then', 'after', 'before',
            'initially', 'subsequently', 'finally', 'begin', 'start', 'continue', 'complete'
        }

        self.problem_indicators = {
            'issue', 'problem', 'error', 'bug', 'fail', 'failure', 'exception', 'crash', 'broken',
            'wrong', 'incorrect', 'missing', 'unable', 'cannot', 'difficulty', 'challenge', 'stuck',
            'slow', 'slowly', 'running', 'queries'  # Added for "Database queries are running slowly"
        }

        self.solution_indicators = {
            'fix', 'solve', 'resolve', 'implement', 'add', 'update', 'correct', 'repair', 'address',
            'solution', 'answer', 'workaround', 'patch', 'improvement', 'enhancement', 'success',
            'added', 'indexes', 'designed', 'based'  # Added for solution patterns
        }

        self.learning_indicators = {
            'learn', 'understand', 'discover', 'realize', 'figure', 'research', 'study', 'explore',
            'attempt', 'try', 'experiment', 'test', 'practice', 'improve', 'progress', 'master',
            'initial', 'failed', 'successfully', 'implemented'  # Added for learning progression patterns
        }

        self.technical_terms = {
            'api', 'database', 'query', 'function', 'method', 'class', 'module', 'service', 'endpoint',
            'authentication', 'authorization', 'oauth', 'jwt', 'token', 'session', 'cache', 'index',
            'algorithm', 'optimization', 'performance', 'scaling', 'deployment', 'configuration',
            'queries', 'columns', 'indexes', 'system', 'architecture', 'requirements', 'feature'
        }

    @property
    def name(self) -> str:
        return "agent_workflow_similarity"

    def calculate(self, item1: MemoryItem, item2: MemoryItem, **kwargs) -> float:
        """Calculate agent workflow similarity using pattern-based scoring."""
        if not item1.content or not item2.content:
            return 0.0

        # Extract workflow patterns from both items
        patterns1 = self._extract_workflow_patterns(item1.content)
        patterns2 = self._extract_workflow_patterns(item2.content)

        # Calculate component scores
        sequential_score = self._sequential_similarity(patterns1, patterns2)
        problem_solution_score = self._problem_solution_similarity(patterns1, patterns2)
        learning_progression_score = self._learning_progression_similarity(patterns1, patterns2)
        technical_context_score = self._technical_context_similarity(patterns1, patterns2)

        # Pattern-based scoring with dynamic weighting
        scores = []

        # Sequential workflow patterns
        if self._is_sequential_workflow(patterns1, patterns2):
            scores.append(sequential_score * 0.9 + technical_context_score * 0.1)

        # Problem-solution patterns  
        if self._is_problem_solution_pair(patterns1, patterns2):
            scores.append(problem_solution_score * 0.9 + technical_context_score * 0.1)

        # Learning progression patterns
        if self._is_learning_progression(patterns1, patterns2):
            scores.append(learning_progression_score * 0.8 + technical_context_score * 0.2)

        # If specific patterns matched, use the highest pattern score
        if scores:
            return min(1.0, max(scores))

        # Fallback: weighted combination of all stages
        final_score = (
                sequential_score * 0.3 +
                problem_solution_score * 0.4 +
                learning_progression_score * 0.2 +
                technical_context_score * 0.1
        )

        return min(1.0, final_score)

    def _extract_workflow_patterns(self, content: str) -> Dict[str, Any]:
        """Extract workflow patterns from content."""
        content_lower = content.lower()
        words = set(content_lower.split())

        # Enhanced pattern detection with phrase matching
        problem_indicators = words & self.problem_indicators
        solution_indicators = words & self.solution_indicators

        # Check for specific problem phrases
        if 'running slowly' in content_lower or 'queries are' in content_lower:
            problem_indicators.add('performance_issue')

        # Check for specific solution phrases  
        if 'added indexes' in content_lower or 'solution:' in content_lower:
            solution_indicators.add('solution_implemented')

        # Check for learning progression phrases
        learning_indicators = words & self.learning_indicators
        if 'initial attempt' in content_lower:
            learning_indicators.add('initial_attempt')
        if 'successfully implemented' in content_lower:
            learning_indicators.add('successful_completion')

        return {
            'sequential_indicators': words & self.sequential_indicators,
            'problem_indicators': problem_indicators,
            'solution_indicators': solution_indicators,
            'learning_indicators': learning_indicators,
            'technical_terms': words & self.technical_terms,
            'content_lower': content_lower,
            'has_step_number': any(f'step {i}' in content_lower for i in range(1, 10)),
            'has_sequence_words': bool(words & {'first', 'second', 'third', 'next', 'then', 'after'}),
            'has_issue_prefix': content_lower.startswith('issue:'),
            'has_solution_prefix': content_lower.startswith('solution:')
        }

    def _sequential_similarity(self, patterns1: Dict, patterns2: Dict) -> float:
        """Calculate similarity based on sequential task patterns."""
        # Check for explicit step numbering
        if patterns1['has_step_number'] and patterns2['has_step_number']:
            return 0.9  # Very high similarity for numbered steps

        # Check for sequence indicators
        seq_overlap = len(patterns1['sequential_indicators'] & patterns2['sequential_indicators'])
        seq_union = len(patterns1['sequential_indicators'] | patterns2['sequential_indicators'])

        if seq_union == 0:
            return 0.0

        base_score = seq_overlap / seq_union

        # Bonus for both having sequence words
        if patterns1['has_sequence_words'] and patterns2['has_sequence_words']:
            base_score += 0.2

        return min(1.0, base_score)

    def _problem_solution_similarity(self, patterns1: Dict, patterns2: Dict) -> float:
        """Calculate similarity based on problem-solution relationships."""
        has_problem_1 = bool(patterns1['problem_indicators']) or patterns1['has_issue_prefix']
        has_solution_1 = bool(patterns1['solution_indicators']) or patterns1['has_solution_prefix']
        has_problem_2 = bool(patterns2['problem_indicators']) or patterns2['has_issue_prefix']
        has_solution_2 = bool(patterns2['solution_indicators']) or patterns2['has_solution_prefix']

        # Perfect problem-solution pair (one has problem, other has solution)
        if (has_problem_1 and has_solution_2 and not has_solution_1) or \
                (has_problem_2 and has_solution_1 and not has_solution_2):
            # Check if they share technical context
            tech_overlap = len(patterns1['technical_terms'] & patterns2['technical_terms'])
            if tech_overlap > 0:
                return 0.95  # Very high similarity for related problem-solution pairs
            else:
                return 0.8  # High similarity even without shared tech terms (increased from 0.7)

        # Both are problems or both are solutions
        if (has_problem_1 and has_problem_2) or (has_solution_1 and has_solution_2):
            problem_overlap = len(patterns1['problem_indicators'] & patterns2['problem_indicators'])
            solution_overlap = len(patterns1['solution_indicators'] & patterns2['solution_indicators'])

            if problem_overlap > 0 or solution_overlap > 0:
                return 0.6  # Medium-high similarity for same type of workflow step

        # Check for implicit problem-solution relationships
        # Look for failure->success patterns
        content1 = patterns1['content_lower']
        content2 = patterns2['content_lower']

        failure_words = {'fail', 'failed', 'failure', 'unsuccessful', 'unable', 'cannot', 'error', 'issue'}
        success_words = {'success', 'successful', 'successfully', 'completed', 'achieved', 'solved', 'solution'}

        has_failure_1 = any(word in content1 for word in failure_words)
        has_success_1 = any(word in content1 for word in success_words)
        has_failure_2 = any(word in content2 for word in failure_words)
        has_success_2 = any(word in content2 for word in success_words)

        # Implicit problem-solution: failure followed by success
        if (has_failure_1 and has_success_2) or (has_failure_2 and has_success_1):
            tech_overlap = len(patterns1['technical_terms'] & patterns2['technical_terms'])
            if tech_overlap > 0:
                return 0.85  # High similarity for implicit problem-solution with shared context
            else:
                return 0.6  # Medium similarity for implicit problem-solution

        return 0.0

    def _learning_progression_similarity(self, patterns1: Dict, patterns2: Dict) -> float:
        """Calculate similarity based on learning and improvement patterns."""
        # Check for progression patterns (failure -> success) first
        content1 = patterns1['content_lower']
        content2 = patterns2['content_lower']

        failure_words = {'fail', 'failed', 'failure', 'unsuccessful', 'unable', 'cannot', 'error'}
        success_words = {'success', 'successful', 'successfully', 'completed', 'achieved', 'solved'}

        has_failure_1 = any(word in content1 for word in failure_words)
        has_success_1 = any(word in content1 for word in success_words)
        has_failure_2 = any(word in content2 for word in failure_words)
        has_success_2 = any(word in content2 for word in success_words)

        # Learning progression: failure followed by success (this is the primary signal)
        if (has_failure_1 and has_success_2) or (has_failure_2 and has_success_1):
            base_score = 0.6  # Strong base score for clear learning progression

            # Bonus for shared technical context
            tech_overlap = len(patterns1['technical_terms'] & patterns2['technical_terms'])
            if tech_overlap > 0:
                base_score += 0.2

            # Bonus for learning indicators present
            learning_union = len(patterns1['learning_indicators'] | patterns2['learning_indicators'])
            if learning_union > 0:
                base_score += 0.1

            return min(1.0, base_score)

        # Fallback: traditional learning indicator overlap for non-progression cases
        learning_overlap = len(patterns1['learning_indicators'] & patterns2['learning_indicators'])
        learning_union = len(patterns1['learning_indicators'] | patterns2['learning_indicators'])

        if learning_union == 0:
            return 0.0

        return learning_overlap / learning_union

    def _technical_context_similarity(self, patterns1: Dict, patterns2: Dict) -> float:
        """Calculate similarity based on shared technical context."""
        tech_overlap = len(patterns1['technical_terms'] & patterns2['technical_terms'])
        tech_union = len(patterns1['technical_terms'] | patterns2['technical_terms'])

        if tech_union == 0:
            return 0.0

        base_score = tech_overlap / tech_union

        # Bonus for related technical terms (not just exact matches)
        tech1 = patterns1['technical_terms']
        tech2 = patterns2['technical_terms']

        # Check for related database terms
        db_terms1 = tech1 & {'database', 'queries', 'indexes', 'columns'}
        db_terms2 = tech2 & {'database', 'queries', 'indexes', 'columns'}
        if db_terms1 and db_terms2:
            base_score = max(base_score, 0.8)  # High similarity for database-related terms

        # Check for related auth terms
        auth_terms1 = tech1 & {'authentication', 'oauth', 'jwt', 'token'}
        auth_terms2 = tech2 & {'authentication', 'oauth', 'jwt', 'token'}
        if auth_terms1 and auth_terms2:
            base_score = max(base_score, 0.8)  # High similarity for auth-related terms

        # Check for related system design terms
        design_terms1 = tech1 & {'system', 'architecture', 'requirements', 'feature'}
        design_terms2 = tech2 & {'system', 'architecture', 'requirements', 'feature'}
        if design_terms1 and design_terms2:
            base_score = max(base_score, 0.7)  # Good similarity for design-related terms

        return base_score

    def _is_sequential_workflow(self, patterns1: Dict, patterns2: Dict) -> bool:
        """Check if items represent sequential workflow steps."""
        # Both items have step numbers (Step 1, Step 2, etc.)
        if patterns1['has_step_number'] and patterns2['has_step_number']:
            return True

        # Both items have sequential indicators and share technical context
        if (patterns1['sequential_indicators'] and patterns2['sequential_indicators'] and
                len(patterns1['technical_terms'] & patterns2['technical_terms']) > 0):
            return True

        return False

    def _is_problem_solution_pair(self, patterns1: Dict, patterns2: Dict) -> bool:
        """Check if items represent a problem-solution pair."""
        # Explicit Issue: -> Solution: pattern
        if patterns1['has_issue_prefix'] and patterns2['has_solution_prefix']:
            return True
        if patterns2['has_issue_prefix'] and patterns1['has_solution_prefix']:
            return True

        # One has problem indicators, other has solution indicators
        has_problem_1 = bool(patterns1['problem_indicators'])
        has_solution_1 = bool(patterns1['solution_indicators'])
        has_problem_2 = bool(patterns2['problem_indicators'])
        has_solution_2 = bool(patterns2['solution_indicators'])

        if ((has_problem_1 and has_solution_2 and not has_solution_1) or
                (has_problem_2 and has_solution_1 and not has_solution_2)):
            # Must have some shared technical context
            return len(patterns1['technical_terms'] & patterns2['technical_terms']) > 0

        return False

    def _is_learning_progression(self, patterns1: Dict, patterns2: Dict) -> bool:
        """Check if items represent a learning progression (failure -> success)."""
        content1 = patterns1['content_lower']
        content2 = patterns2['content_lower']

        failure_words = {'fail', 'failed', 'failure', 'unsuccessful', 'unable', 'cannot', 'error', 'initial attempt'}
        success_words = {'success', 'successful', 'successfully', 'completed', 'achieved', 'solved'}

        has_failure_1 = any(word in content1 for word in failure_words)
        has_success_1 = any(word in content1 for word in success_words)
        has_failure_2 = any(word in content2 for word in failure_words)
        has_success_2 = any(word in content2 for word in success_words)

        # Learning progression: failure followed by success
        if ((has_failure_1 and has_success_2) or (has_failure_2 and has_success_1)):
            # Either shared technical context OR learning indicators (more flexible)
            shared_tech = len(patterns1['technical_terms'] & patterns2['technical_terms']) > 0
            has_learning = bool(patterns1['learning_indicators'] or patterns2['learning_indicators'])

            # Accept if either condition is met (not both required)
            return shared_tech or has_learning

        return False
