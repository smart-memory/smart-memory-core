"""
Opinion Synthesis Evolver 

Synthesizes opinions from episodic patterns, similar to how
EpisodicToZettelEvolver creates zettels from events.

Detects recurring patterns in episodic memories and forms
opinions with confidence scores.

Example: After 10 interactions where user chose functional style,
creates opinion: "User prefers functional programming" (confidence: 0.85)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.models.memory_item import MemoryItem
from smartmemory.models.opinion import OpinionMetadata, Disposition
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

logger = logging.getLogger(__name__)


@dataclass
class OpinionSynthesisConfig(MemoryBaseModel):
    """Configuration for opinion synthesis."""
    min_pattern_occurrences: int = 3  # Minimum times a pattern must occur
    min_confidence: float = 0.5  # Minimum confidence to create opinion
    lookback_days: int = 30  # How far back to look for patterns
    
    # Disposition for opinion formation
    skepticism: float = 0.5
    literalism: float = 0.5
    empathy: float = 0.5
    
    # LLM settings for pattern detection
    use_llm: bool = True
    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class OpinionSynthesisRequest(StageRequest):
    lookback_days: int = 30
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class OpinionSynthesisEvolver(EvolverPlugin):
    """
    Synthesizes opinions from episodic memory patterns.
    
    Pattern types detected:
    - Preferences: Repeated choices/selections
    - Behaviors: Recurring actions or workflows
    - Themes: Recurring topics or interests
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="opinion_synthesis",
            version="1.0.0",
            author="SmartMemory Team",
            description="Synthesizes opinions from episodic patterns ",
            plugin_type="evolver",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["synthesis", "opinion", ]
        )

    def __init__(self, config: Optional[OpinionSynthesisConfig] = None):
        self.config = config or OpinionSynthesisConfig()

    def evolve(self, memory, log=None):
        """
        Main evolution method - synthesizes opinions from episodic patterns.

        Args:
            memory: SmartMemory instance with access to episodic store
            log: Optional logger
        """
        log = log or logger
        cfg = self.config
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.opinion_synthesis", {"memory_id": memory_id, "lookback_days": cfg.lookback_days}):
            disposition = Disposition(
                skepticism=cfg.skepticism,
                literalism=cfg.literalism,
                empathy=cfg.empathy,
            )

            # 1. Get recent episodic memories
            episodic_items = self._get_recent_episodic(memory, cfg.lookback_days)
            if not episodic_items:
                log.info("No episodic memories found for opinion synthesis")
                return

            log.info(f"Analyzing {len(episodic_items)} episodic memories for patterns")

            # 2. Detect patterns
            patterns = self._detect_patterns(episodic_items, cfg)
            if not patterns:
                log.info("No significant patterns detected")
                return

            log.info(f"Detected {len(patterns)} patterns")

            # 3. Form opinions from patterns
            opinions_created = 0
            for pattern in patterns:
                # Check if opinion already exists
                existing = self._find_existing_opinion(memory, pattern['subject'])
                if existing:
                    log.debug(f"Opinion already exists for {pattern['subject']}, skipping")
                    continue

                # Calculate confidence based on pattern strength and disposition
                confidence = self._calculate_confidence(pattern, disposition)
                if confidence < cfg.min_confidence:
                    log.debug(f"Confidence {confidence:.2f} below threshold for {pattern['subject']}")
                    continue

                # Create opinion
                opinion = self._create_opinion(pattern, confidence, disposition)
                self._store_opinion(memory, opinion)
                opinions_created += 1
                log.info(f"Created opinion: {opinion.content[:50]}... (confidence: {confidence:.2f})")

            log.info(f"Opinion synthesis complete: {opinions_created} opinions created")

    def _get_recent_episodic(self, memory, days: int) -> List[MemoryItem]:
        """Get recent episodic memories."""
        try:
            # Try to use episodic store's method if available
            if hasattr(memory, 'episodic') and hasattr(memory.episodic, 'get_events_since'):
                return memory.episodic.get_events_since(days=days)
            
            # Fallback: search for episodic memories
            if hasattr(memory, 'search'):
                results = memory.search(
                    query="*",
                    memory_types=['episodic'],
                    limit=500,
                )
                return results if results else []
            
            return []
        except Exception as e:
            logger.error(f"Failed to get episodic memories: {e}")
            return []

    def _detect_patterns(self, items: List[MemoryItem], cfg: OpinionSynthesisConfig) -> List[Dict[str, Any]]:
        """
        Detect patterns in episodic memories.
        
        Returns list of pattern dicts with:
        - subject: What the pattern is about
        - subject_type: Type of pattern (preference, behavior, theme)
        - description: Human-readable description
        - evidence_ids: item_ids that support this pattern
        - strength: How strong the pattern is (0-1)
        """
        patterns = []
        
        # Simple pattern detection: group by metadata tags/topics
        tag_counts: Dict[str, List[str]] = {}
        topic_counts: Dict[str, List[str]] = {}
        
        for item in items:
            # Count tags
            tags = item.metadata.get('tags', [])
            for tag in tags:
                if tag not in tag_counts:
                    tag_counts[tag] = []
                tag_counts[tag].append(item.item_id)
            
            # Count topics
            topics = item.metadata.get('topics', [])
            for topic in topics:
                if topic not in topic_counts:
                    topic_counts[topic] = []
                topic_counts[topic].append(item.item_id)
        
        # Create patterns from frequent tags
        for tag, evidence_ids in tag_counts.items():
            if len(evidence_ids) >= cfg.min_pattern_occurrences:
                patterns.append({
                    'subject': tag,
                    'subject_type': 'theme',
                    'description': f"User frequently engages with {tag}",
                    'evidence_ids': evidence_ids,
                    'strength': min(1.0, len(evidence_ids) / 10),
                })
        
        # Create patterns from frequent topics
        for topic, evidence_ids in topic_counts.items():
            if len(evidence_ids) >= cfg.min_pattern_occurrences:
                patterns.append({
                    'subject': topic,
                    'subject_type': 'interest',
                    'description': f"User shows interest in {topic}",
                    'evidence_ids': evidence_ids,
                    'strength': min(1.0, len(evidence_ids) / 10),
                })
        
        # TODO: Add LLM-based pattern detection for more sophisticated analysis
        # if cfg.use_llm:
        #     patterns.extend(self._detect_patterns_llm(items, cfg))
        
        return patterns

    def _find_existing_opinion(self, memory, subject: str) -> Optional[MemoryItem]:
        """Check if an opinion about this subject already exists."""
        try:
            if hasattr(memory, 'search'):
                results = memory.search(
                    query=subject,
                    memory_types=['opinion'],
                    limit=5,
                )
                for result in (results or []):
                    if result.metadata.get('subject') == subject:
                        return result
            return None
        except Exception:
            return None

    def _calculate_confidence(self, pattern: Dict[str, Any], disposition: Disposition) -> float:
        """
        Calculate confidence score for an opinion based on pattern and disposition.
        
        Higher skepticism = lower confidence
        More evidence = higher confidence
        """
        base_confidence = pattern['strength']
        
        # Adjust for skepticism (higher skepticism = need more evidence)
        skepticism_factor = 1 - (disposition.skepticism * 0.3)
        
        # Adjust for evidence count
        evidence_count = len(pattern['evidence_ids'])
        evidence_factor = min(1.0, evidence_count / 10)
        
        confidence = base_confidence * skepticism_factor * (0.5 + 0.5 * evidence_factor)
        return min(1.0, max(0.0, confidence))

    def _create_opinion(self, pattern: Dict[str, Any], confidence: float, disposition: Disposition) -> MemoryItem:
        """Create an opinion MemoryItem from a pattern."""
        opinion_id = f"opinion_{hashlib.sha256(pattern['subject'].encode()).hexdigest()[:12]}"
        
        opinion_meta = OpinionMetadata(
            confidence=confidence,
            formed_from=pattern['evidence_ids'],
            disposition=disposition,
            subject=pattern['subject'],
            subject_type=pattern['subject_type'],
        )
        
        return MemoryItem(
            item_id=opinion_id,
            content=pattern['description'],
            memory_type='opinion',
            metadata={
                **opinion_meta.to_dict(),
                'pattern_type': pattern['subject_type'],
            }
        )

    def _store_opinion(self, memory, opinion: MemoryItem):
        """Store the opinion in memory."""
        try:
            if hasattr(memory, 'add'):
                memory.add(opinion)
            elif hasattr(memory, 'vector_store') and hasattr(memory.vector_store, 'add'):
                memory.vector_store.add(opinion)
        except Exception as e:
            logger.error(f"Failed to store opinion: {e}")
