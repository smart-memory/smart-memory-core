"""
Observation Synthesis Evolver 

Creates entity summaries by synthesizing facts across memories,
similar to ZettelEmergentStructure's pattern detection.

Example: Combines "Alice works at Google" + "Alice started in 2020" + 
"Alice was promoted in 2023" → Observation about Alice's career.
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.models.memory_item import MemoryItem
from smartmemory.models.opinion import ObservationMetadata
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

logger = logging.getLogger(__name__)


@dataclass
class ObservationSynthesisConfig(MemoryBaseModel):
    """Configuration for observation synthesis."""
    min_facts_per_entity: int = 2  # Minimum facts needed to create observation
    lookback_days: int = 90  # How far back to look for facts
    max_observations_per_run: int = 20  # Limit observations per evolution cycle

    # LLM settings for synthesis
    use_llm: bool = True
    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class ObservationSynthesisRequest(StageRequest):
    lookback_days: int = 90
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class ObservationSynthesisEvolver(EvolverPlugin):
    """
    Creates entity summaries by synthesizing facts across memories.
    
    Similar to ZettelEmergentStructure's detect_concept_emergence(),
    but focused on creating coherent entity descriptions.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="observation_synthesis",
            version="1.0.0",
            author="SmartMemory Team",
            description="Creates entity summaries from scattered facts ",
            plugin_type="evolver",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["synthesis", "observation", "entity"]
        )

    def __init__(self, config: Optional[ObservationSynthesisConfig] = None):
        self.config = config or ObservationSynthesisConfig()

    def evolve(self, memory, log=None):
        """
        Main evolution method - synthesizes observations from entity facts.

        Args:
            memory: SmartMemory instance
            log: Optional logger
        """
        log = log or logger
        cfg = self.config
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.observation_synthesis", {"memory_id": memory_id, "lookback_days": cfg.lookback_days}):
            # 1. Gather all entities and their related facts
            entity_facts = self._gather_entity_facts(memory, cfg.lookback_days)
            if not entity_facts:
                log.info("No entities with sufficient facts found for observation synthesis")
                return

            log.info(f"Found {len(entity_facts)} entities with facts")

            # 2. Filter to entities with enough facts
            eligible_entities = {
                entity_id: facts
                for entity_id, facts in entity_facts.items()
                if len(facts) >= cfg.min_facts_per_entity
            }

            if not eligible_entities:
                log.info(f"No entities have >= {cfg.min_facts_per_entity} facts")
                return

            log.info(f"{len(eligible_entities)} entities eligible for observation synthesis")

            # 3. Synthesize observations
            observations_created = 0
            for entity_id, facts in list(eligible_entities.items())[:cfg.max_observations_per_run]:
                # Check if observation already exists
                existing = self._find_existing_observation(memory, entity_id)
                if existing:
                    # Update existing observation with new facts
                    updated = self._update_observation(memory, existing, facts)
                    if updated:
                        log.debug(f"Updated observation for entity {entity_id}")
                    continue

                # Create new observation
                observation = self._synthesize_observation(entity_id, facts)
                if observation:
                    self._store_observation(memory, observation)
                    observations_created += 1
                    log.info(f"Created observation for entity {entity_id}: {observation.content[:50]}...")

            log.info(f"Observation synthesis complete: {observations_created} observations created")

    def _gather_entity_facts(self, memory, days: int) -> Dict[str, List[MemoryItem]]:
        """
        Gather facts about entities from semantic and episodic memories.
        
        Returns dict mapping entity_id -> list of facts about that entity.
        """
        entity_facts: Dict[str, List[MemoryItem]] = defaultdict(list)
        
        try:
            # Get semantic memories (facts)
            semantic_items = []
            if hasattr(memory, 'search'):
                results = memory.search(
                    query="*",
                    memory_types=['semantic'],
                    limit=500,
                )
                semantic_items = results if results else []
            
            # Extract entity references from each fact
            for item in semantic_items:
                # Check entities field
                entities = item.entities or []
                for entity in entities:
                    entity_id = entity.get('item_id') or entity.get('name', '')
                    if entity_id:
                        entity_facts[entity_id].append(item)
                
                # Check metadata for entity references
                subject = item.metadata.get('subject')
                if subject:
                    entity_facts[subject].append(item)
                
                # Check relations for entity references
                relations = item.relations or []
                for rel in relations:
                    source = rel.get('source_id') or rel.get('subject', '')
                    target = rel.get('target_id') or rel.get('object', '')
                    if source:
                        entity_facts[source].append(item)
                    if target:
                        entity_facts[target].append(item)
            
            return dict(entity_facts)
            
        except Exception as e:
            logger.error(f"Failed to gather entity facts: {e}")
            return {}

    def _find_existing_observation(self, memory, entity_id: str) -> Optional[MemoryItem]:
        """Check if an observation for this entity already exists."""
        try:
            if hasattr(memory, 'search'):
                results = memory.search(
                    query=entity_id,
                    memory_types=['observation'],
                    limit=5,
                )
                for result in (results or []):
                    if result.metadata.get('entity_id') == entity_id:
                        return result
            return None
        except Exception:
            return None

    def _update_observation(self, memory, existing: MemoryItem, new_facts: List[MemoryItem]) -> bool:
        """Update an existing observation with new facts."""
        try:
            obs_meta = ObservationMetadata.from_dict(existing.metadata)
            
            # Add new source facts
            existing_sources = set(obs_meta.source_facts)
            new_sources = [f.item_id for f in new_facts if f.item_id not in existing_sources]
            
            if not new_sources:
                return False  # No new facts to add
            
            for source_id in new_sources:
                obs_meta.add_source(source_id)
            
            # Update the item
            existing.metadata.update(obs_meta.to_dict())
            
            if hasattr(memory, 'update'):
                memory.update(existing)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update observation: {e}")
            return False

    def _synthesize_observation(self, entity_id: str, facts: List[MemoryItem]) -> Optional[MemoryItem]:
        """
        Synthesize an observation from multiple facts about an entity.
        """
        try:
            # Extract entity info from facts
            entity_name = self._extract_entity_name(entity_id, facts)
            entity_type = self._extract_entity_type(facts)
            
            # Combine fact contents into a summary
            fact_contents = [f.content for f in facts if f.content]
            if not fact_contents:
                return None
            
            # Simple synthesis: combine facts
            # TODO: Use LLM for more sophisticated synthesis
            summary = self._simple_synthesis(entity_name, fact_contents)
            
            # Detect aspects covered
            aspects = self._detect_aspects(facts)
            
            # Create observation metadata
            obs_meta = ObservationMetadata(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type,
                source_facts=[f.item_id for f in facts],
                aspects_covered=aspects,
                completeness=min(1.0, len(aspects) * 0.2),
            )
            
            observation_id = f"obs_{hashlib.sha256(entity_id.encode()).hexdigest()[:12]}"
            
            return MemoryItem(
                item_id=observation_id,
                content=summary,
                memory_type='observation',
                metadata=obs_meta.to_dict(),
            )
            
        except Exception as e:
            logger.error(f"Failed to synthesize observation for {entity_id}: {e}")
            return None

    def _extract_entity_name(self, entity_id: str, facts: List[MemoryItem]) -> str:
        """Extract the best name for an entity from facts."""
        # Check facts for name mentions
        for fact in facts:
            name = fact.metadata.get('entity_name') or fact.metadata.get('name')
            if name:
                return name
            
            # Check entities list
            for entity in (fact.entities or []):
                if entity.get('item_id') == entity_id or entity.get('name') == entity_id:
                    return entity.get('name', entity_id)
        
        return entity_id

    def _extract_entity_type(self, facts: List[MemoryItem]) -> Optional[str]:
        """Extract entity type from facts."""
        type_counts: Dict[str, int] = defaultdict(int)
        
        for fact in facts:
            entity_type = fact.metadata.get('entity_type') or fact.metadata.get('type')
            if entity_type:
                type_counts[entity_type] += 1
            
            for entity in (fact.entities or []):
                etype = entity.get('entity_type') or entity.get('type')
                if etype:
                    type_counts[etype] += 1
        
        if type_counts:
            return max(type_counts.keys(), key=lambda k: type_counts[k])
        return None

    def _simple_synthesis(self, entity_name: str, fact_contents: List[str]) -> str:
        """Simple synthesis by combining facts."""
        # Deduplicate and limit
        unique_facts = list(dict.fromkeys(fact_contents))[:10]
        
        if len(unique_facts) == 1:
            return f"About {entity_name}: {unique_facts[0]}"
        
        facts_text = "; ".join(unique_facts[:5])
        return f"Summary of {entity_name}: {facts_text}"

    def _detect_aspects(self, facts: List[MemoryItem]) -> List[str]:
        """Detect which aspects of an entity are covered by the facts."""
        aspects: Set[str] = set()
        
        aspect_keywords = {
            'career': ['work', 'job', 'company', 'role', 'position', 'employed'],
            'education': ['school', 'university', 'degree', 'study', 'graduate'],
            'location': ['live', 'city', 'country', 'address', 'located'],
            'preferences': ['prefer', 'like', 'favorite', 'enjoy', 'choose'],
            'relationships': ['friend', 'family', 'colleague', 'partner', 'knows'],
            'skills': ['skill', 'expert', 'proficient', 'knows how', 'can'],
            'interests': ['interest', 'hobby', 'passion', 'curious about'],
        }
        
        for fact in facts:
            content_lower = fact.content.lower() if fact.content else ""
            for aspect, keywords in aspect_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    aspects.add(aspect)
        
        return list(aspects)

    def _store_observation(self, memory, observation: MemoryItem):
        """Store the observation in memory."""
        try:
            if hasattr(memory, 'add'):
                memory.add(observation)
            elif hasattr(memory, 'vector_store') and hasattr(memory.vector_store, 'add'):
                memory.vector_store.add(observation)
        except Exception as e:
            logger.error(f"Failed to store observation: {e}")
