"""
Conversation-Aware LLM Extractor.

Enhances entity and relation extraction with conversation context, enabling:
- Context-enriched extraction prompts
- Coreference resolution (pronouns, references)
- Speaker-entity relationship extraction
- Entity continuity across conversation turns

conversation-aware knowledge graph extraction.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from smartmemory.plugins.extractors.llm import LLMExtractor, LLMExtractorConfig, EntityOut, TripleOut
from smartmemory.plugins.base import PluginMetadata
from smartmemory.conversation.context import ConversationContext
from smartmemory.observability.tracing import trace_span
from smartmemory.utils.llm import call_llm

logger = logging.getLogger(__name__)


class ConversationAwareLLMExtractor(LLMExtractor):
    """
    LLM extractor enhanced with conversation context awareness.
    
    Extends LLMExtractor to:
    1. Build context from conversation history
    2. Enrich extraction prompts with context
    3. Resolve coreferences using conversation entities
    4. Extract speaker-entity relationships
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="conversation_aware_llm",
            version="1.0.0",
            author="SmartMemory Team",
            description="Conversation-aware entity and relation extraction using LLM",
            plugin_type="extractor",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "llm", "conversation", "coreference"]
        )
    
    def __init__(
        self, 
        prompt_overrides: Optional[Dict[str, Any]] = None, 
        config: Optional[LLMExtractorConfig] = None,
        max_history_turns: int = 5,
        resolve_coreferences: bool = True,
        extract_speaker_relations: bool = True
    ):
        super().__init__(prompt_overrides, config)
        self.max_history_turns = max_history_turns
        self.resolve_coreferences = resolve_coreferences
        self.extract_speaker_relations = extract_speaker_relations
        
        # Common pronouns and references to resolve
        self.pronouns = {
            'it', 'its', 'that', 'this', 'these', 'those',
            'he', 'him', 'his', 'she', 'her', 'hers',
            'they', 'them', 'their', 'theirs'
        }
    
    def extract(
        self,
        text: str,
        conversation_context: Optional[ConversationContext] = None
    ) -> dict:
        """
        Extract entities and relations with conversation awareness.

        Args:
            text: Current message text
            conversation_context: Conversation context with history, entities, and coreference chains

        Returns:
            dict with 'entities', 'relations', and optionally 'speaker_relations'
        """
        with trace_span("pipeline.extract.conversation_aware_llm", {"text_length": len(text)}):
            return self._extract_impl(text, conversation_context)

    def _extract_impl(
        self,
        text: str,
        conversation_context: Optional[ConversationContext] = None,
    ) -> dict:
        # Check if we have any useful context (coreference chains count as useful context)
        has_useful_context = (
            conversation_context and (
                conversation_context.turn_history or
                conversation_context.entities or
                conversation_context.coreference_chains
            )
        )

        # If no useful context, fall back to base extraction
        if not has_useful_context:
            return super().extract(text)

        # Build context text from conversation history and coreference chains
        context_text = self._build_context_text(conversation_context)

        # Extract entities with context
        entities = self._extract_entities_with_context(text, context_text)

        # Resolve coreferences if enabled
        # Use fastcoref chains (high quality) or fall back to heuristics with entity history
        if self.resolve_coreferences:
            has_coref_data = (
                conversation_context.coreference_chains or
                conversation_context.entities
            )
            if has_coref_data:
                entities = self._resolve_coreferences(entities, conversation_context)

        # Extract relations with context
        relations = []
        if entities:
            relations = self._extract_relations_with_context(text, entities, context_text)

        result = {
            'entities': entities,
            'relations': relations
        }

        # Extract speaker relations if enabled
        if self.extract_speaker_relations and conversation_context.turn_history:
            speaker_relations = self._extract_speaker_relations(
                text,
                entities,
                conversation_context
            )
            if speaker_relations:
                result['speaker_relations'] = speaker_relations

        return result
    
    def _build_context_text(self, conversation_context: ConversationContext) -> str:
        """
        Build context text from conversation history.

        Args:
            conversation_context: Conversation context with history

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add recent conversation history
        if conversation_context.turn_history:
            history_turns = conversation_context.turn_history[-self.max_history_turns:]
            if history_turns:
                context_parts.append("Recent conversation:")
                for turn in history_turns:
                    role = turn.get('role', 'user').capitalize()
                    content = turn.get('content', '')
                    context_parts.append(f"[{role}]: {content}")

        # Add coreference chains from fastcoref preprocessing
        if conversation_context.coreference_chains:
            context_parts.append("\nCoreference resolutions (pronouns mapped to entities):")
            for chain in conversation_context.coreference_chains:
                head = chain.get('head', '')
                mentions = chain.get('mentions', [])
                other_mentions = [m for m in mentions if m != head]
                if other_mentions:
                    context_parts.append(f"- {head} (also referred to as: {', '.join(other_mentions)})")

        # Add known entities
        if conversation_context.entities:
            context_parts.append("\nKnown entities in conversation:")
            for entity in conversation_context.entities[-10:]:  # Last 10 entities
                name = entity.get('name', '')
                entity_type = entity.get('type', 'concept')
                context_parts.append(f"- {name} ({entity_type})")

        # Add topics if available
        if conversation_context.topics:
            context_parts.append(f"\nConversation topics: {', '.join(conversation_context.topics)}")

        return "\n".join(context_parts)
    
    def _extract_entities_with_context(
        self,
        text: str,
        context_text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract entities using conversation context.

        Args:
            text: Current message text
            context_text: Formatted conversation context

        Returns:
            List of extracted entities
        """
        api_key = self._get_api_key()

        # Build context-aware prompt
        system_prompt = """You are extracting entities from a conversation.
Use the conversation context to:
1. Use coreference resolutions provided (if any) to map pronouns to actual entities
2. Maintain entity consistency across turns
3. Identify new entities mentioned

IMPORTANT: If coreference resolutions are provided, use the HEAD entity name instead of pronouns.
For example, if "The company" maps to "Apple Inc.", extract "Apple Inc." not "The company".

Return entities in JSON format with 'name', 'entity_type', and optional 'confidence'."""

        user_prompt = f"""CONVERSATION CONTEXT:
{context_text}

CURRENT MESSAGE:
{text}

Extract all entities from the CURRENT MESSAGE. Use the coreference resolutions above to resolve pronouns and references to their actual entity names.

Return a JSON object with an 'entities' array."""
        
        # Call LLM
        try:
            parsed, response = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.cfg.model_name,
                api_key=api_key,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
                response_format={"type": "json_object"}
            )
            
            parsed_result = parsed
            if parsed and isinstance(parsed, dict):
                raw_entities = parsed.get('entities', [])
                return self._normalize_entities(raw_entities)
            
        except Exception as e:
            logger.error(f"Context-aware entity extraction failed: {e}")
        
        return []
    
    def _extract_relations_with_context(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        context_text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract relations using conversation context.
        
        Args:
            text: Current message text
            entities: Extracted entities
            context_text: Formatted conversation context
            
        Returns:
            List of extracted relations
        """
        if not entities:
            return []
        
        api_key = self._get_api_key()
        
        # Build entity list for prompt
        entity_names = [e.get('name', '') for e in entities]
        entity_list = ", ".join(entity_names)
        
        system_prompt = """You are extracting relationships from a conversation.
Focus on relationships between the entities provided.
Consider the conversation context to identify implicit relationships.

Return relationships as subject-predicate-object triples in JSON format."""
        
        user_prompt = f"""CONVERSATION CONTEXT:
{context_text}

CURRENT MESSAGE:
{text}

ENTITIES IN MESSAGE:
{entity_list}

Extract relationships between these entities. Return a JSON object with a 'relations' array.
Each relation should have 'subject', 'predicate', and 'object'."""
        
        try:
            parsed, response = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.cfg.model_name,
                api_key=api_key,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
                response_format={"type": "json_object"}
            )
            
            if parsed and isinstance(parsed, dict):
                raw_relations = parsed.get('relations', [])
                return self._normalize_relations(raw_relations)
            
        except Exception as e:
            logger.error(f"Context-aware relation extraction failed: {e}")
        
        return []
    
    def _resolve_coreferences(
        self,
        entities: List[Dict[str, Any]],
        conversation_context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """
        Resolve pronouns and references using coreference chains and context.

        Uses fastcoref chains when available (higher quality), falls back to
        heuristic resolution using conversation entity history.

        Args:
            entities: Extracted entities (may contain pronouns)
            conversation_context: Conversation context with known entities and coref chains

        Returns:
            Entities with coreferences resolved
        """
        resolved = []

        # Build lookup from fastcoref chains: mention -> head entity
        coref_lookup = {}
        if conversation_context.coreference_chains:
            for chain in conversation_context.coreference_chains:
                head = chain.get('head', '')
                for mention in chain.get('mentions', []):
                    if mention.lower() != head.lower():
                        coref_lookup[mention.lower()] = head

        for entity in entities:
            name = entity.get('name', '').strip()
            name_lower = name.lower()

            # First try fastcoref chain lookup (highest quality)
            if name_lower in coref_lookup:
                resolved_name = coref_lookup[name_lower]
                resolved.append({
                    'name': resolved_name,
                    'entity_type': entity.get('entity_type', 'concept'),
                    'confidence': 0.9,  # High confidence from fastcoref
                    'resolved_from': name,
                    'resolution_source': 'fastcoref'
                })
            # Then check if it's a pronoun or reference needing heuristic resolution
            elif name_lower in self.pronouns or self._is_demonstrative_reference(name_lower):
                resolved_entity = self._resolve_from_context(entity, conversation_context)
                if resolved_entity:
                    resolved_entity['resolution_source'] = 'heuristic'
                    resolved.append(resolved_entity)
                else:
                    resolved.append(entity)
            else:
                resolved.append(entity)

        return resolved
    
    def _is_demonstrative_reference(self, text: str) -> bool:
        """Check if text is a demonstrative reference like 'that algorithm'."""
        demonstratives = ['that', 'this', 'these', 'those']
        words = text.split()
        return len(words) > 1 and words[0] in demonstratives
    
    def _resolve_from_context(
        self,
        entity: Dict[str, Any],
        conversation_context: ConversationContext
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a pronoun/reference to an entity from context.
        
        Args:
            entity: Entity to resolve (contains pronoun/reference)
            conversation_context: Conversation context
            
        Returns:
            Resolved entity or None
        """
        name = entity.get('name', '').lower()
        entity_type = entity.get('entity_type', '')
        
        # Simple heuristic: return most recent entity of compatible type
        # In production, this could use more sophisticated coreference resolution
        
        if not conversation_context.entities:
            return None
        
        # For pronouns like "it", return last mentioned non-person entity
        if name in ['it', 'its', 'that', 'this']:
            for ctx_entity in reversed(conversation_context.entities):
                ctx_type = ctx_entity.get('type', '').lower()
                if ctx_type not in ['person', 'user', 'assistant']:
                    return {
                        'name': ctx_entity.get('name', ''),
                        'entity_type': ctx_entity.get('type', 'concept'),
                        'confidence': 0.7,  # Lower confidence for resolved entities
                        'resolved_from': name
                    }
        
        # For person pronouns, return last mentioned person
        if name in ['he', 'him', 'his', 'she', 'her', 'hers']:
            for ctx_entity in reversed(conversation_context.entities):
                ctx_type = ctx_entity.get('type', '').lower()
                if ctx_type in ['person', 'user']:
                    return {
                        'name': ctx_entity.get('name', ''),
                        'entity_type': 'person',
                        'confidence': 0.7,
                        'resolved_from': name
                    }
        
        # For demonstrative references like "that algorithm", match by type
        if self._is_demonstrative_reference(name):
            words = name.split()
            reference_type = words[-1]  # e.g., "algorithm" from "that algorithm"
            
            for ctx_entity in reversed(conversation_context.entities):
                ctx_name = ctx_entity.get('name', '').lower()
                if reference_type in ctx_name:
                    return {
                        'name': ctx_entity.get('name', ''),
                        'entity_type': ctx_entity.get('type', 'concept'),
                        'confidence': 0.6,
                        'resolved_from': name
                    }
        
        return None
    
    def _extract_speaker_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        conversation_context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between speaker and entities.
        
        Args:
            text: Current message text
            entities: Extracted entities
            conversation_context: Conversation context
            
        Returns:
            List of speaker-entity relations
        """
        if not entities or not conversation_context.turn_history:
            return []
        
        # Get current speaker role
        current_turn = conversation_context.turn_history[-1]
        speaker_role = current_turn.get('role', 'user').capitalize()
        
        api_key = self._get_api_key()
        
        # Build entity list
        entity_names = [e.get('name', '') for e in entities]
        entity_list = ", ".join(entity_names)
        
        system_prompt = """You are identifying relationships between a speaker and entities they mention.

Common speaker-entity relationships:
- ASKS_ABOUT: Speaker is asking about an entity
- EXPLAINS: Speaker is explaining an entity
- MENTIONS: Speaker mentions an entity
- INTERESTED_IN: Speaker shows interest in an entity
- DISCUSSES: Speaker is discussing an entity

Return relationships in JSON format."""
        
        user_prompt = f"""SPEAKER: {speaker_role}
MESSAGE: {text}
ENTITIES: {entity_list}

Identify relationships between the speaker ({speaker_role}) and the entities mentioned.
Return a JSON object with a 'speaker_relations' array.
Each relation should have 'subject' (speaker role), 'predicate' (relationship type), and 'object' (entity name)."""
        
        try:
            parsed, response = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.cfg.model_name,
                api_key=api_key,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
                response_format={"type": "json_object"}
            )
            
            if parsed and isinstance(parsed, dict):
                return parsed.get('speaker_relations', [])
            
        except Exception as e:
            logger.error(f"Speaker relation extraction failed: {e}")
        
        return []
    
    def _normalize_entities(self, raw_entities: List[Any]) -> List[Dict[str, Any]]:
        """Normalize raw entity data to standard format."""
        normalized = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            name = (e.get('name') or '').strip()
            if not name:
                continue
            normalized.append({
                'name': name,
                'entity_type': (e.get('entity_type') or 'concept').strip().lower(),
                'confidence': e.get('confidence', 0.8)
            })
        return normalized
    
    def _normalize_relations(self, raw_relations: List[Any]) -> List[Dict[str, Any]]:
        """Normalize raw relation data to standard format."""
        normalized = []
        for r in raw_relations:
            if not isinstance(r, dict):
                continue
            subject = (r.get('subject') or '').strip()
            predicate = (r.get('predicate') or '').strip()
            obj = (r.get('object') or '').strip()
            if subject and predicate and obj:
                normalized.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj
                })
        return normalized
