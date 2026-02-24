"""
LLM-based entity and relation extractor using OpenAI-compatible models.

This module extracts entities and SPO triples from text using a two-step process:
1. Extract entities (nodes)
2. Extract relations (edges) between those specific entities

This approach reduces hallucinations and ensures referential integrity.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal

from smartmemory.integration.llm.prompts.prompt_provider import get_prompt_value, apply_placeholders
from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.entity_types import ENTITY_TYPES
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.utils import get_config
from smartmemory.utils.cache import get_cache
from smartmemory.utils.llm import call_llm
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata
from smartmemory.utils.deduplication import deduplicate_entities

logger = logging.getLogger(__name__)


class EntityOut(BaseModel):
    name: str = Field(..., description="Canonical entity surface form")
    entity_type: str = "concept"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    attrs: Optional[Dict[str, Any]] = None


class TripleOut(BaseModel):
    subject: str
    predicate: str
    object: str
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    polarity: Optional[Literal["positive", "negative"]] = None


@dataclass
class LLMExtractorConfig(MemoryBaseModel):
    """Typed config for the LLM extractor."""
    model_name: str = "gpt-5-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1000
    
    # Prompt keys
    system_template_key: str = "plugins.extractors.llm.system_template"
    entity_extraction_template_key: str = "plugins.extractors.llm.entity_extraction_template"
    relation_extraction_template_key: str = "plugins.extractors.llm.relation_extraction_template"
    
    # Configuration
    include_entity_types_in_prompt: bool = False
    reasoning_effort: Optional[str] = "minimal"


class LLMExtractor(ExtractorPlugin):
    """
    LLM-based entity and relation extractor using a two-step process.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="llm",
            version="2.0.0",
            author="SmartMemory Team",
            description="Two-step Entity and Relation extraction using LLM",
            plugin_type="extractor",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "llm", "openai"]
        )
    
    def __init__(self, prompt_overrides: Optional[Dict[str, Any]] = None, config: Optional[LLMExtractorConfig] = None):
        self.overrides: Dict[str, Any] = dict(prompt_overrides or {})
        self.cfg = config or LLMExtractorConfig()
    
    def extract(self, text: str) -> dict:
        """
        Extract entities and relations from text using two-step LLM process.
        """
        with trace_span("pipeline.extract.llm", {"text_length": len(text)}):
            content = text

            # 1. Check Cache
            try:
                cache = get_cache()
                cached_result = cache.get_entity_extraction(content)
                if cached_result:
                    logger.debug(f"Cache hit: {content[:50]}...")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache unavailable: {e}")
                cache = None

            # 2. Setup API Key
            api_key = self._get_api_key()

            # 3. Step 1: Extract Entities
            entities = self._extract_entities(content, api_key)

            # 4. Step 2: Extract Relations (if entities found)
            relations = []
            if entities:
                relations = self._extract_relations(content, entities, api_key)

            # 5. Format Result
            extraction_result = {
                'entities': entities,
                'relations': relations,
            }

            # 6. Cache Result
            if cache:
                try:
                    cache.set_entity_extraction(content, extraction_result)
                except Exception as e:
                    logger.warning(f"Failed to cache: {e}")

            return extraction_result

    def _get_api_key(self) -> str:
        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            try:
                legacy = get_config('extractor')
                llm_cfg = legacy.get('llm') or {}
                api_key = llm_cfg.get("openai_api_key")
            except Exception:
                pass
        if not api_key:
            raise ValueError(f"No API key found. Set {self.cfg.api_key_env}.")
        return api_key

    def _extract_entities(self, text: str, api_key: str) -> List[MemoryItem]:
        """Step 1: Extract entities from text."""
        
        # Prepare Prompt
        system_template = self._get_template(self.cfg.system_template_key, "system_template")
        user_template = self._get_template(self.cfg.entity_extraction_template_key, "entity_extraction_template")
        
        # Default fallback if template key not found in prompts.json
        if not user_template:
            user_template = (
                "Extract all significant entities from the text below.\n"
                "Return a JSON object with key 'entities' containing a list of objects.\n"
                "Each object must have: 'name' (string), 'entity_type' (string), 'confidence' (float).\n\n"
                "TEXT:\n{{TEXT}}"
            )

        system_message = apply_placeholders(system_template, {})
        if self.cfg.include_entity_types_in_prompt and ENTITY_TYPES:
            types_str = ", ".join(sorted(set(t.strip().lower() for t in ENTITY_TYPES if isinstance(t, str) and t.strip())))
            system_message += f"\n\nALLOWED ENTITY TYPES:\n{types_str}"

        user_prompt = apply_placeholders(user_template, {"TEXT": text})

        # Call LLM
        parsed, raw = call_llm(
            model=self.cfg.model_name,
            system_prompt=system_message,
            user_content=user_prompt,
            response_model=None,
            response_format={"type": "json_object"},
            json_only_instruction="Return ONLY JSON with key 'entities'.",
            max_output_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            api_key=api_key,
            config_section="extractor",
        )

        # Parse Results
        data = parsed or {}
        if not data and raw and isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                pass

        raw_entities = data.get('entities', [])
        memory_items = []
        seen = set()

        for e in raw_entities:
            if not isinstance(e, dict): continue
            name = (e.get('name') or '').strip()
            etype = (e.get('entity_type') or 'concept').strip().lower()
            if not name: continue
            
            key = f"{name.lower()}|{etype}"
            if key in seen: continue
            seen.add(key)

            # Create MemoryItem
            ent_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            meta = {
                'name': name,
                'entity_type': etype,
                'confidence': e.get('confidence'),
                **(e.get('attrs') or {})
            }
            
            memory_items.append(MemoryItem(
                content=name,
                item_id=ent_id,
                memory_type='concept',
                metadata=meta
            ))

        return deduplicate_entities(memory_items)

    def _extract_relations(self, text: str, entities: List[MemoryItem], api_key: str) -> List[Dict[str, Any]]:
        """Step 2: Extract relations between provided entities."""
        
        # Prepare Entity List for Prompt
        entity_list_str = "\n".join([f"- {e.metadata['name']} ({e.metadata['entity_type']}) [ID: {e.item_id}]" for e in entities])
        
        # Prepare Prompt
        system_template = self._get_template(self.cfg.system_template_key, "system_template")
        user_template = self._get_template(self.cfg.relation_extraction_template_key, "relation_extraction_template")
        
        # Default fallback
        if not user_template:
            user_template = (
                "Identify relationships between the following entities based on the text.\n"
                "Entities:\n{{ENTITIES}}\n\n"
                "Text:\n{{TEXT}}\n\n"
                "Return JSON with key 'relations' containing a list of objects.\n"
                "Each object: {subject: str, predicate: str, object: str}."
            )

        system_message = apply_placeholders(system_template, {})
        user_prompt = apply_placeholders(user_template, {
            "TEXT": text,
            "ENTITIES": entity_list_str
        })

        # Call LLM
        parsed, raw = call_llm(
            model=self.cfg.model_name,
            system_prompt=system_message,
            user_content=user_prompt,
            response_model=None,
            response_format={"type": "json_object"},
            json_only_instruction="Return ONLY JSON with key 'relations'. Use exact entity names.",
            max_output_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            api_key=api_key,
            config_section="extractor",
        )

        # Parse Results
        data = parsed or {}
        if not data and raw and isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                pass

        raw_relations = data.get('relations', [])
        valid_relations = []
        
        # Map names to IDs for resolution
        name_map = {e.metadata['name'].lower(): e.item_id for e in entities}

        for r in raw_relations:
            if not isinstance(r, dict): continue
            subj = (r.get('subject') or '').strip()
            pred = _normalize_predicate((r.get('predicate') or '').strip())
            obj = (r.get('object') or '').strip()
            
            if not (subj and pred and obj): continue
            
            # Resolve IDs
            sid = name_map.get(subj.lower())
            oid = name_map.get(obj.lower())
            
            if sid and oid:
                valid_relations.append({
                    'source_id': sid,
                    'target_id': oid,
                    'relation_type': pred
                })

        return valid_relations

    def _get_template(self, key: str, override_key: str) -> Optional[str]:
        return (self.overrides.get(override_key) if self.overrides else None) or get_prompt_value(key)


def _normalize_predicate(predicate: str) -> str:
    """Normalize predicate to be FalkorDB edge-label safe."""
    if not predicate:
        return "unknown"
    pred = predicate.lower()
    pred = re.sub(r'[^a-z0-9]+', '_', pred)
    pred = re.sub(r'_+', '_', pred)
    pred = pred.strip('_')
    if pred and pred[0].isdigit():
        pred = '_' + pred
    if not pred or not pred[0].isalpha():
        pred = 'rel_' + pred
    if len(pred) > 63:
        pred = pred[:63]
    if not re.match(r'^[a-z][a-z0-9_]{0,62}$', pred):
        pred = 'relation'
    return pred
