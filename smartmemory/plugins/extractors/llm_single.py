"""
Single-call LLM-based entity and relation extractor.

Extracts entities AND relations in a single LLM call to minimize latency.
This is optimized for speed over the two-step process which provides better
referential integrity but doubles API call overhead.

Use this when:
- Speed is critical (< 2s extraction target)
- Using fast models (Groq, Gemini Flash)
- Latency matters more than perfect precision

Use LLMExtractor (two-call) when:
- Maximum precision is required
- Relations must exactly match entity IDs
- Using slower models where extra call is negligible
"""

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.utils import get_config
from smartmemory.utils.cache import get_cache
from smartmemory.utils.llm import call_llm
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata
from smartmemory.utils.deduplication import deduplicate_entities

logger = logging.getLogger(__name__)

# Thread-local flag for cache hit signalling (CFS-1).
# Set True when extract() returns a cached result; consumed by LLMExtractStage._track_tokens().
_extract_thread_local = threading.local()


def was_last_extract_cached() -> bool:
    """Return True if the most recent ``extract()`` call on this thread was a cache hit.

    Consume-once: resets to False after reading.
    """
    hit = getattr(_extract_thread_local, "cache_hit", False)
    _extract_thread_local.cache_hit = False
    return hit


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> blocks from reasoning model responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# Unified extraction prompt that extracts entities AND relations in one shot
SINGLE_CALL_PROMPT = """Extract entities and relationships from this text for a knowledge graph.

TEXT:
{text}

ENTITY TYPES (use the most specific type):
- person: People, real or fictional
- organization: Companies, institutions, groups, platforms-as-companies (e.g. YouTube, Google)
- location: Countries, cities, geographic features, landmarks, buildings (e.g. Eiffel Tower, Los Gatos)
- event: Named events, conferences, historical occurrences (e.g. Macworld, World War II)
- product: Commercial/consumer products, physical devices (e.g. iPhone, Starship)
- work_of_art: Creative works — books, songs, movies, TV shows, symphonies (e.g. Stranger Things, Symphony No. 9)
- temporal: Dates, time periods, years (e.g. 1903, November 2022)
- concept: Abstract ideas, theories, academic fields (e.g. Physics, theory of relativity)
- technology: Programming languages, frameworks, protocols, software tools (e.g. Python, Docker, Swift, iOS)
- award: Prizes, honors (e.g. Nobel Prize)

RULES:
1. Split compound entities into atomic parts. "Nobel Prize in Physics" → two entities: "Nobel Prize" (award) and "Physics" (concept). "Los Gatos, California" → "Los Gatos" (location) and "California" (location).
2. Extract ALL relationships, including implicit ones. "Django is a Python framework" implies Django→framework_for→Python. "ChatGPT launched in November 2022" implies ChatGPT→launched_in→November 2022.
3. Extract hierarchical location relations. "Austin, Texas" implies Austin→located_in→Texas.
4. Use exact entity names from your entity list in relations (case-sensitive match).
5. Be comprehensive — extract every entity and every relationship between them.

Return JSON:
{{
  "entities": [
    {{"name": "exact name", "entity_type": "type", "confidence": 0.95}}
  ],
  "relations": [
    {{"subject": "entity1 name", "predicate": "relationship", "object": "entity2 name"}}
  ]
}}

Return ONLY valid JSON."""


# Full JSON schema for structured output (used with LM Studio and other local providers)
EXTRACTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "extraction_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["name", "entity_type", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entities", "relations"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class LLMSingleExtractorConfig(MemoryBaseModel):
    """Configuration for single-call LLM extractor."""

    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_base_url: Optional[str] = None  # For Groq, Gemini, etc.
    use_json_schema: bool = False  # Use full JSON schema (for local models that support structured output)
    temperature: float = 0.0
    max_tokens: int = 2000
    reasoning_effort: Optional[str] = None  # For o-series models


class LLMSingleExtractor(ExtractorPlugin):
    """
    Single-call LLM extractor for maximum speed.

    Extracts entities and relations in ONE API call instead of two.
    Trade-off: Slightly lower precision for ~50% faster extraction.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="llm_single",
            version="1.0.0",
            author="SmartMemory Team",
            description="Single-call entity and relation extraction for speed",
            plugin_type="extractor",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.2.7",
            tags=["ner", "relation-extraction", "llm", "fast"],
        )

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        config: Optional[LLMSingleExtractorConfig] = None,
    ):
        self.cfg = config or LLMSingleExtractorConfig()
        if model_name:
            self.cfg.model_name = model_name
        if api_base_url:
            self.cfg.api_base_url = api_base_url
        self._api_key_override = api_key

    def extract(self, text: str) -> dict:
        """
        Extract entities and relations from text in a single LLM call.

        Returns:
            dict with 'entities' (List[MemoryItem]) and 'relations' (List[dict])
        """
        plugin_name = self.metadata().name  # llm_single or groq
        with trace_span(f"pipeline.extract.{plugin_name}", {"text_length": len(text)}):
            return self._extract_impl(text)

    def _extract_impl(self, text: str) -> dict:
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        # Build cache key upfront
        cache_key = f"single_{self.cfg.model_name}_{text}"

        # Check cache
        cache = None
        try:
            cache = get_cache()
            cached = cache.get_entity_extraction(cache_key)
            if cached:
                logger.debug(f"Cache hit for: {text[:50]}...")
                _extract_thread_local.cache_hit = True
                return cached
        except Exception as e:
            logger.debug(f"Cache unavailable: {e}")
            cache = None

        # Mark as NOT a cache hit before calling LLM
        _extract_thread_local.cache_hit = False

        # Get API key
        api_key = self._get_api_key()

        # Build prompt
        user_prompt = SINGLE_CALL_PROMPT.format(text=text)

        # Choose response format: full JSON schema for local models, basic json_object for cloud
        resp_fmt: Dict[str, Any] = EXTRACTION_JSON_SCHEMA if self.cfg.use_json_schema else {"type": "json_object"}

        # Single LLM call
        parsed, raw = call_llm(
            model=self.cfg.model_name,
            system_prompt="You are an expert knowledge graph extractor. Extract entities and relationships accurately.",
            user_content=user_prompt,
            response_format=resp_fmt,
            max_output_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            api_key=api_key,
            api_base=self.cfg.api_base_url,
            reasoning_effort=self.cfg.reasoning_effort,
            config_section="extractor",
        )

        # Parse response
        data = parsed or {}
        if not data and raw and isinstance(raw, str):
            # Strip thinking tags first (reasoning models like DeepSeek-R1, QwQ, Phi-4-reasoning)
            cleaned = _strip_thinking_tags(raw) if "<think>" in raw else raw
            try:
                data = json.loads(cleaned)
            except Exception:
                # Strip markdown fences (common with Groq, Gemini)
                cleaned = _strip_markdown_fences(cleaned)
                try:
                    data = json.loads(cleaned)
                except Exception:
                    logger.warning(f"Failed to parse response: {raw[:200]}")
                    return {"entities": [], "relations": []}

        # Process entities
        raw_entities = data.get("entities", [])
        entities = self._process_entities(raw_entities)

        # Process relations
        raw_relations = data.get("relations", [])
        relations = self._process_relations(raw_relations, entities)

        result = {"entities": entities, "relations": relations}

        # Cache result (only if we got something useful)
        if cache and (result["entities"] or result["relations"]):
            try:
                cache.set_entity_extraction(cache_key, result)
            except Exception as e:
                logger.debug(f"Cache set failed: {e}")

        return result

    def _get_api_key(self) -> str:
        """Resolve API key from override, config, or environment."""
        if self._api_key_override:
            return self._api_key_override

        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            try:
                legacy = get_config("extractor")
                llm_cfg = legacy.get("llm") or {}
                api_key = llm_cfg.get("openai_api_key")
            except Exception:
                pass

        if not api_key:
            raise ValueError(f"No API key found. Set {self.cfg.api_key_env}.")
        return api_key

    def _process_entities(self, raw_entities: List[dict]) -> List[MemoryItem]:
        """Convert raw entity dicts to MemoryItems."""
        memory_items = []
        seen = set()

        for e in raw_entities:
            if not isinstance(e, dict):
                continue

            name = (e.get("name") or "").strip()
            etype = (e.get("entity_type") or "concept").strip().lower()

            if not name:
                continue

            key = f"{name.lower()}|{etype}"
            if key in seen:
                continue
            seen.add(key)

            ent_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            meta = {
                "name": name,
                "entity_type": etype,
                "confidence": e.get("confidence"),
            }

            # Include any extra attributes
            for k, v in e.items():
                if k not in ("name", "entity_type", "confidence") and v is not None:
                    meta[k] = v

            memory_items.append(MemoryItem(content=name, item_id=ent_id, memory_type="concept", metadata=meta))

        return deduplicate_entities(memory_items)

    def _process_relations(self, raw_relations: List[dict], entities: List[MemoryItem]) -> List[Dict[str, Any]]:
        """Convert raw relations to validated relation dicts."""
        # Build name -> ID map
        name_map = {e.metadata["name"].lower(): e.item_id for e in entities}

        valid_relations = []
        seen = set()

        for r in raw_relations:
            if not isinstance(r, dict):
                continue

            subj = (r.get("subject") or "").strip()
            pred = _normalize_predicate((r.get("predicate") or "").strip())
            obj = (r.get("object") or "").strip()

            if not (subj and pred and obj):
                continue

            # Resolve to entity IDs
            sid = name_map.get(subj.lower())
            oid = name_map.get(obj.lower())

            if sid and oid:
                key = f"{sid}|{pred}|{oid}"
                if key in seen:
                    continue
                seen.add(key)

                valid_relations.append({"source_id": sid, "target_id": oid, "relation_type": pred})

        return valid_relations


def _normalize_predicate(predicate: str) -> str:
    """Normalize predicate to be FalkorDB edge-label safe."""
    if not predicate:
        return "related_to"
    pred = predicate.lower()
    pred = re.sub(r"[^a-z0-9]+", "_", pred)
    pred = re.sub(r"_+", "_", pred)
    pred = pred.strip("_")
    if pred and pred[0].isdigit():
        pred = "_" + pred
    if not pred or not pred[0].isalpha():
        pred = "rel_" + pred
    if len(pred) > 63:
        pred = pred[:63]
    if not re.match(r"^[a-z][a-z0-9_]{0,62}$", pred):
        pred = "related_to"
    return pred


# Groq extractor class for registry lazy-loading (no-arg constructor)
class GroqExtractor(LLMSingleExtractor):
    """Default extractor using Groq Llama-3.3-70b (100% E-F1, 89.3% R-F1, 878ms)."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required for Groq extraction. "
                "Get a free key at https://console.groq.com"
            )
        super().__init__(
            model_name="llama-3.3-70b-versatile",
            api_key=api_key,
            api_base_url="https://api.groq.com/openai/v1",
        )


# Factory functions for common model configurations
def create_groq_extractor(model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
    """Create extractor using Groq's fast inference."""
    return LLMSingleExtractor(
        model_name=model,
        api_key=api_key or os.getenv("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
    )


def create_gemini_extractor(model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
    """Create extractor using Google's Gemini Flash."""
    return LLMSingleExtractor(
        model_name=model,
        api_key=api_key or os.getenv("GOOGLE_API_KEY"),
    )


def create_claude_extractor(model: str = "claude-3-5-haiku-latest", api_key: Optional[str] = None):
    """Create extractor using Anthropic's Haiku."""
    return LLMSingleExtractor(
        model_name=model,
        api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
    )
