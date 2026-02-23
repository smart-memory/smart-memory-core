"""
Registry module for ingestion pipeline stages.

This module handles registration and management of extractors, enrichers, adapters,
and converters for the ingestion pipeline. It consolidates all registry logic and
provides fallback mechanisms for extractor selection.
"""
import logging
from typing import Dict, List, Optional, Callable

from smartmemory.utils import get_config

logger = logging.getLogger(__name__)


class IngestionRegistry:
    """
    Centralized registry for ingestion pipeline stages.
    
    Manages registration of extractors, enrichers, adapters, and converters
    with automatic fallback mechanisms and performance instrumentation.
    """

    def __init__(self):
        # Store extractor classes for lazy loading
        self.extractor_registry: Dict[str, type] = {}  # Class types
        self.extractor_instances: Dict[str, any] = {}  # Instantiated extractors (cache)
        
        self.enricher_registry: Dict[str, Callable] = {}
        self.adapter_registry: Dict[str, Callable] = {}
        self.converter_registry: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default extractor classes for lazy loading.

        Uses direct module imports (not __init__.py) to avoid transitive dependency
        failures — e.g. HybridExtractor needing GLiNER at import time.

        Priority: groq (best quality+speed) > llm (OpenAI) > spaCy (no API key needed)
        """
        # Groq: 100% E-F1, 89.3% R-F1, 878ms — best overall extractor
        try:
            from smartmemory.plugins.extractors.llm_single import GroqExtractor
            self.register_extractor_class('groq', GroqExtractor)
        except ImportError as e:
            logger.warning(f"Failed to import Groq extractor: {e}")

        # OpenAI LLM extractor (two-call, higher precision)
        try:
            from smartmemory.plugins.extractors.llm import LLMExtractor
            self.register_extractor_class('llm', LLMExtractor)
        except ImportError as e:
            logger.warning(f"Failed to import LLM extractor: {e}")

        # Single-call LLM extractor (faster, configurable model)
        try:
            from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor
            self.register_extractor_class('llm_single', LLMSingleExtractor)
        except ImportError as e:
            logger.warning(f"Failed to import LLM single extractor: {e}")

        # Conversation-aware LLM extractor
        try:
            from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor
            self.register_extractor_class('conversation_aware_llm', ConversationAwareLLMExtractor)
        except ImportError as e:
            logger.warning(f"Failed to import conversation-aware LLM extractor: {e}")

        # spaCy: last-resort fallback — no API keys needed, works offline
        try:
            from smartmemory.plugins.extractors.spacy import SpacyExtractor
            self.register_extractor_class('spacy', SpacyExtractor)
        except ImportError as e:
            logger.warning(f"Failed to import spaCy extractor: {e}")

    def register_adapter(self, name: str, adapter_fn: Callable):
        """Register a new input adapter by name."""
        self.adapter_registry[name] = adapter_fn

    def register_converter(self, name: str, converter_fn: Callable):
        """Register a new type converter by name."""
        self.converter_registry[name] = converter_fn

    def register_extractor_class(self, name: str, extractor_class: type):
        """Register an extractor class for lazy instantiation."""
        self.extractor_registry[name] = extractor_class
        logger.debug(f"Registered extractor class: {name}")
    
    def register_extractor(self, name: str, extractor_fn: Callable):
        """Register a new entity/relation extractor by name."""
        self.extractor_instances[name] = extractor_fn

    def register_enricher(self, name: str, enricher_fn: Callable):
        """Register a new enrichment routine by name."""
        self.enricher_registry[name] = enricher_fn

    def get_fallback_order(self, primary: Optional[str] = None) -> List[str]:
        """
        Return a config-driven extractor fallback order, filtered to registered extractors.
        Defaults to ['groq', 'llm', 'llm_single', 'conversation_aware_llm', 'spacy'].
        Removes the primary extractor if provided.
        """
        try:
            cfg = get_config('extractor') or {}
        except Exception:
            cfg = {}

        order = cfg.get('fallback_order')
        if not order:
            order = ['groq', 'llm', 'llm_single', 'conversation_aware_llm', 'spacy']

        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for name in order:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        # Remove primary if provided
        if primary:
            deduped = [n for n in deduped if n != primary]

        # Keep only registered extractors (factories or instances)
        return [n for n in deduped if n in self.extractor_registry or n in self.extractor_instances]

    def select_extractor_for_context(self, conversation_context: Optional[dict] = None) -> Optional[str]:
        """
        Select the appropriate extractor based on context.

        If conversation_context is provided and has turn_history, selects 'conversation_aware_llm'.
        Otherwise falls back to select_default_extractor().

        Args:
            conversation_context: Optional conversation context dict

        Returns:
            Extractor name to use
        """
        # Check if we have meaningful conversation context
        if conversation_context:
            turn_history = conversation_context.get('turn_history', [])
            if turn_history and len(turn_history) > 0:
                # Use conversation-aware extractor if registered
                if self.is_extractor_registered('conversation_aware_llm'):
                    logger.debug("Selecting conversation_aware_llm extractor due to conversation context")
                    return 'conversation_aware_llm'

        # Fall back to default selection
        return self.select_default_extractor()

    def select_default_extractor(self) -> str:
        """
        Select the default extractor name using config and availability.

        Prefers extractor['default'] if registered, else the first available from
        the fallback order, else 'ontology' if registered, else any registered
        extractor, else 'spacy' as the absolute last resort.

        Raises:
            ValueError: If no extractors are registered at all.
        """
        try:
            cfg = get_config('extractor') or {}
        except Exception as e:
            logger.warning(f"Failed to load extractor config: {e}")
            cfg = {}

        default = cfg.get('default', 'groq')

        if default and (default in self.extractor_registry or default in self.extractor_instances):
            return default
        else:
            available = list(self.extractor_registry.keys())
            logger.warning(f"Configured default extractor '{default}' not registered. Available: {available}")

        # Try fallback order
        for name in self.get_fallback_order(primary=None):
            if name in self.extractor_registry or name in self.extractor_instances:
                return name

        # As a last resort, consider ontology or any registered
        if 'ontology' in self.extractor_registry:
            return 'ontology'

        first_available = next(iter(self.extractor_registry.keys()), None)
        if first_available:
            return first_available

        raise ValueError(
            "No extractors are registered. Ensure at least spaCy (en_core_web_sm) "
            "is installed, or set OPENAI_API_KEY for LLM-based extraction."
        )

    def get_extractor(self, name: str) -> Optional[Callable]:
        """Get an extractor by name, instantiating lazily if needed."""
        # Check if already instantiated
        if name in self.extractor_instances:
            return self.extractor_instances[name]
        
        # Check if we have a class registered
        extractor_class = self.extractor_registry.get(name)
        if extractor_class:
            try:
                logger.debug(f"Lazy loading extractor: {name}")
                instance = extractor_class()  # Instantiate the class
                self.extractor_instances[name] = instance
                return instance
            except Exception as e:
                logger.error(f"Failed to instantiate extractor '{name}': {e}")
                return None
        
        return None

    def get_enricher(self, name: str) -> Optional[Callable]:
        """Get an enricher by name."""
        return self.enricher_registry.get(name)

    def get_adapter(self, name: str) -> Optional[Callable]:
        """Get an adapter by name."""
        return self.adapter_registry.get(name)

    def get_converter(self, name: str) -> Optional[Callable]:
        """Get a converter by name."""
        return self.converter_registry.get(name)

    def list_extractors(self) -> List[str]:
        """List all registered extractor names (factories + instances)."""
        return list(set(self.extractor_registry.keys()) | set(self.extractor_instances.keys()))

    def list_enrichers(self) -> List[str]:
        """List all registered enricher names."""
        return list(self.enricher_registry.keys())

    def list_adapters(self) -> List[str]:
        """List all registered adapter names."""
        return list(self.adapter_registry.keys())

    def list_converters(self) -> List[str]:
        """List all registered converter names."""
        return list(self.converter_registry.keys())

    def is_extractor_registered(self, name: str) -> bool:
        """Check if an extractor is registered (factory or instance)."""
        return name in self.extractor_registry or name in self.extractor_instances

    def is_enricher_registered(self, name: str) -> bool:
        """Check if an enricher is registered."""
        return name in self.enricher_registry
