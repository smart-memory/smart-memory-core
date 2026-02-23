"""ExtractorPipeline component for componentized memory ingestion pipeline.

.. deprecated::
    Use ``smartmemory.pipeline.stages`` (SimplifyStage, EntityRulerStage,
    LLMExtractStage, OntologyConstrainStage) via PipelineRunner instead.
    This class will be removed in a future version.

Handles entity/relation extraction with fallback chain and multiple extractor support.
Includes automatic chunking for large texts.
"""
import logging
import warnings
from typing import Dict, Any, Optional, List

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import ExtractionConfig
from smartmemory.memory.pipeline.state import ClassificationState
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.utils import get_config
from smartmemory.utils.pipeline_utils import create_error_result

logger = logging.getLogger(__name__)

# Default chunk size for large text processing (characters)
DEFAULT_CHUNK_SIZE = 8000
DEFAULT_CHUNK_OVERLAP = 200


class ExtractorPipeline(PipelineComponent[ExtractionConfig]):
    """Entity and relation extraction with fallback chain.

    .. deprecated::
        Use ``smartmemory.pipeline.stages`` (SimplifyStage, EntityRulerStage,
        LLMExtractStage, OntologyConstrainStage) via ``PipelineRunner`` instead.
        This class will be removed in a future version.

    Supports multiple extractors (llm, spacy, gliner, relik, ontology) with automatic fallback.
    """

    def __init__(self, memory=None):
        warnings.warn(
            "ExtractorPipeline is deprecated. Use smartmemory.pipeline.stages via PipelineRunner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._logger = logging.getLogger(__name__)
        self.memory = memory
        # Registry maps extractor name -> extractor class
        self.extractor_registry: Dict[str, Any] = {}
        # Cache for resolved extractors (name -> callable)
        self._extractor_cache: Dict[str, Any] = {}
        self._last_validation_errors: List[str] = []
        self._register_default_extractors()

    def _register_default_extractors(self):
        """Register default extractors from plugin system."""
        # Use the plugin system to get all registered extractors
        from smartmemory.plugins.manager import get_plugin_manager
        
        plugin_manager = get_plugin_manager()
        registry = plugin_manager.registry
        
        # Get all extractor classes from the registry
        for extractor_name in registry.list_plugins('extractor'):
            extractor_class = registry.get_extractor(extractor_name)
            if extractor_class:
                # Register the class directly
                self.extractor_registry[extractor_name] = extractor_class
        
        # Log registered extractors
        self._logger.info("Registered extractors from plugin system: %s", sorted(self.extractor_registry.keys()))

    def register_extractor(self, name: str, extractor_class):
        """Register a new extractor class by name."""
        self.extractor_registry[name] = extractor_class

    def _resolve_extractor(self, name: str):
        """Resolve an extractor by name, instantiating plugin classes on-demand."""
        if name in self._extractor_cache:
            return self._extractor_cache[name]
        
        val = self.extractor_registry.get(name)
        if val is None:
            return None
        
        # Check if it's a plugin class (has metadata method)
        if hasattr(val, 'metadata') and callable(getattr(val, 'metadata', None)):
            try:
                # It's a plugin class - instantiate it and create a callable
                instance = val()
                extractor_callable = lambda item: instance.extract(getattr(item, 'content', str(item)))
                self._extractor_cache[name] = extractor_callable
                return extractor_callable
            except Exception as e:
                self._logger.error("Failed to instantiate extractor class '%s': %s", name, e)
                return None
        
        # If it's already a callable, use it directly
        if callable(val):
            self._extractor_cache[name] = val
            return val
        
        self._logger.error("Extractor '%s' is neither a plugin class nor a callable", name)
        return None

    def _get_fallback_order(self, primary: str = None) -> List[str]:
        """Get fallback order for extractors, excluding primary"""
        # Default fallback order: LLM first, then local
        default_fallback = ['llm']
        
        # Read from config but be resilient to ConfigDict fail-fast semantics
        try:
            cfg = get_config('extractor')  # ConfigDict
            try:
                fallback_order = cfg['fallback_order']  # type: ignore[index]
            except KeyError:
                fallback_order = default_fallback
        except Exception:
            fallback_order = default_fallback

        if primary and isinstance(fallback_order, list):
            try:
                return [x for x in fallback_order if x != primary]
            except Exception:
                return []
        return fallback_order if isinstance(fallback_order, list) else default_fallback

    def validate_config(self, config: ExtractionConfig) -> bool:
        """Validate ExtractorPipeline configuration using typed config"""
        try:
            errors: List[str] = []

            # extractor_name: optional string
            extractor_name = getattr(config, 'extractor_name', None)
            if extractor_name is not None and not isinstance(extractor_name, str):
                errors.append("extractor_name must be a string if provided")

            # max_entities: positive int
            max_entities = getattr(config, 'max_entities', None)
            if max_entities is not None and (not isinstance(max_entities, int) or max_entities <= 0):
                errors.append("max_entities must be a positive integer")

            # enable_relations: bool if provided
            er = getattr(config, 'enable_relations', None)
            if er is not None and not isinstance(er, bool):
                errors.append("enable_relations must be a boolean")

            if errors:
                try:
                    self._logger.warning("ExtractorPipeline config validation failed: %s", "; ".join(errors))
                except Exception:
                    pass
                self._last_validation_errors = errors
                return False

            self._last_validation_errors = []
            return True
        except Exception:
            return False

    def _select_default_extractor(self) -> Optional[str]:
        """Select the default extractor using config and availability"""
        try:
            cfg = get_config('extractor') or {}
        except Exception:
            cfg = {}

        default = cfg.get('default', 'llm')
        if default in self.extractor_registry:
            return default

        # Use first available from fallback order
        for name in self._get_fallback_order(primary=None):
            if name in self.extractor_registry:
                return name

        # Last resort - ontology or any registered
        if 'ontology' in self.extractor_registry:
            return 'ontology'

        return next(iter(self.extractor_registry.keys()), None)

    def _extract_with_ontology(self, memory_item, fallback_order: List[str]) -> Dict[str, Any]:
        """Handle ontology-aware extraction with fallback"""
        try:
            extractor = self._resolve_extractor('ontology')
            if extractor is None:
                raise ImportError("ontology extractor not available")
            # Pull optional knobs from the last validated config if available
            # (Run() sets config; here we grab from a private attr if present, else default)
            cfg = getattr(self, "_current_cfg", None)
            kwargs = {}
            if cfg is not None:
                try:
                    if getattr(cfg, 'ontology_enabled', None) is not None:
                        kwargs['ontology_enabled'] = bool(cfg.ontology_enabled)
                    if getattr(cfg, 'ontology_constraints', None):
                        kwargs['ontology_constraints'] = list(cfg.ontology_constraints)
                    if getattr(cfg, 'model', None):
                        kwargs['model'] = str(cfg.model)
                    if getattr(cfg, 'temperature', None) is not None:
                        kwargs['temperature'] = float(cfg.temperature)
                    if getattr(cfg, 'max_tokens', None) is not None:
                        kwargs['max_tokens'] = int(cfg.max_tokens)
                    if getattr(cfg, 'reasoning_effort', None):
                        kwargs['reasoning_effort'] = str(cfg.reasoning_effort)
                    if getattr(cfg, 'max_reasoning_tokens', None) is not None:
                        kwargs['max_reasoning_tokens'] = int(cfg.max_reasoning_tokens)
                except Exception:
                    pass
            result = extractor.extract_entities_and_relations(
                memory_item.content,
                **kwargs,
            )

            # Convert OntologyNode objects to MemoryItem objects
            entities = []
            for node in result['entities']:
                memory_item_converted = node.to_memory_item()
                entities.append(memory_item_converted)

            relations = result['relations']

            # If ontology returns nothing, fall back
            if not entities and not relations:
                return self._extract_with_fallback(memory_item, fallback_order)

            return {
                'entities': entities,
                'relations': relations
            }

        except Exception:
            # On failure, fall back to configured extractors
            return self._extract_with_fallback(memory_item, fallback_order)

    def _extract_with_fallback(self, memory_item, fallback_order: List[str]) -> Dict[str, Any]:
        """Try extractors in fallback order until one succeeds.
        
        Automatically uses chunking for large texts (> DEFAULT_CHUNK_SIZE).
        """
        # Check if text is large enough to require chunking
        content = getattr(memory_item, 'content', str(memory_item))
        if len(content) > DEFAULT_CHUNK_SIZE:
            return self._extract_with_chunking(memory_item, fallback_order)
        
        for fb_name in fallback_order:
            fb = self._resolve_extractor(fb_name)
            if not fb:
                continue

            try:
                fb_res = fb(memory_item)
            except Exception as e:
                logger.warning(f"Extractor {fb_name} failed: {e}")
                continue

            # Normalize different output formats
            if isinstance(fb_res, dict):
                fb_entities = fb_res.get('entities', [])
                fb_relations = fb_res.get('relations', [])
            elif isinstance(fb_res, tuple) and len(fb_res) == 3:
                _it, fb_entities, fb_relations = fb_res
            else:
                fb_entities, fb_relations = [], []

            # Convert entities to MemoryItem objects for consistency
            converted_entities = []
            for i, ent in enumerate(fb_entities):
                if isinstance(ent, MemoryItem):
                    converted_entities.append(ent)
                elif isinstance(ent, dict):
                    entity_content = ent.get('name', ent.get('text', f'entity_{i}'))
                    entity_type = ent.get('type', ent.get('entity_type', ent.get('label', 'entity')))
                    metadata = {
                        'name': ent.get('name', entity_content),
                        'entity_type': entity_type,
                        'confidence': ent.get('confidence', 1.0),
                        'source': 'legacy_extractor'
                    }
                    # Copy additional fields to metadata
                    for k, v in ent.items():
                        if k not in ['name', 'text', 'type', 'label', 'confidence']:
                            metadata[k] = v
                    converted_entities.append(MemoryItem(
                        content=entity_content,
                        memory_type=entity_type,
                        metadata=metadata
                    ))
                elif isinstance(ent, str):
                    converted_entities.append(MemoryItem(
                        content=ent,
                        memory_type='entity',
                        metadata={'name': ent, 'confidence': 1.0, 'source': 'legacy_extractor'}
                    ))
                else:
                    ent_str = str(ent)
                    converted_entities.append(MemoryItem(
                        content=ent_str,
                        memory_type='entity',
                        metadata={'name': ent_str, 'confidence': 1.0, 'source': 'legacy_extractor'}
                    ))

            # Return if we got any results
            if converted_entities or fb_relations:
                return {
                    'entities': converted_entities,
                    'relations': fb_relations
                }

        # Nothing worked - return empty results
        return {
            'entities': [],
            'relations': []
        }

    def _extract_with_chunking(self, memory_item, fallback_order: List[str]) -> Dict[str, Any]:
        """
        Extract from large text using chunking with parallel processing.
        
        chunk processing approach.
        """
        from smartmemory.utils.chunking import chunk_text, _extract_parallel
        from smartmemory.memory.pipeline.stages.clustering import aggregate_graphs
        
        content = getattr(memory_item, 'content', str(memory_item))
        
        # Chunk the text
        chunks = chunk_text(
            content, 
            chunk_size=DEFAULT_CHUNK_SIZE, 
            overlap=DEFAULT_CHUNK_OVERLAP,
            strategy="sentence"
        )
        
        logger.info(f"Large text ({len(content)} chars) split into {len(chunks)} chunks")
        
        # Create extraction function for chunks
        def extract_chunk(chunk_text: str) -> Dict[str, Any]:
            # Create a temporary memory item for the chunk
            # Metadata (including any auth context) is copied from parent
            chunk_item = MemoryItem(
                content=chunk_text,
                memory_type=getattr(memory_item, 'memory_type', 'semantic'),
                metadata=dict(getattr(memory_item, 'metadata', {}))
            )
            
            # Try extractors in order
            for fb_name in fallback_order:
                fb = self._resolve_extractor(fb_name)
                if not fb:
                    continue
                try:
                    fb_res = fb(chunk_item)
                    if isinstance(fb_res, dict):
                        return fb_res
                    elif isinstance(fb_res, tuple) and len(fb_res) == 3:
                        _, entities, relations = fb_res
                        return {'entities': entities, 'relations': relations}
                except Exception as e:
                    logger.debug(f"Chunk extraction with {fb_name} failed: {e}")
                    continue
            return {'entities': [], 'relations': []}
        
        # Extract from chunks in parallel
        if len(chunks) > 1:
            chunk_results = _extract_parallel(extract_chunk, chunks, max_workers=4)
        else:
            chunk_results = [extract_chunk(chunks[0])] if chunks else []
        
        if not chunk_results:
            return {'entities': [], 'relations': []}
        
        # Aggregate results with deduplication
        aggregated = aggregate_graphs(chunk_results, cluster=True)
        
        # Convert entities to MemoryItem format
        converted_entities = []
        for ent in aggregated.get('entities', []):
            if isinstance(ent, MemoryItem):
                converted_entities.append(ent)
            elif isinstance(ent, dict):
                entity_content = ent.get('name', ent.get('text', 'entity'))
                entity_type = ent.get('type', ent.get('entity_type', 'entity'))
                metadata = {
                    'name': ent.get('name', entity_content),
                    'entity_type': entity_type,
                    'confidence': ent.get('confidence', 1.0),
                    'source': 'chunked_extraction'
                }
                for k, v in ent.items():
                    if k not in ['name', 'text', 'type', 'entity_type', 'confidence']:
                        metadata[k] = v
                converted_entities.append(MemoryItem(
                    content=entity_content,
                    memory_type=entity_type,
                    metadata=metadata
                ))
        
        logger.info(f"Chunked extraction: {len(converted_entities)} entities, "
                   f"{len(aggregated.get('relations', []))} relations from {len(chunks)} chunks")
        
        return {
            'entities': converted_entities,
            'relations': aggregated.get('relations', []),
            'chunk_count': len(chunks),
            'entity_clusters': aggregated.get('entity_clusters', {}),
            'edge_clusters': aggregated.get('edge_clusters', {})
        }

    def run(self, classification_state: ClassificationState, config: ExtractionConfig) -> ComponentResult:
        """
        Execute ExtractorPipeline with given classification state and configuration.
        
        Args:
            classification_state: ClassificationState from previous stage
            config: Extraction configuration dict with optional 'extractor_name'
        
        Returns:
            ComponentResult with entities, relations, and extraction metadata
        """
        try:
            # Accept either ClassificationState or InputState; require success
            if not classification_state or not getattr(classification_state, 'success', False):
                return create_error_result(
                    'extractor_pipeline',
                    ValueError('Invalid or failed prior state')
                )

            # Validate config with clear failure signaling
            if not self.validate_config(config):
                # Surface concrete reasons to caller/UI
                msg = "; ".join(self._last_validation_errors) if getattr(self, "_last_validation_errors", None) else "invalid configuration"
                return create_error_result('extractor_pipeline', ValueError(f'Extractor config invalid: {msg}'))

            # Get memory item from prior state (prefer data field, fallback to attribute)
            memory_item = None
            try:
                memory_item = classification_state.data.get('memory_item')
            except Exception:
                memory_item = None
            if not memory_item:
                memory_item = getattr(classification_state, 'memory_item', None)
            if not memory_item:
                return create_error_result(
                    'extractor_pipeline',
                    ValueError('No memory_item available for extraction')
                )

            # Cache config for ontology subcall knobs
            try:
                self._current_cfg = config
            except Exception:
                self._current_cfg = None

            # Determine extractor to use
            extractor_name = getattr(config, 'extractor_name', None)
            if not extractor_name:
                extractor_name = self._select_default_extractor()

            if not extractor_name:
                return create_error_result('extractor_pipeline', ValueError('No extractor available: none registered and no default could be selected'))

            # Get fallback order and filter to registered ones
            fallback_order = [n for n in self._get_fallback_order(primary=extractor_name) if n in self.extractor_registry]

            # Handle extraction based on extractor type
            memory_id = getattr(memory_item, "item_id", None)
            with trace_span("pipeline.extract", {"memory_id": memory_id, "extractor": extractor_name}):
                if extractor_name == 'ontology':
                    extraction_result = self._extract_with_ontology(memory_item, fallback_order)
                else:
                    # Try primary extractor first, then fallback
                    all_extractors = [n for n in ([extractor_name] + fallback_order) if n in self.extractor_registry]
                    if not all_extractors:
                        available = ", ".join(sorted(self.extractor_registry.keys()))
                        return create_error_result('extractor_pipeline', ValueError(f"Requested extractor '{extractor_name}' is not registered; available: [{available}]"))
                    extraction_result = self._extract_with_fallback(memory_item, all_extractors)

            # Build extraction metadata
            extraction_metadata = {
                'extractor_requested': extractor_name,
                'extractor_used': extractor_name,  # TODO: track actual extractor used in fallback
                'entities_found': len(extraction_result['entities']),
                'relations_found': len(extraction_result['relations']),
                'fallback_available': len(fallback_order) > 0
            }

            return ComponentResult(
                success=True,
                data={
                    'memory_item': memory_item,  # Pass memory_item forward for storage
                    'entities': extraction_result['entities'],
                    'relations': extraction_result['relations'],
                    'extraction_metadata': extraction_metadata,
                    'extractor_used': extractor_name
                },
                metadata={
                    'stage': 'extractor_pipeline',
                    'extractor_name': extractor_name,
                    'results_count': len(extraction_result['entities']) + len(extraction_result['relations'])
                }
            )

        except Exception as e:
            return create_error_result(
                'extractor_pipeline',
                e,
                extractor_name=getattr(config, 'extractor_name', 'unknown')
            )
