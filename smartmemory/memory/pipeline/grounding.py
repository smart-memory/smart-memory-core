"""GroundingEngine component for componentized memory ingestion pipeline.
Handles synchronous grounding with external knowledge bases and provenance linking.
"""
import logging
from typing import Dict, Any

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import GroundingConfig
from smartmemory.memory.pipeline.state import EnrichmentState
from smartmemory.observability.tracing import trace_span
from smartmemory.utils.pipeline_utils import create_error_result

logger = logging.getLogger(__name__)


class GroundingEngine(PipelineComponent[GroundingConfig]):
    """
    Component responsible for grounding entities with external knowledge bases.
    Creates GROUNDED_IN edges for provenance candidates and KB matches.
    """

    def __init__(self, grounding_instance=None):
        if grounding_instance is None:
            raise ValueError("GroundingEngine requires a valid grounding_instance")
        self.grounding = grounding_instance

    def _ground_with_wikipedia(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ground entities with Wikipedia data if present in enrichment"""
        enrichment_result = context.get('enrichment_result')
        wiki_data = enrichment_result.get('wikipedia_data') or {} if enrichment_result else {}

        grounded_entities = []
        kb_matches = {}

        if wiki_data:
            for entity_name, wiki_info in wiki_data.items():
                if isinstance(wiki_info, dict) and 'url' in wiki_info:
                    grounded_entities.append({
                        'entity': entity_name,
                        'source': 'wikipedia',
                        'url': wiki_info['url'],
                        'title': wiki_info.get('title', entity_name),
                        'summary': wiki_info.get('summary', '')
                    })
                    kb_matches[entity_name] = {
                        'source': 'wikipedia',
                        'confidence': wiki_info.get('confidence', 0.8),
                        'metadata': wiki_info
                    }

        return {
            'grounded_entities': grounded_entities,
            'kb_matches': kb_matches,
            'wikipedia_matches': len(grounded_entities)
        }

    def _ground_with_custom_kb(self, context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Ground entities with custom knowledge bases"""
        grounded_entities = []
        kb_matches = {}

        # Custom KB grounding logic would go here
        # For now, this is a placeholder implementation
        kb_sources = config.get('knowledge_bases', [])

        for kb_source in kb_sources:
            # Placeholder for custom KB integration
            pass

        return {
            'grounded_entities': grounded_entities,
            'kb_matches': kb_matches,
            'custom_kb_matches': len(grounded_entities)
        }

    def validate_config(self, config: GroundingConfig) -> bool:
        """Validate GroundingEngine configuration using typed config"""
        try:
            kb_sources = getattr(config, 'knowledge_bases', [])
            if kb_sources is not None and not isinstance(kb_sources, list):
                return False
            strategy = getattr(config, 'grounding_strategy', 'wikipedia')
            if strategy not in ['wikipedia', 'custom', 'both', 'none', 'hybrid', 'knowledge_base']:
                return False
            ct = float(getattr(config, 'confidence_threshold', 0.7))
            if not (0.0 <= ct <= 1.0):
                return False
            return True
        except Exception:
            return False

    def run(self, enrichment_state: EnrichmentState, config: GroundingConfig) -> ComponentResult:
        """
        Execute GroundingEngine with given enrichment state and configuration.
        
        Args:
            enrichment_state: EnrichmentState from previous stage with enrichment results
            config: Grounding configuration dict
        
        Returns:
            ComponentResult with grounding results and metadata
        """
        try:
            if not enrichment_state or not enrichment_state.success:
                return create_error_result('grounding_engine', ValueError('Invalid or failed enrichment state'))

            # Extract context and provenance candidates
            context = enrichment_state.data.get('context', {})
            provenance_candidates = enrichment_state.data.get('provenance_candidates', [])

            if not context:
                return ComponentResult(
                    success=False,
                    data={'error': 'No context available from enrichment state'},
                    metadata={'stage': 'grounding_engine'}
                )

            # Initialize grounding results
            all_grounded_entities = []
            all_kb_matches = {}
            grounding_metadata = {
                'grounding_strategy': getattr(config, 'grounding_strategy', 'wikipedia'),
                'confidence_threshold': getattr(config, 'confidence_threshold', 0.7),
                'provenance_candidates_count': len(provenance_candidates)
            }

            # Perform grounding based on strategy
            strategy = getattr(config, 'grounding_strategy', 'wikipedia')

            if strategy in ['wikipedia', 'both']:
                # Ground with Wikipedia
                wiki_results = self._ground_with_wikipedia(context)
                all_grounded_entities.extend(wiki_results['grounded_entities'])
                all_kb_matches.update(wiki_results['kb_matches'])
                grounding_metadata['wikipedia_matches'] = wiki_results['wikipedia_matches']

            if strategy in ['custom', 'both']:
                # Ground with custom knowledge bases
                # Convert subset to dict for the helper expecting dict
                custom_results = self._ground_with_custom_kb(context, {'knowledge_bases': getattr(config, 'knowledge_bases', [])})
                all_grounded_entities.extend(custom_results['grounded_entities'])
                all_kb_matches.update(custom_results['kb_matches'])
                grounding_metadata['custom_kb_matches'] = custom_results['custom_kb_matches']

            # Use the grounding module for provenance candidates
            grounding_success = True
            grounding_error = None

            if provenance_candidates and strategy != 'none':
                try:
                    # Add provenance candidates to context for grounding
                    context['provenance_candidates'] = provenance_candidates
                    with trace_span("pipeline.ground", {"memory_id": context.get('item_id'), "strategy": strategy, "provenance_candidate_count": len(provenance_candidates)}):
                        self.grounding.ground(context)
                    grounding_metadata['provenance_grounding_success'] = True
                except Exception as e:
                    grounding_success = False
                    grounding_error = str(e)
                    grounding_metadata['provenance_grounding_success'] = False
                    logger.warning(f"Provenance grounding failed: {e}")

            # Filter results by confidence threshold
            confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
            filtered_matches = {}
            for entity, match_info in all_kb_matches.items():
                if match_info.get('confidence', 0.0) >= confidence_threshold:
                    filtered_matches[entity] = match_info

            # Update grounding metadata
            grounding_metadata.update({
                'total_entities_processed': len(all_grounded_entities),
                'high_confidence_matches': len(filtered_matches),
                'grounding_success': grounding_success,
                'grounding_error': grounding_error
            })

            return ComponentResult(
                success=True,
                data={
                    'grounded_entities': all_grounded_entities,
                    'kb_matches': all_kb_matches,
                    'filtered_matches': filtered_matches,
                    'grounding_metadata': grounding_metadata,
                    'context': context  # Pass final context
                },
                metadata={
                    'stage': 'grounding_engine',
                    'entities_grounded': len(all_grounded_entities),
                    'kb_sources_used': len(getattr(config, 'knowledge_bases', []) or []) + (1 if strategy in ['wikipedia', 'both'] else 0)
                }
            )

        except Exception as e:
            return create_error_result('grounding_engine', e)
