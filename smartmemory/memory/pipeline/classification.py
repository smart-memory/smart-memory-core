"""
ClassificationEngine component for componentized memory ingestion pipeline.
Handles memory type classification with configurable rules and indicators.
"""
from typing import Dict, Set

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import ClassificationConfig
from smartmemory.memory.pipeline.state import InputState
from smartmemory.utils.pipeline_utils import create_error_result


class ClassificationEngine(PipelineComponent[ClassificationConfig]):
    """
    Component responsible for classifying memory items into types for routing.
    Supports configurable classification rules and type indicators.
    """

    def __init__(self):
        self.default_types = {'semantic', 'zettel'}
        self.type_indicators = {
            'episodic': {'episodic', 'event', 'episode'},
            'procedural': {'procedural', 'procedure', 'task', 'step'},
            'working': {'working', 'temporary', 'scratch'}
        }

    def _classify_from_metadata(self, item) -> Set[str]:
        """Extract types from item metadata and tags"""
        types = set(self.default_types)  # Always include core types

        # Get metadata if available
        metadata = getattr(item, 'metadata', {}) or {}
        explicit_type = metadata.get('type')
        tags = metadata.get('tags', [])

        # Add explicit type if present
        if explicit_type:
            types.add(explicit_type)

        # Check for type indicators in metadata and tags
        for memory_type, indicators in self.type_indicators.items():
            # Check explicit type
            if explicit_type and explicit_type.lower() in indicators:
                types.add(memory_type)

            # Check tags
            if any(tag.lower() in indicators for tag in tags):
                types.add(memory_type)

        return types

    def _classify_from_content(self, item, config: ClassificationConfig) -> Set[str]:
        """Extract types from content analysis if enabled"""
        additional_types = set()

        if not bool(getattr(config, 'content_analysis_enabled', False)):
            return additional_types

        content = getattr(item, 'content', '') or ''
        content_lower = content.lower()

        # Simple keyword-based classification
        content_indicators = getattr(config, 'content_indicators', {}) or {}
        for memory_type, keywords in content_indicators.items():
            if any(keyword.lower() in content_lower for keyword in keywords):
                additional_types.add(memory_type)

        return additional_types

    def _apply_confidence_scoring(self, types: Set[str], config: ClassificationConfig) -> Dict[str, float]:
        """Apply confidence scores to classified types"""
        confidence_scores = {}

        # Default confidence for core types
        for memory_type in types:
            if memory_type in self.default_types:
                confidence_scores[memory_type] = float(getattr(config, 'default_confidence', 0.9))
            else:
                confidence_scores[memory_type] = float(getattr(config, 'inferred_confidence', 0.7))

        # Apply custom confidence rules if configured
        custom_confidences = getattr(config, 'type_confidences', {}) or {}
        for memory_type in types:
            if memory_type in custom_confidences:
                confidence_scores[memory_type] = custom_confidences[memory_type]

        return confidence_scores

    def validate_config(self, config: ClassificationConfig) -> bool:
        """Validate ClassificationEngine configuration using typed config"""
        try:
            # Confidence values must be within [0,1]
            dc = float(getattr(config, 'default_confidence', 0.9))
            ic = float(getattr(config, 'inferred_confidence', 0.7))
            if not (0.0 <= dc <= 1.0 and 0.0 <= ic <= 1.0):
                return False

            cae = getattr(config, 'content_analysis_enabled', False)
            if not isinstance(cae, bool):
                return False

            ci = getattr(config, 'content_indicators', {})
            if ci is not None and not isinstance(ci, dict):
                return False

            tc = getattr(config, 'type_confidences', {})
            if tc is not None and not isinstance(tc, dict):
                return False

            return True
        except Exception:
            return False

    def run(self, input_state: InputState, config: ClassificationConfig) -> ComponentResult:
        """
        Execute ClassificationEngine with given input state and configuration.
        
        Args:
            input_state: InputState from previous stage containing MemoryItem
            config: Classification configuration dict
        
        Returns:
            ComponentResult with classified types and confidence scores
        """
        # Early config/state validation guards
        if not self.validate_config(config):
            return create_error_result('classification_engine', ValueError('Invalid classification configuration'))

        try:
            if not input_state or not input_state.success:
                return create_error_result('classification_engine', ValueError('Invalid or failed input state'))

            memory_item = input_state.data.get('memory_item')
            if not memory_item:
                return ComponentResult(
                    success=False,
                    data={'error': 'No memory_item in input state'},
                    metadata={'stage': 'classification_engine'}
                )

            # Classify from metadata and tags
            classified_types = self._classify_from_metadata(memory_item)

            # Add content-based classification if enabled
            content_types = self._classify_from_content(memory_item, config)
            classified_types.update(content_types)

            # Convert to list and apply confidence scoring
            classified_types_list = list(classified_types)
            confidence_scores = self._apply_confidence_scoring(classified_types, config)

            # Build classification metadata
            classification_metadata = {
                'classification_method': 'metadata_and_content' if bool(getattr(config, 'content_analysis_enabled', False)) else 'metadata_only',
                'total_types_found': len(classified_types_list),
                'default_types_included': len(classified_types.intersection(self.default_types)),
                'inferred_types_included': len(classified_types - self.default_types)
            }

            return ComponentResult(
                success=True,
                data={
                    'memory_item': memory_item,  # Pass memory_item forward for extraction
                    'classified_types': classified_types_list,
                    'confidence_scores': confidence_scores,
                    'classification_metadata': classification_metadata
                },
                metadata={
                    'stage': 'classification_engine',
                    'types_count': len(classified_types_list),
                    'confidence_range': f"{min(confidence_scores.values()):.2f}-{max(confidence_scores.values()):.2f}"
                }
            )

        except Exception as e:
            return create_error_result('classification_engine', e)
