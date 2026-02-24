"""
Pipeline orchestrator for Studio-controlled memory ingestion.
Provides stage-by-stage execution with automatic state management.
"""
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generic, TypeVar

from smartmemory.memory.pipeline.state import (
    PipelineState, ComponentResult, InputState, ClassificationState,
    ExtractionState, StorageState, LinkingState, EnrichmentState, GroundingState
)
from smartmemory.models.base import MemoryBaseModel

TConfig = TypeVar("TConfig", bound=MemoryBaseModel)


class PipelineComponent(ABC, Generic[TConfig]):
    """Base class for all pipeline stages"""

    @abstractmethod
    def run(self, input_state: Optional[ComponentResult], config: TConfig) -> ComponentResult:
        """Execute component with given input state and config"""
        pass

    def validate_config(self, config: TConfig) -> bool:
        """Validate configuration parameters"""
        return True


from smartmemory.memory.pipeline.config import (
    InputAdapterConfig,
    ClassificationConfig,
    ExtractionConfig,
    StorageConfig,
    LinkingConfig,
    EnrichmentConfig,
    GroundingConfig,
)


class Pipeline:
    """
    Pipeline orchestrator that provides Studio with stage-by-stage control.
    Each stage can be run independently with automatic state management.
    """

    def __init__(self):
        self.state = PipelineState()
        self.components: Dict[str, PipelineComponent] = {}

    def register_component(self, stage_name: str, component: PipelineComponent):
        """Register a component for a specific stage"""
        self.components[stage_name] = component

    def run_input_adapter(self, config: InputAdapterConfig) -> InputState:
        """Run the input adapter stage"""
        start_time = time.time()

        try:
            if 'input' not in self.components:
                error_state = InputState(
                    success=False,
                    data={'error': 'InputAdapter component not registered'},
                    metadata={'stage': 'input', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.input_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['input']

            # Pass typed config to the component
            result = component.run(None, config)
            result.execution_time = time.time() - start_time

            # Convert to InputState
            input_state = InputState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                memory_item=result.data.get('memory_item'),
                input_metadata=result.data.get('input_metadata', {})
            )

            self.state.input_state = input_state
            self.state.update_timestamp()
            return input_state

        except Exception as e:
            error_state = InputState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'input', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.input_state = error_state
            self.state.update_timestamp()
            return error_state

    def run_classification(self, config: ClassificationConfig) -> ClassificationState:
        """Run the classification stages stage"""

        start_time = time.time()

        try:
            if 'classification' not in self.components:
                error_state = ClassificationState(
                    success=False,
                    data={'error': 'ClassificationEngine component not registered'},
                    metadata={'stage': 'classification', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.classification_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['classification']
            if not component.validate_config(config):
                error_state = ClassificationState(
                    success=False,
                    data={'error': 'Invalid configuration for ClassificationEngine'},
                    metadata={'stage': 'classification', 'error_type': 'InvalidConfig'},
                    execution_time=time.time() - start_time
                )
                self.state.classification_state = error_state
                self.state.update_timestamp()
                return error_state

            input_state = self.state.get_latest_for_stage('classification')
            # Pass typed config to the component
            result = component.run(input_state, config)
            result.execution_time = time.time() - start_time

            # Convert to ClassificationState
            classification_state = ClassificationState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                classified_types=result.data.get('classified_types', []),
                confidence_scores=result.data.get('confidence_scores', {}),
                classification_metadata=result.data.get('classification_metadata', {})
            )

            self.state.classification_state = classification_state
            self.state.update_timestamp()
            return classification_state

        except Exception as e:
            error_state = ClassificationState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'classification', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.classification_state = error_state
            self.state.update_timestamp()
            return error_state

    def run_extraction(self, config: ExtractionConfig) -> ExtractionState:
        """Run the extractor pipeline stage"""
        # Prefer input dependency, but do not hard fail here; component may handle gracefully.

        start_time = time.time()

        try:
            if 'extraction' not in self.components:
                error_state = ExtractionState(
                    success=False,
                    data={'error': 'ExtractorPipeline component not registered'},
                    metadata={'stage': 'extraction', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.extraction_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['extraction']
            if not component.validate_config(config):
                error_state = ExtractionState(
                    success=False,
                    data={'error': 'Invalid configuration for ExtractorPipeline'},
                    metadata={'stage': 'extraction', 'error_type': 'InvalidConfig'},
                    execution_time=time.time() - start_time
                )
                self.state.extraction_state = error_state
                self.state.update_timestamp()
                return error_state

            # Feed the input stage result directly into extraction
            latest_input_state = self.state.get_latest_for_stage('input')
            # Pass typed config to the component
            result = component.run(latest_input_state, config)
            result.execution_time = time.time() - start_time

            # Convert to ExtractionState
            extraction_state = ExtractionState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                entities=result.data.get('entities', []),
                relations=result.data.get('relations', []),
                extraction_metadata=result.data.get('extraction_metadata', {}),
                extractor_used=result.data.get('extractor_used', '')
            )

            self.state.extraction_state = extraction_state
            self.state.update_timestamp()
            return extraction_state

        except Exception as e:
            error_state = ExtractionState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'extraction', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.extraction_state = error_state
            self.state.update_timestamp()
            return error_state

    def run_storage(self, config: StorageConfig) -> StorageState:
        """Run the storage stages stage"""
        if not self.state.is_stage_ready('storage'):
            error_state = StorageState(
                success=False,
                data={'error': 'Extraction stage must be completed successfully first'},
                metadata={'stage': 'storage', 'error_type': 'StageNotReady'},
                execution_time=0.0
            )
            self.state.storage_state = error_state
            self.state.update_timestamp()
            return error_state

        start_time = time.time()

        try:
            if 'storage' not in self.components:
                error_state = StorageState(
                    success=False,
                    data={'error': 'StorageEngine component not registered'},
                    metadata={'stage': 'storage', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.storage_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['storage']
            if not component.validate_config(config):
                error_state = StorageState(
                    success=False,
                    data={'error': 'Invalid configuration for StorageEngine'},
                    metadata={'stage': 'storage', 'error_type': 'InvalidConfig'},
                    execution_time=time.time() - start_time
                )
                self.state.storage_state = error_state
                self.state.update_timestamp()
                return error_state

            extraction_state = self.state.get_latest_for_stage('storage')
            result = component.run(extraction_state, config)
            result.execution_time = time.time() - start_time

            # Convert to StorageState
            storage_state = StorageState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                stored_nodes=result.data.get('stored_nodes', []),
                stored_relations=result.data.get('stored_relations', []),
                storage_metadata=result.data.get('storage_metadata', {})
            )

            self.state.storage_state = storage_state
            self.state.update_timestamp()
            return storage_state

        except Exception as e:
            error_state = StorageState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'storage', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.storage_state = error_state
            self.state.update_timestamp()
            return error_state

    def run_linking(self, config: LinkingConfig) -> LinkingState:
        """Run the linking stages stage"""
        if not self.state.is_stage_ready('linking'):
            error_state = LinkingState(
                success=False,
                data={'error': 'Storage stage must be completed successfully first'},
                metadata={'stage': 'linking', 'error_type': 'StageNotReady'},
                execution_time=0.0
            )
            self.state.linking_state = error_state
            self.state.update_timestamp()
            return error_state

        start_time = time.time()

        try:
            if 'linking' not in self.components:
                error_state = LinkingState(
                    success=False,
                    data={'error': 'LinkingEngine component not registered'},
                    metadata={'stage': 'linking', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.linking_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['linking']
            if not component.validate_config(config):
                error_state = LinkingState(
                    success=False,
                    data={'error': 'Invalid configuration for LinkingEngine'},
                    metadata={'stage': 'linking', 'error_type': 'InvalidConfig'},
                    execution_time=time.time() - start_time
                )
                self.state.linking_state = error_state
                self.state.update_timestamp()
                return error_state

            storage_state = self.state.get_latest_for_stage('linking')
            result = component.run(storage_state, config)
            result.execution_time = time.time() - start_time

            # Convert to LinkingState
            linking_state = LinkingState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                linked_entities=result.data.get('linked_entities', []),
                dedup_results=result.data.get('dedup_results', {}),
                linking_metadata=result.data.get('linking_metadata', {})
            )

            self.state.linking_state = linking_state
            self.state.update_timestamp()
            return linking_state

        except Exception as e:
            error_state = LinkingState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'linking', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.linking_state = error_state
            self.state.update_timestamp()
            return error_state

    def run_enrichment(self, config: EnrichmentConfig) -> EnrichmentState:
        """Run the enrichment stage"""
        if not self.state.is_stage_ready('enrichment'):
            error_state = EnrichmentState(
                success=False,
                data={'error': 'Linking stage must be completed successfully first'},
                metadata={'stage': 'enrichment', 'error_type': 'StageNotReady'},
                execution_time=0.0
            )
            self.state.enrichment_state = error_state
            self.state.update_timestamp()
            return error_state

        start_time = time.time()

        try:
            if 'enrichment' not in self.components:
                error_state = EnrichmentState(
                    success=False,
                    data={'error': 'EnrichmentPipeline component not registered'},
                    metadata={'stage': 'enrichment', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.enrichment_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['enrichment']
            if not component.validate_config(config):
                error_state = EnrichmentState(
                    success=False,
                    data={'error': 'Invalid configuration for EnrichmentPipeline'},
                    metadata={'stage': 'enrichment', 'error_type': 'InvalidConfig'},
                    execution_time=time.time() - start_time
                )
                self.state.enrichment_state = error_state
                self.state.update_timestamp()
                return error_state

            linking_state = self.state.get_latest_for_stage('enrichment')
            result = component.run(linking_state, config)
            result.execution_time = time.time() - start_time

            enrichment_state = EnrichmentState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                enriched_data=result.data.get('enriched_data', {}),
                plugin_results=result.data.get('plugin_results', {}),
                enrichment_metadata=result.data.get('enrichment_metadata', {})
            )

            self.state.enrichment_state = enrichment_state
            self.state.update_timestamp()
            return enrichment_state

        except Exception as e:
            error_state = EnrichmentState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'enrichment', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.enrichment_state = error_state
            self.state.update_timestamp()
            return error_state

    def run_grounding(self, config: GroundingConfig) -> GroundingState:
        """Run the grounding stages stage"""
        if not self.state.is_stage_ready('grounding'):
            error_state = GroundingState(
                success=False,
                data={'error': 'Enrichment stage must be completed successfully first'},
                metadata={'stage': 'grounding', 'error_type': 'StageNotReady'},
                execution_time=0.0
            )
            self.state.grounding_state = error_state
            self.state.update_timestamp()
            return error_state

        start_time = time.time()

        try:
            if 'grounding' not in self.components:
                error_state = GroundingState(
                    success=False,
                    data={'error': 'GroundingEngine component not registered'},
                    metadata={'stage': 'grounding', 'error_type': 'ComponentNotRegistered'},
                    execution_time=time.time() - start_time
                )
                self.state.grounding_state = error_state
                self.state.update_timestamp()
                return error_state

            component = self.components['grounding']
            if not component.validate_config(config):
                error_state = GroundingState(
                    success=False,
                    data={'error': 'Invalid configuration for GroundingEngine'},
                    metadata={'stage': 'grounding', 'error_type': 'InvalidConfig'},
                    execution_time=time.time() - start_time
                )
                self.state.grounding_state = error_state
                self.state.update_timestamp()
                return error_state

            enrichment_state = self.state.get_latest_for_stage('grounding')
            result = component.run(enrichment_state, config)
            result.execution_time = time.time() - start_time

            # Convert to GroundingState
            grounding_state = GroundingState(
                success=result.success,
                data=result.data,
                metadata=result.metadata,
                execution_time=result.execution_time,
                grounded_entities=result.data.get('grounded_entities', []),
                kb_matches=result.data.get('kb_matches', {}),
                grounding_metadata=result.data.get('grounding_metadata', {})
            )

            self.state.grounding_state = grounding_state
            self.state.update_timestamp()
            return grounding_state

        except Exception as e:
            error_state = GroundingState(
                success=False,
                data={'error': str(e)},
                metadata={'stage': 'grounding', 'error_type': type(e).__name__},
                execution_time=time.time() - start_time
            )
            self.state.grounding_state = error_state
            self.state.update_timestamp()
            return error_state

    def get_pipeline_state(self) -> PipelineState:
        """Get current pipeline state"""
        return self.state

    def reset_pipeline(self):
        """Reset pipeline to initial state"""
        self.state = PipelineState(pipeline_id=str(uuid.uuid4()))

    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of all stages for Studio UI"""
        completed = self.state.get_completed_stages()
        next_runnable = self.state.get_next_runnable_stage()
        total_stages = 7  # input, classification, extraction, storage, linking, enrichment, grounding
        completion_percentage = (len(completed) / total_stages) * 100.0 if total_stages > 0 else 0.0

        return {
            'pipeline_id': self.state.pipeline_id,
            'completed_stages': completed,
            'next_runnable_stage': next_runnable,
            'total_stages': total_stages,
            'completion_percentage': completion_percentage,
            'created_at': self.state.created_at.isoformat(),
            'updated_at': self.state.updated_at.isoformat(),
            'stage_states': {
                'input': self.state.input_state.success if self.state.input_state else None,
                'classification': self.state.classification_state.success if self.state.classification_state else None,
                'extraction': self.state.extraction_state.success if self.state.extraction_state else None,
                'storage': self.state.storage_state.success if self.state.storage_state else None,
                'linking': self.state.linking_state.success if self.state.linking_state else None,
                'enrichment': self.state.enrichment_state.success if self.state.enrichment_state else None,
                'grounding': self.state.grounding_state.success if self.state.grounding_state else None,
            }
        }
