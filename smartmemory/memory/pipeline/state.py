"""
Pipeline state management for componentized memory ingestion flow.
Provides stage checkpoints and dependency resolution for Studio integration.
"""
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List


@dataclass
class ComponentResult:
    """Base result class for pipeline stages"""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InputState(ComponentResult):
    """State after InputAdapter stage"""
    memory_item: Any = None
    input_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationState(ComponentResult):
    """State after ClassificationEngine stage"""
    classified_types: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    classification_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionState(ComponentResult):
    """State after ExtractorPipeline stage"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    extractor_used: str = ""


@dataclass
class StorageState(ComponentResult):
    """State after StorageEngine stage"""
    stored_nodes: List[Dict[str, Any]] = field(default_factory=list)
    stored_relations: List[Dict[str, Any]] = field(default_factory=list)
    storage_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkingState(ComponentResult):
    """State after LinkingEngine stage"""
    linked_entities: List[Dict[str, Any]] = field(default_factory=list)
    dedup_results: Dict[str, Any] = field(default_factory=dict)
    linking_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichmentState(ComponentResult):
    """State after EnrichmentPipeline stage"""
    enriched_data: Dict[str, Any] = field(default_factory=dict)
    plugin_results: Dict[str, Any] = field(default_factory=dict)
    enrichment_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundingState(ComponentResult):
    """State after GroundingEngine stage"""
    grounded_entities: List[Dict[str, Any]] = field(default_factory=list)
    kb_matches: Dict[str, Any] = field(default_factory=dict)
    grounding_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Complete pipeline state with stage checkpoints"""

    # Stage states
    input_state: Optional[InputState] = None
    classification_state: Optional[ClassificationState] = None
    extraction_state: Optional[ExtractionState] = None
    storage_state: Optional[StorageState] = None
    linking_state: Optional[LinkingState] = None
    enrichment_state: Optional[EnrichmentState] = None
    grounding_state: Optional[GroundingState] = None

    # Pipeline metadata
    pipeline_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_latest_for_stage(self, stage_name: str) -> Optional[ComponentResult]:
        """Get the most recent valid state for a given stage"""
        stage_dependencies = {
            'input': self.input_state,
            'classification': self.input_state,
            'extraction': self.classification_state or self.input_state,
            'storage': self.extraction_state,
            'linking': self.storage_state,
            'enrichment': self.linking_state,
            'grounding': self.enrichment_state
        }

        if stage_name not in stage_dependencies:
            raise ValueError(f"Unknown stage: {stage_name}")

        return stage_dependencies[stage_name]

    def is_stage_ready(self, stage_name: str) -> bool:
        """Check if a stage has its dependencies satisfied"""
        dependency_state = self.get_latest_for_stage(stage_name)
        return dependency_state is not None and dependency_state.success

    def get_completed_stages(self) -> List[str]:
        """Get list of successfully completed stages"""
        stages = ['input', 'classification', 'extraction', 'storage', 'linking', 'enrichment', 'grounding']
        completed = []

        for stage in stages:
            state_attr = f"{stage}_state"
            state = getattr(self, state_attr, None)
            if state is not None and state.success:
                completed.append(stage)

        return completed

    def get_next_runnable_stage(self) -> Optional[str]:
        """Get the next stage that can be run"""
        stages = ['input', 'classification', 'extraction', 'storage', 'linking', 'enrichment', 'grounding']
        completed = self.get_completed_stages()

        for stage in stages:
            if stage not in completed:
                if stage == 'input':  # Input has no dependencies
                    return stage
                elif self.is_stage_ready(stage):
                    return stage
                else:
                    return None  # Dependencies not met

        return None  # All stages completed

    def update_timestamp(self):
        """Update the last modified timestamp"""
        self.updated_at = datetime.now(UTC)
