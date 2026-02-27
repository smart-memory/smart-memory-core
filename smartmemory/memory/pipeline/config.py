"""
Pipeline configuration dataclasses for componentized ingestion flow.
All configs inherit from MemoryBaseModel for consistent user context.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from smartmemory.models.base import MemoryBaseModel


@dataclass
class ClassificationConfig(MemoryBaseModel):
    """Configuration for ClassificationEngine stage."""
    content_analysis_enabled: bool = False
    default_confidence: float = 0.9
    inferred_confidence: float = 0.7
    content_indicators: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        # Set default content indicators if none provided
        if not self.content_indicators:
            self.content_indicators = {
                'episodic': ['event', 'happened', 'occurred', 'experience'],
                'semantic': ['definition', 'concept', 'knowledge', 'fact'],
                'procedural': ['how to', 'steps', 'process', 'method'],
                'zettel': ['note', 'idea', 'thought', 'reminder'],
                'reasoning': [
                    'therefore', 'because', 'reasoning', 'conclude', 'concluded',
                    'inference', 'deduced', 'it follows', 'thus', 'hence',
                    'my reasoning', 'i think because', 'this means',
                ],
                'decision': [
                    'decided', 'decision', 'chose', 'choosing', 'i will',
                    'going to', 'resolved', 'agreed', 'committed', 'opted',
                    'my decision', 'we decided', "i've decided",
                ],
            }


@dataclass
class LinkingConfig(MemoryBaseModel):
    """Configuration for LinkingEngine stage.
    
    Linking performs deduplication and cross-reference resolution.
    Two modes available:
    - basic: Fast embedding-based similarity matching
    - semantic: LLM-based semantic equivalence (catches "Joe" ↔ "Joseph")
    """
    linking_algorithm: str = "default"
    similarity_threshold: float = 0.8
    deduplication_enabled: bool = True
    cross_reference_resolution: bool = True
    working_memory_activation: bool = True
    # Clustering options
    clustering_mode: str = "basic"  # "basic" | "semantic"
    clustering_model: Optional[str] = None  # LLM model for semantic clustering
    clustering_context: Optional[str] = None  # Domain context hint for clustering

    def __post_init__(self):
        # Validate threshold range
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
        # Validate clustering mode
        valid_modes = {"basic", "semantic", "none"}
        if self.clustering_mode not in valid_modes:
            raise ValueError(f"clustering_mode must be one of {valid_modes}, got '{self.clustering_mode}'")


@dataclass
class StorageConfig(MemoryBaseModel):
    """Configuration for StorageEngine stage."""
    storage_strategy: str = "dual_node"
    enable_triple_processing: bool = True
    ontology_extraction: bool = False
    relationship_creation: bool = True

    def __post_init__(self):
        # Validate storage strategy
        valid_strategies = {"dual_node", "single_node", "memory_only"}
        if self.storage_strategy not in valid_strategies:
            raise ValueError(f"storage_strategy must be one of {valid_strategies}, got '{self.storage_strategy}'")


@dataclass
class CoreferenceConfig(MemoryBaseModel):
    """Configuration for CoreferenceStage.

    Coreference resolution resolves pronouns and vague references to explicit
    entity names (e.g., "The company announced..." → "Apple announced...").
    This improves entity extraction quality by making implicit references explicit.
    """
    enabled: bool = True
    """Whether coreference resolution is enabled."""

    resolver: str = "fastcoref"
    """Resolver to use: 'fastcoref' (recommended) or 'spacy'."""

    device: str = "auto"
    """Device for model inference: 'auto', 'mps', 'cuda', or 'cpu'."""

    min_text_length: int = 50
    """Minimum text length to process (skip very short content)."""

    def __post_init__(self):
        # Validate resolver
        valid_resolvers = {"fastcoref", "spacy"}
        if self.resolver not in valid_resolvers:
            raise ValueError(f"resolver must be one of {valid_resolvers}, got '{self.resolver}'")

        # Validate device
        valid_devices = {"auto", "mps", "cuda", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{self.device}'")


@dataclass
class GroundingConfig(MemoryBaseModel):
    """Configuration for GroundingEngine stage."""
    grounding_strategy: str = "wikipedia"
    knowledge_bases: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    enable_provenance_grounding: bool = True
    max_grounding_depth: int = 3

    def __post_init__(self):
        # Set default knowledge bases if none provided
        if not self.knowledge_bases:
            self.knowledge_bases = ["wikipedia"]

        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")

        # Validate strategy
        valid_strategies = {"wikipedia", "custom", "hybrid", "none"}
        if self.grounding_strategy not in valid_strategies:
            raise ValueError(f"grounding_strategy must be one of {valid_strategies}, got '{self.grounding_strategy}'")


@dataclass
class EnrichmentConfig(MemoryBaseModel):
    """Configuration for EnrichmentPipeline stage."""
    enricher_names: Optional[List[str]] = None
    enricher_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-enricher configuration. Keys are enricher names, values are config dicts."""
    execution_order: str = "parallel"
    enable_derived_items: bool = True
    max_enrichment_depth: int = 2

    def __post_init__(self):
        # Validate execution order
        valid_orders = {"parallel", "sequential", "priority"}
        if self.execution_order not in valid_orders:
            raise ValueError(f"execution_order must be one of {valid_orders}, got '{self.execution_order}'")


@dataclass
class ExtractionConfig(MemoryBaseModel):
    """Configuration for ExtractorPipeline stage."""
    extractor_name: Optional[str] = None
    enable_fallback_chain: bool = True
    ontology_extraction: bool = False
    legacy_extractor_support: bool = True
    max_extraction_attempts: int = 3
    # Studio/server DTO compatibility
    max_entities: int = 10
    enable_relations: bool = True
    # Ontology/LLM knobs (optional)
    ontology_profile: Optional[str] = None
    ontology_enabled: Optional[bool] = None
    ontology_constraints: List[str] = field(default_factory=list)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    max_reasoning_tokens: Optional[int] = None

    def __post_init__(self):
        # Validate extractor name if provided
        valid_extractors = {"llm", "spacy", "gliner", "relik", "ontology", None}
        if self.extractor_name not in valid_extractors:
            # Allow any string for custom extractors, just warn
            pass


@dataclass
class InputAdapterConfig(MemoryBaseModel):
    """Configuration for InputAdapter stage."""
    adapter_name: str = "default"
    content: Any = None
    memory_type: str = "semantic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalize_metadata: bool = True
    validate_input: bool = True

    def __post_init__(self):
        # Validate adapter name
        valid_adapters = {"default", "text", "dict", "json", "file"}
        if self.adapter_name not in valid_adapters:
            # Allow custom adapter names
            pass


@dataclass
class PipelineConfigBundle(MemoryBaseModel):
    """Bundle of all pipeline stage configurations."""
    input_adapter: Optional[InputAdapterConfig] = None
    classification: Optional[ClassificationConfig] = None
    coreference: Optional[CoreferenceConfig] = None
    extraction: Optional[ExtractionConfig] = None
    storage: Optional[StorageConfig] = None
    linking: Optional[LinkingConfig] = None
    enrichment: Optional[EnrichmentConfig] = None
    grounding: Optional[GroundingConfig] = None

    def __post_init__(self):
        # Initialize default configs if None
        if self.input_adapter is None:
            self.input_adapter = InputAdapterConfig()
        if self.classification is None:
            self.classification = ClassificationConfig()
        if self.coreference is None:
            self.coreference = CoreferenceConfig()
        if self.extraction is None:
            self.extraction = ExtractionConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.linking is None:
            self.linking = LinkingConfig()
        if self.enrichment is None:
            self.enrichment = EnrichmentConfig()
        if self.grounding is None:
            self.grounding = GroundingConfig()

    def get_config_for_stage(self, stage: str) -> Optional[MemoryBaseModel]:
        """Get configuration for a specific pipeline stage."""
        stage_mapping = {
            'input': self.input_adapter,
            'input_adapter': self.input_adapter,
            'classification': self.classification,
            'coreference': self.coreference,
            'extraction': self.extraction,
            'extractor': self.extraction,
            'storage': self.storage,
            'linking': self.linking,
            'enrichment': self.enrichment,
            'grounding': self.grounding
        }
        return stage_mapping.get(stage)
