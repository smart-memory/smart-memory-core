"""PipelineConfig — nested dataclass configuration for the v2 pipeline.

Follows the project convention of dataclass + MemoryBaseModel (not Pydantic).
Coexists with the legacy ``smartmemory.memory.pipeline.config.PipelineConfigBundle``
until Phase 3 when old orchestrators are deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smartmemory.models.base import MemoryBaseModel


# ------------------------------------------------------------------ #
# Leaf configs
# ------------------------------------------------------------------ #


@dataclass
class RetryConfig(MemoryBaseModel):
    """Global retry behaviour (default for all stages)."""

    max_retries: int = 2
    backoff_seconds: float = 1.0
    on_failure: str = "abort"  # "abort" | "skip"

    def __post_init__(self):
        if self.on_failure not in ("abort", "skip"):
            raise ValueError(f"on_failure must be 'abort' or 'skip', got '{self.on_failure}'")


@dataclass
class StageRetryPolicy(MemoryBaseModel):
    """Per-stage retry policy override.

    When set for a stage, overrides the global RetryConfig for that stage.
    """

    max_retries: int = 2
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    on_failure: str = "abort"  # "abort" | "skip"

    def __post_init__(self):
        if self.on_failure not in ("abort", "skip"):
            raise ValueError(f"on_failure must be 'abort' or 'skip', got '{self.on_failure}'")


@dataclass
class ClassifyConfig(MemoryBaseModel):
    """Classification stage knobs."""

    content_analysis_enabled: bool = False
    default_confidence: float = 0.9
    inferred_confidence: float = 0.7


@dataclass
class CoreferenceConfig(MemoryBaseModel):
    """Coreference resolution stage knobs."""

    enabled: bool = True
    resolver: str = "fastcoref"
    device: str = "auto"
    min_text_length: int = 50


@dataclass
class SimplifyConfig(MemoryBaseModel):
    """Text simplification stage knobs."""

    enabled: bool = True
    split_clauses: bool = True
    extract_relative: bool = True
    passive_to_active: bool = True
    extract_appositives: bool = True
    min_token_count: int = 4


@dataclass
class EntityRulerConfig(MemoryBaseModel):
    """Rule-based entity extraction."""

    enabled: bool = True
    patterns_path: Optional[str] = None
    pattern_sources: List[str] = field(default_factory=lambda: ["builtin"])
    min_confidence: float = 0.85
    spacy_model: str = "en_core_web_sm"


@dataclass
class LLMExtractConfig(MemoryBaseModel):
    """LLM-based entity/relation extraction."""

    enabled: bool = True
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_entities: int = 10
    max_relations: int = 30
    enable_relations: bool = True
    extract_decisions: bool = False  # CORE-SYS2-1b: widen LLM schema to auto-extract decisions
    decision_confidence_threshold: float = 0.75  # CORE-SYS2-1b: min confidence to store a decision


@dataclass
class PromotionConfig(MemoryBaseModel):
    """Entity type promotion rules."""

    auto_promote_threshold: int = 3
    require_approval: bool = False
    reasoning_validation: bool = False
    min_frequency: int = 2
    min_confidence: float = 0.7
    min_type_consistency: float = 0.8
    min_name_length: int = 3


@dataclass
class ConstrainConfig(MemoryBaseModel):
    """Post-extraction constraint filtering."""

    max_entities: int = 20
    max_relations: int = 40
    confidence_threshold: float = 0.5
    domain_range_validation: bool = True


@dataclass
class RelationDiscoveryConfig(MemoryBaseModel):
    """Relation type discovery and promotion rules (CORE-EXT-1c)."""

    enabled: bool = True
    min_frequency: int = 3
    min_cluster_frequency: int = 5
    auto_promote_threshold: int = 10
    auto_promote_min_workspaces: int = 1
    embedding_similarity_threshold: float = 0.75
    max_novel_labels_per_workspace: int = 1000


# ------------------------------------------------------------------ #
# Detection configs
# ------------------------------------------------------------------ #


@dataclass
class ReasoningDetectConfig(MemoryBaseModel):
    """Opt-in reasoning trace detection (CORE-SYS2-1c)."""

    enabled: bool = False  # opt-in
    min_quality_score: float = 0.4  # matches ReasoningEvaluation.should_store threshold
    use_llm_detection: bool = True  # allow LLM fallback for implicit reasoning


# ------------------------------------------------------------------ #
# Composite configs
# ------------------------------------------------------------------ #


@dataclass
class ExtractionConfig(MemoryBaseModel):
    """Composite extraction configuration."""

    extractor_name: Optional[str] = None
    entity_ruler: EntityRulerConfig = field(default_factory=EntityRulerConfig)
    llm_extract: LLMExtractConfig = field(default_factory=LLMExtractConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    relation_discovery: RelationDiscoveryConfig = field(default_factory=RelationDiscoveryConfig)
    constrain: ConstrainConfig = field(default_factory=ConstrainConfig)
    reasoning_detect: ReasoningDetectConfig = field(default_factory=ReasoningDetectConfig)
    max_extraction_attempts: int = 3


@dataclass
class StoreConfig(MemoryBaseModel):
    """Storage stage configuration."""

    strategy: str = "dual_node"
    enable_triple_processing: bool = True


@dataclass
class LinkConfig(MemoryBaseModel):
    """Linking stage configuration."""

    algorithm: str = "default"
    similarity_threshold: float = 0.8
    deduplication_enabled: bool = True


@dataclass
class WikidataConfig(MemoryBaseModel):
    """Wikipedia/Wikidata grounding configuration."""

    enabled: bool = True
    sparql_enabled: bool = True  # When False, only SQLite alias lookup; no HTTP
    confidence_threshold: float = 0.7
    max_grounding_depth: int = 3


@dataclass
class SentimentConfig(MemoryBaseModel):
    """Sentiment enrichment."""

    enabled: bool = True


@dataclass
class TemporalConfig(MemoryBaseModel):
    """Temporal enrichment."""

    enabled: bool = True


@dataclass
class TopicConfig(MemoryBaseModel):
    """Topic enrichment."""

    enabled: bool = True


@dataclass
class EnrichConfig(MemoryBaseModel):
    """Composite enrichment configuration."""

    enricher_names: Optional[List[str]] = None
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    topic: TopicConfig = field(default_factory=TopicConfig)
    wikidata: WikidataConfig = field(default_factory=WikidataConfig)


@dataclass
class EvolveConfig(MemoryBaseModel):
    """Evolution stage configuration."""

    run_evolution: bool = True
    run_clustering: bool = True


# ------------------------------------------------------------------ #
# Top-level PipelineConfig
# ------------------------------------------------------------------ #


@dataclass
class PipelineConfig(MemoryBaseModel):
    """Top-level pipeline configuration."""

    workspace_id: Optional[str] = None
    mode: str = "sync"  # "sync" | "async" | "preview"
    retry: RetryConfig = field(default_factory=RetryConfig)
    stage_retry_policies: Dict[str, StageRetryPolicy] = field(default_factory=dict)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    coreference: CoreferenceConfig = field(default_factory=CoreferenceConfig)
    simplify: SimplifyConfig = field(default_factory=SimplifyConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    evolve: EvolveConfig = field(default_factory=EvolveConfig)

    def __post_init__(self):
        valid_modes = {"sync", "async", "preview"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")

    @classmethod
    def default(cls, workspace_id: Optional[str] = None) -> PipelineConfig:
        """Standard production configuration."""
        return cls(workspace_id=workspace_id)

    @classmethod
    def preview(cls, workspace_id: Optional[str] = None) -> PipelineConfig:
        """Preview mode — no storage or evolution side-effects."""
        return cls(
            workspace_id=workspace_id,
            mode="preview",
            evolve=EvolveConfig(run_evolution=False, run_clustering=False),
        )

    @classmethod
    def lite(cls, workspace_id: Optional[str] = None) -> "PipelineConfig":
        """Lite mode — maximum quality with zero external dependencies.

        Disables only stages that require network access or LLM API calls.
        All local processing (spaCy NER, EntityRuler, sentiment, temporal,
        topic enrichment) runs at full quality.

        Disabled (external dependencies):
        - coreference: fastcoref downloads models on first use
        - llm_extract: requires LLM API key (OpenAI, Groq, etc.)
        - wikipedia_enricher / link_expansion_enricher: HTTP calls
        - wikidata grounding: HTTP calls to Wikidata REST + SPARQL

        Enabled (fully local):
        - simplify: clause splitting, passive→active (pure spaCy)
        - entity_ruler: pattern-matched NER at ~4ms (no network)
        - basic_enricher, sentiment_enricher, temporal_enricher, topic_enricher
        - store, link stages
        - evolution/clustering: disabled (HebbianCoRetrievalEvolver uses raw Cypher)
        """
        return cls(
            workspace_id=workspace_id,
            coreference=CoreferenceConfig(enabled=False),
            extraction=ExtractionConfig(
                llm_extract=LLMExtractConfig(enabled=False),
            ),
            enrich=EnrichConfig(
                enricher_names=["basic_enricher", "sentiment_enricher", "temporal_enricher", "topic_enricher"],
                wikidata=WikidataConfig(sparql_enabled=False),
            ),
            evolve=EvolveConfig(run_evolution=False, run_clustering=False),
        )

    @classmethod
    def tier1(cls, workspace_id: Optional[str] = None) -> "PipelineConfig":
        """Tier 1 extraction profile: spaCy + EntityRuler only, no LLM call.

        Runs the full pipeline except coreference resolution and LLM extraction.
        All enrichment, linking, grounding, and evolution stages run normally.
        The caller enqueues the stored item_id for Tier 2 LLM processing.
        """
        return cls(
            workspace_id=workspace_id,
            coreference=CoreferenceConfig(enabled=False),
            extraction=ExtractionConfig(
                llm_extract=LLMExtractConfig(enabled=False),
            ),
        )

    @classmethod
    def with_decisions(cls, workspace_id: Optional[str] = None) -> "PipelineConfig":
        """Default pipeline with automatic decision extraction enabled (CORE-SYS2-1b).

        Same as default() but with extract_decisions=True on the LLM stage.
        Decisions above the confidence threshold (0.75) are stored automatically.
        """
        config = cls.default(workspace_id=workspace_id)
        config.extraction.llm_extract.extract_decisions = True
        return config

    @classmethod
    def with_reasoning(cls, workspace_id: Optional[str] = None) -> "PipelineConfig":
        """Default pipeline with reasoning detection enabled (CORE-SYS2-1c).

        Same as default() but with reasoning_detect.enabled=True.
        Reasoning traces above the quality threshold (0.4) are stored automatically.

        Note: This factory is intended for standalone use or testing, NOT as a
        ``pipeline_profile=`` constructor argument. Profile application copies only
        4 fields and does not propagate reasoning_detect. To enable reasoning in
        production, pass ``extract_reasoning=True`` to ``ingest()`` instead.
        """
        config = cls.default(workspace_id=workspace_id)
        config.extraction.reasoning_detect.enabled = True
        return config


# ------------------------------------------------------------------ #
# OntologyConfig (loaded from graph, not part of PipelineConfig)
# ------------------------------------------------------------------ #


@dataclass
class OntologyConfig(MemoryBaseModel):
    """Snapshot of known entity types loaded from the ontology graph."""

    entity_types: Dict[str, Any] = field(default_factory=dict)
