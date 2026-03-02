"""Tests for PipelineConfig.lite() classmethod."""

from smartmemory.pipeline.config import PipelineConfig


def test_lite_coreference_disabled():
    """PipelineConfig.lite() disables coreference resolution."""
    config = PipelineConfig.lite()
    assert config.coreference.enabled is False, "lite mode must disable coreference (may download models)"


def test_lite_llm_extract_disabled():
    """PipelineConfig.lite() disables LLM entity/relation extraction."""
    config = PipelineConfig.lite()
    assert config.extraction.llm_extract.enabled is False, "lite mode must disable llm_extract (makes LLM API calls)"


def test_lite_enrichers_basic_only():
    """PipelineConfig.lite() restricts enrichers to local-only (no HTTP enrichers)."""
    config = PipelineConfig.lite()
    assert config.enrich.enricher_names == [
        "basic_enricher",
        "sentiment_enricher",
        "temporal_enricher",
        "topic_enricher",
    ], "lite mode must exclude HTTP enrichers (wikipedia, link_expansion) but keep local ones"


def test_lite_wikidata_enabled_with_sparql_disabled():
    """PipelineConfig.lite() keeps wikidata enabled but disables SPARQL HTTP."""
    config = PipelineConfig.lite()
    assert config.enrich.wikidata.enabled is True, (
        "lite mode keeps wikidata.enabled=True so GroundStage runs (SQLite alias lookup)"
    )
    assert config.enrich.wikidata.sparql_enabled is False, "lite mode must disable sparql_enabled (no HTTP calls)"


def test_lite_evolution_disabled():
    """PipelineConfig.lite() disables evolution (HebbianCoRetrievalEvolver uses Cypher)."""
    config = PipelineConfig.lite()
    assert config.evolve.run_evolution is False, "lite mode must disable evolution (blocked evolver uses raw Cypher)"


def test_lite_clustering_disabled():
    """PipelineConfig.lite() disables clustering."""
    config = PipelineConfig.lite()
    assert config.evolve.run_clustering is False, "lite mode must disable clustering"


def test_lite_accepts_workspace_id():
    """PipelineConfig.lite() propagates workspace_id."""
    config = PipelineConfig.lite(workspace_id="ws1")
    assert config.workspace_id == "ws1"
