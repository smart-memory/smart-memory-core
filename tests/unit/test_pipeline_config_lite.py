"""Tests for PipelineConfig.lite() classmethod."""
from smartmemory.pipeline.config import PipelineConfig


def test_lite_coreference_disabled():
    """PipelineConfig.lite() disables coreference resolution."""
    config = PipelineConfig.lite()
    assert config.coreference.enabled is False, (
        "lite mode must disable coreference (may download models)"
    )


def test_lite_llm_extract_disabled():
    """PipelineConfig.lite() disables LLM entity/relation extraction."""
    config = PipelineConfig.lite()
    assert config.extraction.llm_extract.enabled is False, (
        "lite mode must disable llm_extract (makes LLM API calls)"
    )


def test_lite_enrichers_basic_only():
    """PipelineConfig.lite() restricts enrichers to basic_enricher only."""
    config = PipelineConfig.lite()
    assert config.enrich.enricher_names == ["basic_enricher"], (
        "lite mode must only run basic_enricher — no HTTP enrichers"
    )


def test_lite_wikidata_disabled():
    """PipelineConfig.lite() disables Wikidata/Wikipedia grounding."""
    config = PipelineConfig.lite()
    assert config.enrich.wikidata.enabled is False, (
        "lite mode must disable wikidata grounding (makes HTTP calls)"
    )


def test_lite_accepts_workspace_id():
    """PipelineConfig.lite() propagates workspace_id."""
    config = PipelineConfig.lite(workspace_id="ws1")
    assert config.workspace_id == "ws1"
