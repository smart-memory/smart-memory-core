"""Tests for PipelineConfig.lite() classmethod."""

from unittest.mock import patch

from smartmemory.pipeline.config import PipelineConfig


def test_lite_coreference_disabled():
    """PipelineConfig.lite() disables coreference resolution."""
    config = PipelineConfig.lite()
    assert config.coreference.enabled is False, "lite mode must disable coreference (may download models)"


def test_lite_llm_extract_disabled_when_forced():
    """PipelineConfig.lite(llm_enabled=False) disables LLM extraction."""
    config = PipelineConfig.lite(llm_enabled=False)
    assert config.extraction.llm_extract.enabled is False


def test_lite_llm_extract_enabled_when_forced():
    """PipelineConfig.lite(llm_enabled=True) enables LLM extraction."""
    config = PipelineConfig.lite(llm_enabled=True)
    assert config.extraction.llm_extract.enabled is True


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
def test_lite_llm_autodetect_openai():
    """PipelineConfig.lite() auto-detects OPENAI_API_KEY and enables LLM extraction."""
    config = PipelineConfig.lite()
    assert config.extraction.llm_extract.enabled is True


@patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}, clear=False)
def test_lite_llm_autodetect_groq():
    """PipelineConfig.lite() auto-detects GROQ_API_KEY and enables LLM extraction."""
    config = PipelineConfig.lite()
    assert config.extraction.llm_extract.enabled is True


@patch.dict("os.environ", {}, clear=True)
def test_lite_llm_autodetect_no_key():
    """PipelineConfig.lite() disables LLM extraction when no API key is set."""
    config = PipelineConfig.lite()
    assert config.extraction.llm_extract.enabled is False


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


def test_lite_evolution_enabled():
    """CORE-EVO-LIVE-1: PipelineConfig.lite() enables evolution (incremental worker handles it)."""
    config = PipelineConfig.lite()
    assert config.evolve.run_evolution is True, (
        "lite mode must enable evolution — CORE-EVO-LIVE-1 provides incremental "
        "evolution via daemon thread; EvolveStage skips batch when worker is active"
    )


def test_lite_clustering_disabled():
    """PipelineConfig.lite() disables clustering."""
    config = PipelineConfig.lite()
    assert config.evolve.run_clustering is False, "lite mode must disable clustering"


def test_lite_accepts_workspace_id():
    """PipelineConfig.lite() propagates workspace_id."""
    config = PipelineConfig.lite(workspace_id="ws1")
    assert config.workspace_id == "ws1"
