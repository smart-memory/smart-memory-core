"""Tests for the 4 new SmartMemory constructor parameters added by DIST-LITE-2.

All tests use heavy mocking to avoid requiring live infrastructure.
"""

import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared SmartMemory mock factory
# ---------------------------------------------------------------------------


def _make_mock_graph():
    """Return a minimal SmartGraph mock so SmartMemory.__init__ can run."""
    g = MagicMock()
    g.search = MagicMock()
    g.search.set_smart_memory = MagicMock()
    return g


# ---------------------------------------------------------------------------
# observability=False
# ---------------------------------------------------------------------------

_CONSTRUCTOR_PATCHES = [
    "smartmemory.smart_memory.EvolutionOrchestrator",
    "smartmemory.smart_memory.GlobalClustering",
    "smartmemory.smart_memory.VersionTracker",
    "smartmemory.smart_memory.TemporalQueries",
    "smartmemory.smart_memory.ProcedureMatcher",
    "smartmemory.smart_memory.DriftDetector",
]


def _make_sm(stack, **kwargs):
    """Construct SmartMemory with all heavy deps mocked via the given ExitStack."""
    for target in _CONSTRUCTOR_PATCHES:
        stack.enter_context(patch(target))
    from smartmemory.smart_memory import SmartMemory

    return SmartMemory(graph=_make_mock_graph(), **kwargs)


def test_observability_false_no_env_mutation():
    """SmartMemory(observability=False) stores False in _observability; os.environ NOT mutated at construction."""
    old = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        with ExitStack() as stack:
            mem = _make_sm(stack, observability=False)
            # Constructor must NOT write to os.environ — only _di_context() does
            assert os.environ.get("SMARTMEMORY_OBSERVABILITY") is None, (
                "SmartMemory constructor must NOT write to os.environ"
            )
            assert mem._observability is False

        # Inside _di_context(), _observability_ctx is False; outside, it resets to None
        from smartmemory.observability.tracing import _observability_ctx

        with mem._di_context():
            assert _observability_ctx.get() is False
        assert _observability_ctx.get() is None, "_observability_ctx must reset to None after _di_context exits"
    finally:
        if old is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old


def test_observability_ctx_resets_on_exit():
    """_di_context() resets _observability_ctx to None after the context exits (normal and exception)."""
    from smartmemory.observability.tracing import _observability_ctx

    with ExitStack() as stack:
        mem = _make_sm(stack, observability=False)

    # Normal exit
    with mem._di_context():
        assert _observability_ctx.get() is False
    assert _observability_ctx.get() is None, "_observability_ctx must reset to None after _di_context exits cleanly"

    # Exception path — ContextVar must still reset
    try:
        with mem._di_context():
            assert _observability_ctx.get() is False
            raise ValueError("synthetic")
    except ValueError:
        pass
    assert _observability_ctx.get() is None, (
        "_observability_ctx must reset to None after _di_context exits via exception"
    )


def test_observability_ctx_isolation_across_instances():
    """Two SmartMemory instances with different observability see independent ContextVar values inside _di_context."""
    from smartmemory.observability.tracing import _observability_ctx

    with ExitStack() as stack:
        mem_off = _make_sm(stack, observability=False)

    # Outside any _di_context, the ContextVar is None regardless of instance config
    assert _observability_ctx.get() is None

    # Inside mem_off's _di_context, observability is disabled
    with mem_off._di_context():
        assert _observability_ctx.get() is False

    # After exit, it resets
    assert _observability_ctx.get() is None

    # Nested _di_context calls reset in LIFO order (the token pattern)
    with ExitStack() as stack2:
        mem_off2 = _make_sm(stack2, observability=False)
    with mem_off._di_context():
        inner_val = _observability_ctx.get()
        assert inner_val is False
    assert _observability_ctx.get() is None


# ---------------------------------------------------------------------------
# vector_backend
# ---------------------------------------------------------------------------


def test_vector_backend_stored_not_global():
    """SmartMemory(vector_backend=x) stores x in _vector_backend; set_default_backend is NOT called."""
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx

    sentinel = object()
    with patch("smartmemory.stores.vector.vector_store.VectorStore.set_default_backend") as mock_set:
        with ExitStack() as stack:
            mem = _make_sm(stack, vector_backend=sentinel)
        mock_set.assert_not_called()

    assert mem._vector_backend is sentinel

    # _vector_backend_ctx is None outside _di_context, holds sentinel inside
    assert _vector_backend_ctx.get() is None
    with mem._di_context():
        assert _vector_backend_ctx.get() is sentinel
    assert _vector_backend_ctx.get() is None


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_stored_not_global():
    """SmartMemory(cache=x) stores x in _cache; set_cache_override is NOT called."""
    from smartmemory.utils.cache import NoOpCache, _cache_ctx

    noop = NoOpCache()
    with patch("smartmemory.utils.cache.set_cache_override") as mock_override:
        with ExitStack() as stack:
            mem = _make_sm(stack, cache=noop)
        mock_override.assert_not_called()

    assert mem._cache is noop

    # _cache_ctx is None outside _di_context, holds noop inside
    assert _cache_ctx.get() is None
    with mem._di_context():
        assert _cache_ctx.get() is noop
    assert _cache_ctx.get() is None


# ---------------------------------------------------------------------------
# pipeline_profile applied in _build_pipeline_config
# ---------------------------------------------------------------------------


def test_pipeline_profile_applied_in_build():
    """SmartMemory(pipeline_profile=PipelineConfig.lite()) applies all 7 lite flags."""
    from smartmemory.pipeline.config import PipelineConfig

    profile = PipelineConfig.lite(llm_enabled=False)
    with ExitStack() as stack:
        mem = _make_sm(stack, pipeline_profile=profile)
        config = mem._build_pipeline_config()
        assert config.coreference.enabled is False
        assert config.extraction.llm_extract.enabled is False  # llm_enabled=False overrides auto-detect
        assert config.enrich.enricher_names == [
            "basic_enricher",
            "sentiment_enricher",
            "temporal_enricher",
            "topic_enricher",
        ]
        # 1c: wikidata enabled but sparql disabled
        assert config.enrich.wikidata.enabled is True
        assert config.enrich.wikidata.sparql_enabled is False
        # 1e: CORE-EVO-LIVE-1 — evolution enabled (incremental worker handles it), clustering disabled
        assert config.evolve.run_evolution is True
        assert config.evolve.run_clustering is False


def test_profile_propagates_sparql_enabled():
    """_apply_pipeline_profile propagates sparql_enabled=False from lite profile."""
    from smartmemory.pipeline.config import PipelineConfig

    with ExitStack() as stack:
        mem = _make_sm(stack, pipeline_profile=PipelineConfig.lite())
        config = mem._build_pipeline_config()
        assert config.enrich.wikidata.sparql_enabled is False


def test_profile_propagates_evolution_enabled():
    """CORE-EVO-LIVE-1: _apply_pipeline_profile propagates run_evolution=True from lite profile."""
    from smartmemory.pipeline.config import PipelineConfig

    with ExitStack() as stack:
        mem = _make_sm(stack, pipeline_profile=PipelineConfig.lite())
        config = mem._build_pipeline_config()
        assert config.evolve.run_evolution is True


def test_profile_propagates_clustering_disabled():
    """_apply_pipeline_profile propagates run_clustering=False from lite profile."""
    from smartmemory.pipeline.config import PipelineConfig

    with ExitStack() as stack:
        mem = _make_sm(stack, pipeline_profile=PipelineConfig.lite())
        config = mem._build_pipeline_config()
        assert config.evolve.run_clustering is False


# ---------------------------------------------------------------------------
# defaults unchanged
# ---------------------------------------------------------------------------


def test_defaults_unchanged():
    """SmartMemory() without new params leaves all 4 new attrs at None/True."""
    with ExitStack() as stack:
        mem = _make_sm(stack)
        assert mem._observability is True
        assert mem._pipeline_profile is None
        assert mem._vector_backend is None
        assert mem._cache is None
