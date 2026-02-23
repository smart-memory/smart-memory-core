"""Tests for the 4 new SmartMemory constructor parameters added by DIST-LITE-2.

All tests use heavy mocking to avoid requiring live infrastructure.
"""
import os
import pytest
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


def test_observability_false_sets_env_var():
    """SmartMemory(observability=False) sets SMARTMEMORY_OBSERVABILITY=false in env."""
    old = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        with ExitStack() as stack:
            mem = _make_sm(stack, observability=False)
            assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "false", (
                "observability=False must set SMARTMEMORY_OBSERVABILITY=false"
            )
            assert mem._observability is False
    finally:
        if old is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old


def test_observability_true_restores_env_after_false():
    """SmartMemory(observability=True) restores the pre-disable env value, not just pops it."""
    import smartmemory.smart_memory as _sm_mod
    old_saved = _sm_mod._obs_pre_disable_value
    old_env = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        # Case A: no env var before — constructor set env=false, restore=pop.
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        _sm_mod._obs_pre_disable_value = _sm_mod._SENTINEL
        with ExitStack() as stack:
            _make_sm(stack, observability=False)
        assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "false"
        with ExitStack() as stack:
            _make_sm(stack, observability=True)
        assert "SMARTMEMORY_OBSERVABILITY" not in os.environ, (
            "observability=True must pop env when it was absent before the disable"
        )

        # Case B: user had SMARTMEMORY_OBSERVABILITY=false before — must be restored, not cleared.
        os.environ["SMARTMEMORY_OBSERVABILITY"] = "false"
        _sm_mod._obs_pre_disable_value = _sm_mod._SENTINEL
        with ExitStack() as stack:
            _make_sm(stack, observability=False)
        with ExitStack() as stack:
            _make_sm(stack, observability=True)
        assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "false", (
            "observability=True must restore 'false' when user had it set before the disable"
        )
    finally:
        _sm_mod._obs_pre_disable_value = old_saved
        if old_env is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old_env


def test_observability_restore_is_reentrant():
    """Multiple observability=False calls must not clobber the original env value.

    Repro sequence:
        env = "true"
        SmartMemory(observability=False)  # saves "true", sets "false"
        SmartMemory(observability=False)  # must NOT overwrite save slot with "false"
        SmartMemory(observability=True)   # must restore "true", not "false"
    """
    import smartmemory.smart_memory as _sm_mod
    old_saved = _sm_mod._obs_pre_disable_value
    old_env = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    try:
        os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
        _sm_mod._obs_pre_disable_value = _sm_mod._SENTINEL

        with ExitStack() as stack:
            _make_sm(stack, observability=False)   # saves "true"
        with ExitStack() as stack:
            _make_sm(stack, observability=False)   # must NOT overwrite save slot
        with ExitStack() as stack:
            _make_sm(stack, observability=True)    # restore

        assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "true", (
            "observability=True must restore the original env value even after "
            "multiple consecutive observability=False constructions"
        )
    finally:
        _sm_mod._obs_pre_disable_value = old_saved
        if old_env is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = old_env


# ---------------------------------------------------------------------------
# vector_backend
# ---------------------------------------------------------------------------

def test_vector_backend_calls_set_default():
    """SmartMemory(vector_backend=x) calls VectorStore.set_default_backend(x)."""
    sentinel = object()
    with ExitStack() as stack:
        mock_set = stack.enter_context(
            patch("smartmemory.stores.vector.vector_store.VectorStore.set_default_backend")
        )
        mem = _make_sm(stack, vector_backend=sentinel)
        mock_set.assert_called_once_with(sentinel)
        assert mem._vector_backend is sentinel


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_calls_set_cache_override():
    """SmartMemory(cache=x) calls set_cache_override(x)."""
    from smartmemory.utils.cache import NoOpCache

    noop = NoOpCache()
    with ExitStack() as stack:
        mock_override = stack.enter_context(patch("smartmemory.utils.cache.set_cache_override"))
        mem = _make_sm(stack, cache=noop)
        mock_override.assert_called_once_with(noop)
        assert mem._cache is noop


# ---------------------------------------------------------------------------
# pipeline_profile applied in _build_pipeline_config
# ---------------------------------------------------------------------------

def test_pipeline_profile_applied_in_build():
    """SmartMemory(pipeline_profile=PipelineConfig.lite()) applies all 4 lite flags."""
    from smartmemory.pipeline.config import PipelineConfig

    profile = PipelineConfig.lite()
    with ExitStack() as stack:
        mem = _make_sm(stack, pipeline_profile=profile)
        config = mem._build_pipeline_config()
        assert config.coreference.enabled is False
        assert config.extraction.llm_extract.enabled is False
        assert config.enrich.enricher_names == ["basic_enricher"]
        assert config.enrich.wikidata.enabled is False


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
