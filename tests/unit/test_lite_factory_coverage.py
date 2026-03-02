"""Additional coverage tests for smartmemory.tools.factory.

Focuses on:
- lite_context() ContextVar state does not leak outside the context (CORE-DI-1)
- lite_context() closes the SQLite backend on normal exit and exception
- create_lite_memory creates the data directory if it doesn't exist
- create_lite_memory respects custom data_dir
- SmartMemory constructor (via create_lite_memory) sets coreference.enabled=False,
  llm_extract.enabled=False, and enricher_names=["basic_enricher"] via pipeline_profile
- SmartMemory constructor (via create_lite_memory) stores NoOpCache in _cache without global mutation
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# Ensure the factory module can be imported (requires smartmemory_lite installed)
try:
    from smartmemory.tools.factory import create_lite_memory, lite_context

    _LITE_AVAILABLE = True
except ImportError:
    _LITE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _LITE_AVAILABLE, reason="smartmemory-lite not installed")


# ── lite_context ContextVar isolation contract (CORE-DI-1) ───────────────────


def test_lite_context_vector_backend_ctx_cleared_on_exit(tmp_path):
    """lite_context() does NOT call set_default_backend(None); _vector_backend_ctx is None outside."""
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx

    with patch("smartmemory.stores.vector.vector_store.VectorStore.set_default_backend") as mock_set:
        with lite_context(str(tmp_path)) as memory:
            assert memory is not None
        mock_set.assert_not_called()

    assert _vector_backend_ctx.get() is None, "_vector_backend_ctx must not leak outside lite_context"


def test_lite_context_vector_backend_ctx_cleared_on_exception(tmp_path):
    """_vector_backend_ctx is None outside lite_context even when body raises."""
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx

    with pytest.raises(RuntimeError, match="intentional"):
        with lite_context(str(tmp_path)):
            raise RuntimeError("intentional test error")

    assert _vector_backend_ctx.get() is None, "_vector_backend_ctx must not leak even on exception"


# ── create_lite_memory creates data directory ─────────────────────────────────


def test_create_lite_memory_creates_data_dir(tmp_path):
    """create_lite_memory() creates the data directory if it doesn't exist."""
    from smartmemory.stores.vector.vector_store import VectorStore

    nested_dir = tmp_path / "nested" / "subdir"
    assert not nested_dir.exists()
    try:
        memory = create_lite_memory(str(nested_dir))
        assert nested_dir.exists(), "data directory should be created"
    finally:
        VectorStore.set_default_backend(None)


def test_create_lite_memory_creates_db_file(tmp_path):
    """create_lite_memory() creates memory.db inside data_dir."""
    from smartmemory.stores.vector.vector_store import VectorStore

    try:
        memory = create_lite_memory(str(tmp_path))
        db_path = tmp_path / "memory.db"
        assert db_path.exists(), f"Expected memory.db at {db_path}"
    finally:
        VectorStore.set_default_backend(None)


# ── _apply_lite_pipeline_profile correctness ──────────────────────────────────


def test_apply_lite_pipeline_profile_disables_coreference(tmp_path):
    """The patched _build_pipeline_config sets coreference.enabled=False."""
    from smartmemory.stores.vector.vector_store import VectorStore

    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.coreference.enabled is False, "coreference must be disabled in lite pipeline profile"
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_disables_llm_extract(tmp_path):
    """The patched _build_pipeline_config disables llm_extract when no API key (DEGRADE-1d)."""
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.stores.vector.vector_store import VectorStore

    try:
        # DEGRADE-1d: lite() auto-detects API keys; force off for deterministic test
        memory = create_lite_memory(str(tmp_path), pipeline_profile=PipelineConfig.lite(llm_enabled=False))
        config = memory._build_pipeline_config()
        assert config.extraction.llm_extract.enabled is False, "llm_extract must be disabled when llm_enabled=False"
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_limits_enrichers(tmp_path):
    """The patched _build_pipeline_config limits enrichers to local-only enrichers (no HTTP)."""
    from smartmemory.stores.vector.vector_store import VectorStore

    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        # Lite mode runs all local enrichers; HTTP-dependent ones (wikipedia, link_expansion) are excluded.
        assert config.enrich.enricher_names == [
            "basic_enricher",
            "sentiment_enricher",
            "temporal_enricher",
            "topic_enricher",
        ], "lite mode must exclude HTTP enrichers (wikipedia, link_expansion) but keep local ones"
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_disables_wikidata(tmp_path):
    """The patched _build_pipeline_config keeps wikidata enabled but disables SPARQL HTTP."""
    from smartmemory.stores.vector.vector_store import VectorStore

    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        # DEGRADE-1c: wikidata.enabled stays True (SQLite alias lookup still works),
        # but sparql_enabled is False (no HTTP calls to Wikidata SPARQL endpoint).
        assert config.enrich.wikidata.enabled is True, (
            "wikidata must stay enabled in lite mode (SQLite alias lookup works without HTTP)"
        )
        assert config.enrich.wikidata.sparql_enabled is False, "SPARQL must be disabled in lite mode (no HTTP calls)"
    finally:
        VectorStore.set_default_backend(None)


# ── cache override set by create_lite_memory ─────────────────────────────────


def test_create_lite_memory_uses_noop_cache(tmp_path):
    """create_lite_memory() wires a NoOpCache via SmartMemory(cache=...) constructor."""
    from smartmemory.utils.cache import NoOpCache

    memory = create_lite_memory(str(tmp_path))
    # Constructor stores NoOpCache in _cache without calling set_cache_override (CORE-DI-1)
    assert isinstance(memory._cache, NoOpCache), (
        "create_lite_memory must pass a NoOpCache to SmartMemory(cache=...) constructor"
    )
    assert memory._cache is not None, "_cache attribute must be set"


# ── lite_context observability restore (P2) ───────────────────────────────────


def test_lite_context_restores_observability_env_on_exit(tmp_path):
    """lite_context() restores SMARTMEMORY_OBSERVABILITY to its pre-context value on exit."""
    # Set a known value before entering the context
    os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
    try:
        with lite_context(str(tmp_path)):
            pass  # create_lite_memory sets observability=False here
        assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "true", (
            "lite_context must restore observability env to 'true' on exit"
        )
    finally:
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)


def test_lite_context_restores_observability_env_even_on_exception(tmp_path):
    """lite_context() restores observability env even when the body raises."""
    os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
    try:
        with pytest.raises(RuntimeError, match="intentional"):
            with lite_context(str(tmp_path)):
                raise RuntimeError("intentional test error")
        assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "true", (
            "lite_context must restore observability env on exception"
        )
    finally:
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)


def test_lite_context_removes_observability_env_if_unset_before(tmp_path):
    """lite_context() removes SMARTMEMORY_OBSERVABILITY if it wasn't set before entering."""
    os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
    try:
        with lite_context(str(tmp_path)):
            pass
        assert "SMARTMEMORY_OBSERVABILITY" not in os.environ, (
            "lite_context must remove the env var if it wasn't present before entering"
        )
    finally:
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)


# ── lite_context SQLite backend close (P3) ────────────────────────────────────


def test_lite_context_closes_sqlite_backend_on_exit(tmp_path):
    """lite_context() calls close() on the SQLite backend in the finally block."""

    with lite_context(str(tmp_path)) as memory:
        backend = memory._graph.backend

    # After the context, close() should have been called (SQLiteBackend tracks this).
    # We verify by checking the connection is closed (execute raises after close).
    with pytest.raises(Exception):
        backend._conn.execute("SELECT 1")


def test_create_lite_memory_passes_entity_ruler_patterns(tmp_path):
    """create_lite_memory() forwards entity_ruler_patterns to SmartMemory._entity_ruler_patterns."""
    from smartmemory.stores.vector.vector_store import VectorStore

    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}

    try:
        memory = create_lite_memory(str(tmp_path), entity_ruler_patterns=mock_pm)
        assert memory._entity_ruler_patterns is mock_pm, (
            "create_lite_memory must pass entity_ruler_patterns to SmartMemory constructor"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_lite_context_closes_sqlite_backend_on_exception(tmp_path):
    """lite_context() closes the SQLite backend even when the body raises."""
    with pytest.raises(RuntimeError, match="intentional"):
        with lite_context(str(tmp_path)) as memory:
            backend = memory._graph.backend
            raise RuntimeError("intentional test error")

    with pytest.raises(Exception):
        backend._conn.execute("SELECT 1")


# ── lite_context cleanup on construction failure (P2) ────────────────────────


def test_lite_context_no_ctx_leak_if_create_lite_memory_raises(tmp_path):
    """lite_context() leaves no ContextVar values set when create_lite_memory() raises (memory is None)."""
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx
    from smartmemory.utils.cache import _cache_ctx
    from smartmemory.observability.tracing import _observability_ctx

    with patch(
        "smartmemory.tools.factory.create_lite_memory",
        side_effect=RuntimeError("construction failed"),
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            with lite_context(str(tmp_path)):
                pass  # never reached

    # No ContextVars must have been set (memory is None, _di_context never called)
    assert _vector_backend_ctx.get() is None
    assert _cache_ctx.get() is None
    assert _observability_ctx.get() is None
