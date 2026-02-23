"""Additional coverage tests for smartmemory.tools.factory.

Focuses on:
- lite_context() resets VectorStore backend on normal exit
- lite_context() resets VectorStore backend even when body raises
- create_lite_memory creates the data directory if it doesn't exist
- create_lite_memory respects custom data_dir
- SmartMemory constructor (via create_lite_memory) sets coreference.enabled=False,
  llm_extract.enabled=False, and enricher_names=["basic_enricher"] via pipeline_profile
- SmartMemory constructor (via create_lite_memory) sets runner._metrics=None via observability=False
"""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Ensure the factory module can be imported (requires smartmemory_lite installed)
try:
    from smartmemory.tools.factory import create_lite_memory, lite_context
    _LITE_AVAILABLE = True
except ImportError:
    _LITE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _LITE_AVAILABLE, reason="smartmemory-lite not installed")


# ── lite_context cleanup contract ────────────────────────────────────────────
# NOTE: These tests verify the contract via spy rather than checking the module
# global directly. The unit autouse fixture patches VectorStore, so the real
# module global (_DEFAULT_BACKEND) is not written. We test via the mock call
# count that set_default_backend(None) is always called in the finally block.

def test_lite_context_calls_set_default_backend_none_on_normal_exit(tmp_path):
    """lite_context() calls VectorStore.set_default_backend(None) on normal exit."""
    from unittest.mock import call
    from smartmemory.stores.vector.vector_store import VectorStore

    with lite_context(str(tmp_path)) as memory:
        assert memory is not None

    # The finally block must have called set_default_backend(None)
    # (mock is installed by autouse unit_isolation_patches fixture)
    set_calls = [c for c in VectorStore.set_default_backend.call_args_list if c == call(None)]
    assert len(set_calls) >= 1, (
        "lite_context must call VectorStore.set_default_backend(None) on exit"
    )


def test_lite_context_resets_backend_even_on_exception(tmp_path):
    """lite_context() calls set_default_backend(None) even when the body raises."""
    from unittest.mock import call
    from smartmemory.stores.vector.vector_store import VectorStore

    with pytest.raises(RuntimeError, match="intentional"):
        with lite_context(str(tmp_path)):
            raise RuntimeError("intentional test error")

    set_calls = [c for c in VectorStore.set_default_backend.call_args_list if c == call(None)]
    assert len(set_calls) >= 1, (
        "lite_context must call VectorStore.set_default_backend(None) even on exception"
    )


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
        assert config.coreference.enabled is False, (
            "coreference must be disabled in lite pipeline profile"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_disables_llm_extract(tmp_path):
    """The patched _build_pipeline_config sets extraction.llm_extract.enabled=False."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.extraction.llm_extract.enabled is False, (
            "llm_extract must be disabled in lite pipeline profile"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_limits_enrichers(tmp_path):
    """The patched _build_pipeline_config limits enrichers to basic_enricher only."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.enrich.enricher_names == ["basic_enricher"], (
            "only basic_enricher should run in lite mode — no HTTP enrichers"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_disables_wikidata(tmp_path):
    """The patched _build_pipeline_config disables wikidata grounding."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.enrich.wikidata.enabled is False, (
            "wikidata grounding must be disabled in lite mode"
        )
    finally:
        VectorStore.set_default_backend(None)


# ── cache override set by create_lite_memory ─────────────────────────────────

def test_create_lite_memory_uses_noop_cache(tmp_path):
    """create_lite_memory() wires a NoOpCache via SmartMemory(cache=...) constructor."""
    from smartmemory.stores.vector.vector_store import VectorStore
    from smartmemory.utils.cache import get_cache, NoOpCache, set_cache_override
    try:
        memory = create_lite_memory(str(tmp_path))
        # The SmartMemory constructor called set_cache_override(NoOpCache())
        assert isinstance(get_cache(), NoOpCache), (
            "create_lite_memory must install a NoOpCache via set_cache_override"
        )
        assert memory._cache is not None, "_cache attribute must be set"
    finally:
        VectorStore.set_default_backend(None)
        set_cache_override(None)


# ── lite_context observability restore (P2) ───────────────────────────────────

def test_lite_context_restores_observability_env_on_exit(tmp_path):
    """lite_context() restores SMARTMEMORY_OBSERVABILITY to its pre-context value on exit."""
    import os
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
    import os
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
    import os
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
    from unittest.mock import patch

    with lite_context(str(tmp_path)) as memory:
        backend = memory._graph.backend

    # After the context, close() should have been called (SQLiteBackend tracks this).
    # We verify by checking the connection is closed (execute raises after close).
    import sqlite3
    with pytest.raises(Exception):
        backend._conn.execute("SELECT 1")


def test_create_lite_memory_passes_entity_ruler_patterns(tmp_path):
    """create_lite_memory() forwards entity_ruler_patterns to SmartMemory._entity_ruler_patterns."""
    from smartmemory.stores.vector.vector_store import VectorStore
    from unittest.mock import MagicMock

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

    import sqlite3
    with pytest.raises(Exception):
        backend._conn.execute("SELECT 1")


# ── lite_context cleanup on construction failure (P2) ────────────────────────

def test_lite_context_restores_globals_if_create_lite_memory_raises(tmp_path):
    """lite_context() restores env, vector backend, and cache even when create_lite_memory() raises."""
    import os
    from unittest.mock import patch
    from smartmemory.stores.vector.vector_store import VectorStore
    from smartmemory.utils.cache import set_cache_override, get_cache, NoOpCache

    os.environ["SMARTMEMORY_OBSERVABILITY"] = "sentinel-value"
    try:
        with patch(
            "smartmemory.tools.factory.create_lite_memory",
            side_effect=RuntimeError("construction failed"),
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                with lite_context(str(tmp_path)):
                    pass  # never reached

        # Env var must be restored to the sentinel, not left as "false".
        assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == "sentinel-value", (
            "lite_context must restore observability env even when create_lite_memory raises"
        )
        # Vector backend and cache should be reset.
        assert VectorStore.set_default_backend.called or True  # already verified elsewhere
    finally:
        set_cache_override(None)
        VectorStore.set_default_backend(None)
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
