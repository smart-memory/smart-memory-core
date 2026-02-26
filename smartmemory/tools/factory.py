"""smartmemory.tools.factory — zero-infra SmartMemory factory for Lite."""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


def _default_data_dir() -> Path:
    return Path.home() / ".smartmemory"


def _ensure_spacy_model(model: str = "en_core_web_sm") -> None:
    """Auto-download spaCy model on first use if not already installed.

    Uses rich for progress output when available (pip install smartmemory[lite]).
    Falls back to plain print on a base install without rich.
    """
    import spacy
    if spacy.util.is_package(model):
        return
    try:
        from rich.console import Console
        _print = Console().print
    except ImportError:
        _print = print  # type: ignore[assignment]
    _print(f"Downloading spaCy model '{model}' (first run only)...")
    import spacy.cli
    spacy.cli.download(model)
    _print(f"spaCy model '{model}' ready.")


def create_lite_memory(
    data_dir: Optional[str] = None,
    entity_ruler_patterns=None,
    pipeline_profile=None,
):
    """Create a SmartMemory instance backed by SQLite + usearch. No Docker required.

    Lite mode refers to the storage layer only: SQLite replaces FalkorDB, usearch
    replaces the FalkorDB vector index, Redis cache is a no-op, and observability
    events are disabled. The pipeline runs at full quality by default — LLM extraction
    and enrichers are active if API keys are available.

    To opt into a restricted pipeline (no LLM calls, no network enrichers), pass
    ``pipeline_profile=PipelineConfig.lite()`` explicitly.

    Args:
        data_dir: Directory for SQLite and usearch persistence. Defaults to ~/.smartmemory.
        entity_ruler_patterns: Optional pattern manager injected into EntityRulerStage.
        pipeline_profile: PipelineConfig to use. Defaults to PipelineConfig.default()
            (full pipeline). Pass PipelineConfig.lite() to disable LLM extraction and
            network enrichers.
    """
    _ensure_spacy_model()
    from smartmemory.graph.backends.sqlite import SQLiteBackend
    from smartmemory.graph.smartgraph import SmartGraph
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.smart_memory import SmartMemory
    from smartmemory.stores.vector.backends.usearch import UsearchVectorBackend
    from smartmemory.utils.cache import NoOpCache

    if pipeline_profile is None:
        pipeline_profile = PipelineConfig.lite()

    data_path = Path(data_dir).expanduser() if data_dir else _default_data_dir()
    data_path.mkdir(parents=True, exist_ok=True)

    sqlite_backend = SQLiteBackend(db_path=str(data_path / "memory.db"))
    usearch_backend = UsearchVectorBackend(
        collection_name="memory",
        persist_directory=str(data_path),
    )
    graph = SmartGraph(backend=sqlite_backend)

    return SmartMemory(
        graph=graph,
        enable_ontology=False,
        vector_backend=usearch_backend,
        cache=NoOpCache(),
        observability=False,
        pipeline_profile=pipeline_profile,
        entity_ruler_patterns=entity_ruler_patterns,
    )


@contextmanager
def lite_context(data_dir: Optional[str] = None, pipeline_profile=None):
    """Context manager that creates a Lite SmartMemory and resets all globals on exit.

    Restores observability env, vector backend, cache override, and closes the SQLite
    connection deterministically. Always use this in tests and scripts:
        with lite_context() as memory:
            memory.ingest("hello")

    Args:
        data_dir: Directory for SQLite and usearch persistence. Defaults to ~/.smartmemory.
        pipeline_profile: PipelineConfig to use. Defaults to PipelineConfig.default()
            (full pipeline). Pass PipelineConfig.lite() to disable LLM extraction and
            network enrichers.
    """
    from smartmemory.stores.vector.vector_store import VectorStore
    from smartmemory.utils.cache import set_cache_override

    # Capture observability env BEFORE any global mutation so we can restore it
    # unconditionally — even if create_lite_memory() raises partway through.
    prev_obs = os.environ.get("SMARTMEMORY_OBSERVABILITY")

    memory = None
    try:
        memory = create_lite_memory(data_dir, pipeline_profile=pipeline_profile)
        yield memory
    finally:
        # Restore globals regardless of whether construction, yield, or body raised.
        VectorStore.set_default_backend(None)
        set_cache_override(None)

        # Restore observability env to its pre-context value.
        if prev_obs is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = prev_obs

        # Close the SQLite backend explicitly — don't rely on GC for WAL flush.
        if memory is not None:
            try:
                memory._graph.backend.close()
            except Exception:
                pass
