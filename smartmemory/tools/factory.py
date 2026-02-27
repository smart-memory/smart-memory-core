"""smartmemory.tools.factory — zero-infra SmartMemory factory for Lite."""

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
    event_sink=None,  # DIST-LITE-3: InProcessQueueSink or None
):
    """Create a SmartMemory instance backed by SQLite + usearch. No Docker required.

    Lite mode replaces FalkorDB with SQLite, the FalkorDB vector index with usearch,
    Redis cache with a no-op, and disables observability event emission. The pipeline
    defaults to ``PipelineConfig.lite()`` — EntityRuler + local enrichers only, no LLM
    calls and no network enrichers.

    To run the full pipeline (LLM extraction, network enrichers), pass
    ``pipeline_profile=PipelineConfig.default()`` explicitly.

    Args:
        data_dir: Directory for SQLite and usearch persistence. Defaults to ~/.smartmemory.
        entity_ruler_patterns: Optional pattern manager injected into EntityRulerStage.
        pipeline_profile: PipelineConfig to use. Defaults to ``PipelineConfig.lite()``
            (EntityRuler + local enrichers, no LLM calls). Pass ``PipelineConfig.default()``
            to enable LLM extraction and network enrichers.
        event_sink: Optional in-process event sink (DIST-LITE-3). When provided, pipeline
            events are dispatched to this sink instead of Redis. Defaults to None.
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
        event_sink=event_sink,  # DIST-LITE-3
    )


@contextmanager
def lite_context(data_dir: Optional[str] = None, pipeline_profile=None, event_sink=None):
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
        event_sink: Optional in-process event sink (DIST-LITE-3). Passed to
            ``create_lite_memory()``. Defaults to None.
    """
    memory = None
    try:
        memory = create_lite_memory(data_dir, pipeline_profile=pipeline_profile, event_sink=event_sink)
        yield memory
    finally:
        # Close the SQLite backend explicitly — don't rely on GC for WAL flush.
        if memory is not None:
            try:
                memory._graph.backend.close()
            except Exception:
                pass
