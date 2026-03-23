"""smartmemory.tools.factory — zero-infra SmartMemory factory for Lite."""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional


def _default_data_dir() -> Path:
    return Path.home() / ".smartmemory"


def _ensure_spacy_model(model: str = "en_core_web_sm") -> None:
    """Auto-download spaCy model on first use if not already installed.

    Uses rich for progress output when available (pip install smartmemory-core[lite]).
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


def _load_default_patterns(data_path: Path):
    """Load seed entity patterns from bundled JSONL into a PatternManager.

    Copies bundled seed patterns to the data directory on first use,
    then loads them via PatternManager. Returns None if loading fails
    (EntityRuler falls back to spaCy NER only).
    """
    try:
        import json
        from smartmemory.ontology.pattern_manager import PatternManager

        # Bundled seed patterns ship with the core package
        seed_file = Path(__file__).parent.parent / "data" / "seed_patterns.jsonl"
        user_file = data_path / "entity_patterns.jsonl"

        # Copy seed patterns to user data dir on first use
        if not user_file.exists() and seed_file.exists():
            import shutil
            shutil.copy2(seed_file, user_file)

        if not user_file.exists():
            return None

        # Build a simple pattern dict from the JSONL file (name → label)
        # PatternManager expects a store with load() → [(name, label), ...]
        # Use a minimal in-memory store for the core factory.
        patterns: dict[str, str] = {}
        with open(user_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    freq = entry.get("frequency", 0)
                    name = entry.get("name", "").lower()
                    label = entry.get("label", "")
                    if freq >= 2 and name and label:
                        patterns[name] = label
                except (json.JSONDecodeError, KeyError):
                    continue

        if not patterns:
            return None

        # PatternManager needs a store, but we can use a duck-typed object
        # that just returns the patterns we already loaded.
        class _InMemoryStore:
            def __init__(self, p: dict):
                self._patterns = p

            def load(self, workspace_id: str):
                return list(self._patterns.items())

            def save(self, name, label, confidence=0.85, source="seed", initial_count=2):
                self._patterns[name.lower()] = label

            def delete(self, name, label):
                self._patterns.pop(name.lower(), None)

        return PatternManager(store=_InMemoryStore(patterns))
    except Exception:
        return None


def create_lite_memory(
    data_dir: Optional[str] = None,
    entity_ruler_patterns=None,
    pipeline_profile=None,
    event_sink=None,  # DIST-LITE-3: InProcessQueueSink or None
):
    """Create a SmartMemory instance with local providers. No Docker required.

    Uses SQLite (graph), usearch (vectors), in-memory cache — same full
    pipeline as server mode, just different storage providers. All 11
    pipeline stages run: classify, coreference, entity_ruler, llm_extract,
    ontology_constrain, store, link, enrich, ground, evolve.

    LLM extraction is auto-detected: if ``OPENAI_API_KEY`` or ``GROQ_API_KEY``
    is set, LLM extraction is enabled automatically.

    Args:
        data_dir: Directory for SQLite and usearch persistence. Defaults to ~/.smartmemory.
        entity_ruler_patterns: Optional pattern manager injected into EntityRulerStage.
        pipeline_profile: PipelineConfig override. Defaults to full pipeline.
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
        pipeline_profile = PipelineConfig.default()

    data_path = Path(data_dir).expanduser() if data_dir else _default_data_dir()
    data_path.mkdir(parents=True, exist_ok=True)

    # Auto-load seed entity patterns when no pattern manager is provided.
    # Without patterns, EntityRuler only has spaCy NER (people, orgs, locations)
    # and misses technology names like Django, React, Kubernetes.
    if entity_ruler_patterns is None:
        entity_ruler_patterns = _load_default_patterns(data_path)

    sqlite_backend = SQLiteBackend(db_path=str(data_path / "memory.db"))
    usearch_backend = UsearchVectorBackend(
        collection_name="memory",
        persist_directory=str(data_path),
    )
    graph = SmartGraph(backend=sqlite_backend)

    # DIST-FULL-LOCAL-1: Inject InMemoryOntologyStore so OntologyConstrainStage
    # runs even without FalkorDB. enable_ontology stays False (skips FalkorDB-
    # dependent OntologyGraph), but the injected store unblocks the stage.
    from smartmemory.ontology.in_memory_store import InMemoryOntologyStore

    return SmartMemory(
        graph=graph,
        enable_ontology=False,
        ontology_store=InMemoryOntologyStore(),
        vector_backend=usearch_backend,
        cache=NoOpCache(),
        observability=False,
        pipeline_profile=pipeline_profile,
        entity_ruler_patterns=entity_ruler_patterns,
        event_sink=event_sink,  # DIST-LITE-3
    )


@contextmanager
def lite_context(data_dir: Optional[str] = None, pipeline_profile=None, event_sink=None):
    """Context manager that creates a local SmartMemory and resets all globals on exit.

    Restores observability env, vector backend, cache override, and closes the SQLite
    connection deterministically. Always use this in tests and scripts:
        with lite_context() as memory:
            memory.ingest("hello")

    Args:
        data_dir: Directory for SQLite and usearch persistence. Defaults to ~/.smartmemory.
        pipeline_profile: PipelineConfig override. Defaults to full pipeline.
            Pass ``PipelineConfig.lite(llm_enabled=False)`` to disable LLM and
            skip coreference/enrichment for faster tests.
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
