# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install for development
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# Start infrastructure (FalkorDB on 9010, Redis on 9012, MongoDB on 9013)
# Run from SmartMemory root directory:
docker compose up -d

# Run all tests
PYTHONPATH=. pytest -v tests/

# Run specific test categories
PYTHONPATH=. pytest tests/unit/
PYTHONPATH=. pytest tests/integration/
PYTHONPATH=. pytest tests/e2e/

# Run single test file
PYTHONPATH=. pytest tests/integration/test_llm_models.py -v

# Run single test class/method
PYTHONPATH=. pytest tests/integration/test_llm_models.py::TestDSPyLMCreation::test_gpt4o_mini_any_params -v

# Run examples
PYTHONPATH=. python examples/memory_system_usage_example.py

# Linting and formatting
ruff check --fix .
ruff format .
mypy smartmemory/
```

## Architecture Overview

SmartMemory is a multi-layered AI memory system using FalkorDB (graph + vectors) and Redis (caching).

### Core Components

- **SmartMemory** (`smartmemory/smart_memory.py`): Unified API entry point
- **SmartGraph** (`smartmemory/graph/smartgraph.py`): FalkorDB graph interface. All write methods (`add_node`, `add_edge`, `add_nodes_bulk`, `add_edges_bulk`) support `is_global=True` to skip workspace scoping for shared reference data; for edges this also removes workspace filtering from MATCH clauses.
- **MemoryItem** (`smartmemory/models/memory_item.py`): Core data structure
- **ScopeProvider** (`smartmemory/scope_provider.py`): Multi-tenancy & security scoping

### Memory Types

Core types (MemoryType enum): `working`, `semantic`, `episodic`, `procedural`, `zettel`

Extended types (string values): `reasoning`, `opinion`, `observation`, `decision`

### Processing Pipeline

`ingest()` runs the 11-stage pipeline:

```
classify → coreference → simplify → entity_ruler → llm_extract → ontology_constrain → store → link → enrich → ground → evolve
```

Each stage implements the `StageCommand` protocol (`execute(state, config) → state`, `undo(state) → state`). Pipeline supports breakpoint execution (`run_to()`, `run_from()`, `undo_to()`).

`add()` is simple storage: normalize → store → embed (use for internal/derived items)

### Plugin System

Plugins in `smartmemory/plugins/`:
- **Extractors** (4): LLMExtractor (+ LLMSingleExtractor/GroqExtractor variants), ReasoningExtractor, ConversationAwareLLMExtractor, SpacyExtractor (fallback)
- **Enrichers** (7): BasicEnricher, LinkExpansionEnricher, SentimentEnricher, TemporalEnricher, TopicEnricher, ExtractSkillsToolsEnricher, WikipediaEnricher
- **Grounders** (1): WikipediaGrounder
- **Evolvers** (9): WorkingToEpisodicEvolver, EpisodicToSemanticEvolver, WorkingToProceduralEvolver, EpisodicToZettelEvolver, EpisodicDecayEvolver, OpinionSynthesisEvolver, ObservationSynthesisEvolver, OpinionReinforcementEvolver, DecisionConfidenceEvolver

Custom plugins extend base classes in `smartmemory/plugins/base.py`.

### Observability

All event emission uses `trace_span()` from `smartmemory.observability.tracing`:

```python
from smartmemory.observability.tracing import trace_span, current_trace_id

with trace_span("pipeline.classify", {"memory_type": "semantic"}):
    result = classifier.run(text)
```

- Spans nest automatically via Python contextvars
- OTel-compatible fields: `trace_id`, `span_id`, `parent_span_id`
- Events emitted to Redis Stream `smartmemory:events` on span close
- Enabled by default; disable via `SMARTMEMORY_OBSERVABILITY=false`
- **Deprecated:** `emit_ctx()`, `make_emitter()`, `emit_after()` — use `trace_span()` instead

### Key Directories

- `smartmemory/memory/pipeline/`: Processing stages (classification, extraction, enrichment, linking, grounding)
- `smartmemory/observability/`: Tracing (`trace_span`), events (Redis Stream), logging filter, instrumentation (deprecated)
- `smartmemory/stores/`: Storage backends (vector, persistence)
- `smartmemory/models/`: Data models (MemoryItem, Entity, Opinion, Reasoning)
- `examples/`: Working demonstrations

## API Design

```python
from smartmemory import SmartMemory, MemoryItem

memory = SmartMemory()

# Full pipeline (user-facing ingestion)
item_id = memory.ingest("content here")

# Simple storage (internal operations)
memory.add(MemoryItem(content="...", memory_type="semantic"))

# Search and retrieval
results = memory.search("query", top_k=5)
item = memory.get(item_id)
```

### Constructor parameters (DIST-LITE-2 + DIST-PLUGIN-1)

Five optional params enable zero-infra / lite mode without monkey-patching:

```python
from smartmemory.pipeline.config import PipelineConfig
from smartmemory.utils.cache import NoOpCache

memory = SmartMemory(
    vector_backend=usearch_backend,      # Injects via VectorStore.set_default_backend()
    cache=NoOpCache(),                   # Injects via set_cache_override()
    observability=False,                 # Sets SMARTMEMORY_OBSERVABILITY=false; skips PipelineMetricsEmitter
    pipeline_profile=PipelineConfig.lite(),  # Applied in _build_pipeline_config()
    entity_ruler_patterns=pattern_manager,   # Overrides PatternManager for EntityRulerStage (DIST-PLUGIN-1)
)
```

- `vector_backend`: Any `VectorStoreBackend` instance; `None` leaves default behaviour.
- `cache`: Any cache-compatible object (e.g. `NoOpCache()`); `None` leaves default behaviour.
- `observability`: If `False`, disables all Redis Streams emission and metrics. Defaults to `True`.
- `pipeline_profile`: A `PipelineConfig` instance; its 4 lite flags override the built config. `None` leaves default behaviour.
- `entity_ruler_patterns`: Any object with `get_patterns() → dict[str, str]` interface. Duck-types `PatternManager`. When non-None, overrides the post-ontology-block `pattern_manager` variable passed to `EntityRulerStage`. Used by `smartmemory-cc` to inject `LitePatternManager` (JSONL-backed, zero Docker). `None` leaves default behaviour.
- `PipelineConfig.lite()`: Pre-built profile disabling coreference, LLM extraction, non-basic enrichers, and Wikidata grounding.

## Multi-Tenancy

Security handled by `ScopeProvider`/`MemoryScopeProvider`:
- Methods never take `user_id`, `tenant_id`, `workspace_id` parameters
- Metadata auto-injected on writes, auto-filtered on reads
- Three isolation levels: TENANT, WORKSPACE (default), USER
- OSS mode works without any scoping configuration

## Environment Variables

```bash
FALKORDB_HOST=localhost   # Default: localhost
FALKORDB_PORT=9010        # Default: 9010
REDIS_HOST=localhost      # Default: localhost
REDIS_PORT=9012           # Default: 9012
OPENAI_API_KEY=sk-...     # For embeddings
```

## Code Style

- Line length: 120 (ruff)
- Python: 3.10+
- Type hints required for public APIs
- Google-style docstrings
