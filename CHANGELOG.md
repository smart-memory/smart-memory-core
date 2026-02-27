# Changelog

All notable changes to SmartMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### CORE-DI-1 — Per-Instance Dependency Injection

- `_cache_ctx: ContextVar` in `utils/cache.py` — `get_cache()` checks this before the process-global `_CACHE_OVERRIDE`.
- `_vector_backend_ctx: ContextVar` in `stores/vector/vector_store.py` — `VectorStore.__init__` checks this before module-level `_DEFAULT_BACKEND`.
- `_observability_ctx: ContextVar` in `observability/tracing.py` — `_is_enabled()` checks this before `os.environ`. Wired into `events.py` and `instrumentation.py` via top-level import.
- `SmartMemory._di_context()` — `@contextmanager` that sets all four ContextVars unconditionally at operation entry and resets them LIFO on exit. Always outer wrapper over `trace_span` in `add()`, `delete()`, and `search()` so span enablement and sink routing see the per-instance config.
- 6 new unit tests in `tests/unit/test_di_context.py`: all-four-vars contract, cache/vector/observability thread isolation, exception reset, and no-global-mutation on construct.

### Changed

#### CORE-DI-1 — Per-Instance Dependency Injection

- `SmartMemory.__init__` — removed three global-mutation side effects: `os.environ["SMARTMEMORY_OBSERVABILITY"]`, `VectorStore.set_default_backend()`, `set_cache_override()`. Constructor stores instance attrs only; `_di_context()` activates them per operation.
- `SmartMemory.ingest()` both branches, `add()`, `delete()`, `search()`, `embeddings_search()`, `clear()` — wrapped with `_di_context()`.
- `lite_context()` — removed global-reset teardown block; only the SQLite `backend.close()` remains.
- 8 broken unit tests rewritten in `test_smartmemory_lite_params.py` and `test_lite_factory_coverage.py` to verify ContextVar contract instead of global-mutation contract.

#### DIST-LITE-4 — Loginless Pip-Bundled Graph Viewer (SQLiteBackend read methods)

- `SQLiteBackend.get_all_edges()` — returns all edges as list of dicts with 8 keys: `source_id`, `target_id`, `edge_type`, `memory_type`, `valid_from`, `valid_to`, `created_at`, `properties` (deserialized). Uses `with self._lock:`.
- `SQLiteBackend.get_edges_for_node(node_id)` — filters by `source_id=? OR target_id=?`; same return shape as `get_all_edges()`.
- `SQLiteBackend.get_counts()` — returns `{node_count, edge_count}` from two `COUNT(*)` queries inside a single lock block.

#### DIST-LITE-3 — Lite Mode Graph Viewer Animations (Phase 1 — core)

- `EventSink` — `runtime_checkable` Protocol defining `emit(event_type, payload) -> None`.
- `InProcessQueueSink` — `asyncio.Queue(maxsize=1000)`-backed sink. `emit()` bridges any thread to the asyncio loop via `call_soon_threadsafe(_put)`. The `_put` closure owns the `QueueFull` catch on the loop thread so the caller thread never raises. `attach_loop(loop | None)` wires or detaches the running loop. `_dropped` counter logs at every 100th dropped event.
- `NoOpSink` — zero-overhead null object satisfying `EventSink`.
- `_current_sink: ContextVar[EventSink | None]` — set/reset by `SmartMemory.ingest()` per call so span events from any pipeline stage reach the sink without touching stage interfaces.
- `emit_event()` — dispatches to `_current_sink` before the Redis guard, independent of `SMARTMEMORY_OBSERVABILITY`.
- `SmartMemory.__init__` — new `event_sink=None` param.
- `SmartMemory.ingest()` — both Tier-1 async and sync full-pipeline branches wrap `_pipeline_runner.run()` with ContextVar set/reset in `finally`.
- `create_lite_memory()` and `lite_context()` — `event_sink=None` threaded through.

### Fixed

#### DIST-LITE-3 — Tracing span events blocked from in-process sink

- `SpanContext.emit_event()` guard changed from `if not self.trace_id` to `if not self.trace_id and _current_sink.get() is None`. Lite mode's `observability=False` yields `SpanContext(trace_id="")`, causing the old guard to silently drop all `graph.add_node` / `graph.add_edge` events before reaching the sink.
- `_emit_span()` — checks `_current_sink` at entry: when a sink is active, emits to it and returns early, bypassing the Redis `EventSpooler`.

#### DIST-LITE-6 — Clean Lite Install Path (Zero-Infra Facade)

- Moved `falkordb` and `redis` from base dependencies to the `[server]` optional extra in `pyproject.toml`. `pip install smartmemory` now installs zero infra packages.
- Lazified 8 files that previously had unconditional top-level infra imports: `falkordb.py`, `async_falkordb.py`, `falkor_vector_backend.py`, `ontology/registry.py`, `falkordb_graph_service.py`, `cache.py`, `events.py`, `metrics_consumer.py`.
- Hard-fail modules (FalkorDB backends, `RedisCache`) raise `ImportError` with an actionable `pip install smartmemory[server]` hint on instantiation.
- Soft-fail modules (`EventSpooler`, `MetricsConsumer`) degrade gracefully when Redis is absent — `_connected`/`_redis_available` flag prevents any further Redis calls.
- `create_lite_memory()` now defaults to `PipelineConfig.lite()` when `pipeline_profile` is not supplied, correcting a regression where it ran the full LLM pipeline.
- 22 new unit tests in `tests/unit/test_lite_install_path.py` covering import-time safety, hard-fail with install hints, soft-fail degradation, and spaCy auto-download.

#### CORE-SYS2-1c — Opt-In Reasoning Detection Stage

- `ReasoningDetectConfig` dataclass on `ExtractionConfig` — `enabled: bool = False`, `min_quality_score: float = 0.4`, `use_llm_detection: bool = True`.
- `ReasoningDetectStage` — new pipeline stage after `LLMExtractStage`. Detects reasoning traces (explicit Thought/Action markers or LLM-based implicit detection) via `ReasoningExtractor`. Text priority: `simplified_sentences` > `resolved_text` > `text`. Non-fatal — exceptions logged, never propagated.
- `PipelineState.reasoning_trace` — carries detected `ReasoningTrace` from stage to post-pipeline dispatch. Serialized via `to_dict()`, dropped on `from_dict()` (not reconstructable from dict).
- `PipelineConfig.with_reasoning()` — factory method for standalone use/testing (not usable as `pipeline_profile=` constructor argument).
- `SmartMemory.ingest(extract_reasoning=False)` — opt-in flag. When enabled, post-pipeline dispatch stores reasoning trace as `MemoryItem(memory_type="reasoning", metadata={"auto_extracted": True})` via `self.add()`, linked to source item via `PRODUCED` edge.
- Flag stamps happen AFTER `_apply_pipeline_profile()` to survive lite profile override. Pre-existing 1b gap fixed: `extract_decisions` stamp also moved after profile application.
- Sync-only extraction flags contract: async paths (core `sync=False`, service `mode=async`, EventBus) are never stamped. Contract encoded as automated tests.
- `IngestRequest.extract_reasoning: bool = False` (service layer) — forwarded in both `/memory/ingest` and `/memory/ingest/full` sync routes.
- 36 unit tests + 4 integration tests.

#### CORE-SYS2-1b — Auto-Extract Decisions via LLM Schema Extension

- `_build_extraction_schema(extract_decisions: bool)` — builds the LLM response schema by deep-copying `EXTRACTION_JSON_SCHEMA` and optionally appending a `decisions` array property. Never mutates the module-level constant.
- `EXTRACTION_SCHEMA_VERSION` — integer constant for cache-level schema invalidation. Increment when extraction schema changes.
- `LLMSingleExtractorConfig.extract_decisions: bool = False` — flag controlling whether the extractor requests decisions from the LLM.
- Updated cache key in `LLMSingleExtractor._extract_impl()` to include `extract_decisions`, `EXTRACTION_SCHEMA_VERSION`, and a `sha256` text digest (replaces unstable `hash()`).
- `LLMExtractConfig.extract_decisions: bool = False` and `decision_confidence_threshold: float = 0.75` — pipeline-level controls on `LLMExtractConfig`.
- `PipelineConfig.with_decisions()` — factory method that builds a default config with LLM decision extraction enabled.
- `PipelineState.llm_decisions: List[Dict[str, Any]]` — carries raw decision dicts from `LLMExtractStage` to `SmartMemory.ingest()`.
- `LLMExtractStage` Groq and non-Groq paths both forward `extract_decisions` to the extractor config. `execute()` writes `llm_decisions` to state; `undo()` clears it.
- `SmartMemory.ingest(extract_decisions=False)` — post-pipeline dispatch block stores qualifying decisions via `DecisionManager.create()`. Tags each decision with `"auto_extracted"`. Non-fatal at every level: per-item `(TypeError, ValueError)` guard + outer `Exception` guard around `create()`.
- `IngestRequest.extract_decisions: bool = False` (service layer) — exposed on both `/memory/ingest` and `/memory/ingest/full`.
- 26 unit tests + 4 integration tests.

#### Two-Tier Async Extraction + EntityRuler Self-Learning Loop (CORE-EXT-1)
- `PipelineConfig.tier1()` — sync-only profile (EntityRuler + store + link + enrich + evolve, no LLM extraction).
- `RedisStreamQueue.for_extract()` / `.for_ground()` — factory methods for extract and ground worker queues.
- `smartmemory/background/extraction_worker.py` — `process_extract_job()`: LLM diff → net-new entities/relations → EntityPattern upsert (count++) → ruler reload signal. `enable_ontology` flag from payload gates all ontology writes.
- `smartmemory/background/id_resolver.py` — pure functions bridging SHA-256 LLM ID space to FalkorDB graph node IDs.
- `OntologyGraph.get_entity_patterns()` frequency gate: `count >= 2` required; single-occurrence patterns excluded from ruler.
- `ingest(sync=False)` now runs Tier 1 inline and enqueues to extract stream. Previously bypassed the pipeline entirely.
- Memory type preserved in async path via `meta.setdefault("memory_type", ...)`.

#### CORE-EVO-ENH-3 — Hebbian Co-Retrieval Edge Reinforcement

- **`smartmemory/graph/models/schema_validator.py`** (modified): Registered `CO_RETRIEVED` edge schema in `_register_default_schemas()` after the `RELATED` edge. Source and target node types: `{"semantic", "episodic", "procedural", "zettel"}`. Optional properties: `co_retrieval_count` (int), `weight` (float), `first_co_retrieved_at` (str), `last_co_retrieved_at` (str). No `workspace_id` in schema — auto-stamped by scope provider on every write.
- **`smartmemory/plugins/evolvers/enhanced/hebbian_co_retrieval.py`** (new): `HebbianCoRetrievalEvolver` queries `CO_RETRIEVED` edges with `co_retrieval_count >= weight_threshold` via `execute_cypher()`, deduplicates hits by `item_id` (a node in multiple qualifying edges receives exactly one boost), and applies `min(max_boost, (count − threshold) × retention_boost_per_unit)` to `retention_score` via `update_properties()`. Wrapped in `trace_span("pipeline.evolve.hebbian_co_retrieval", {})`.
- **`HebbianCoRetrievalConfig`**: `weight_threshold=3.0`, `max_edges_per_cycle=500`, `retention_boost_per_unit=0.02`, `max_boost=0.3`, `memory_types=["episodic", "semantic", "procedural"]`.
- **`smartmemory/plugins/evolvers/enhanced/__init__.py`** (modified): `HebbianCoRetrievalEvolver` imported and appended to `ENHANCED_EVOLVERS` list after `RetrievalBasedStrengtheningEvolver`.
- **`smart-memory-worker/worker/consumers/retrievals.py`** (modified): Added `_extract_co_retrieval_pairs(events_raw) → Counter[frozenset]` — iterates all `type == "search"` events in a flush batch, extracts 2-combinations of `item_id`s from each result set, returns a `Counter` keyed by `frozenset`. Added `_flush_co_retrieval_edges(co_pairs, ws_id) → int` — uses `MERGE … ON CREATE / ON MATCH SET r.co_retrieval_count = r.co_retrieval_count + $count` for true increment semantics; canonical pair ordering (`min→max` lexicographic) ensures the same pair always maps to the same directed edge. Injected as a guarded step in `_flush_workspace()` after the profile loop and before `XDEL`. Hebbian failure cannot prevent event deletion.
- **42 new tests**: 11 evolver unit tests (`tests/unit/plugins/evolvers/test_hebbian_co_retrieval.py`) covering boost formula, threshold gating, `max_boost` cap, retention ≤ 1.0, `item_id` dedup, missing-item handling, query failure recovery, and plugin metadata. 10 new consumer tests added to `tests/unit/test_retrieval_flush_consumer.py` covering pair extraction (3-results, repeated pairs, single result, get-event ignored, malformed JSON, canonical ordering) and edge flush (execute_cypher call count, canonical param ordering, exception handling). Full unit suite: 2087 passed.

#### CORE-EVO-ENH-2 — Retrieval-Based Memory Strengthening

- **`smartmemory/observability/retrieval_tracking.py`** (new): Retrieval event accumulation layer. Exports a `retrieval_source: ContextVar[str]` (default `"internal"`) for service-layer source annotation. `emit_retrieval_get(item, scope_provider)` emits one `{type: "get", item_id, source, ts}` event to `smartmemory:retrievals:{ws_id}` (Redis DB 2, MAXLEN `~ 50,000`). `emit_retrieval_search(results, query, scope_provider)` emits one batched event per call with `results_json` containing per-result `{item_id, rank, score}`. Both functions call `SADD smartmemory:active_workspaces {ws_id}` on every write. All exceptions are swallowed — observability failure never affects `get()` or `search()` return values. Uses the same lazy singleton `redis.Redis` pattern as `pipeline_producer.py`.
- **`smartmemory/smart_memory.py`** (modified): `get()` calls `emit_retrieval_get()` after a successful fetch (result is not None). `search()` calls `emit_retrieval_search()` after the canonical search path collects results. Both hooks wrapped in `try/except Exception: pass`.
- **`smartmemory/plugins/evolvers/enhanced/retrieval_based_strengthening.py`** (new): `RetrievalBasedStrengtheningEvolver` reads `metadata["retrieval__profile"]` (written by `RetrievalFlushConsumer`) and adjusts `metadata["retention_score"]` using the formula: `boost = retrieval_weight × log(1+total_count) × (1/(1+avg_rank)) × (v7d/(v30d+ε))`, `retention = min(1.0, base + boost)`. For memories with zero retrievals, applies gentle unretrieved decay: `retention *= (1 - rate × elapsed_days)`. Handles JSON string vs dict profile shapes with explicit `json.loads()` guard (FalkorDB serialises dicts to JSON strings). Registered in `ENHANCED_EVOLVERS`, `_load_builtin_plugins()`, and `evolvers/__init__.py`.
- **`RetrievalBasedStrengtheningConfig`**: `retrieval_weight=0.1`, `unretrieved_decay_rate=0.005`, `lookback_days=30`, `max_memories=500`, `memory_types=["episodic", "semantic", "procedural"]`.
- **`smart-memory-service/memory_service/api/routes/crud.py`** (modified): Imports `retrieval_source` and sets it at the top of `get_memory()`, `search_memory()`, and `search_memory_advanced()` handlers so events are attributed to their API origin.
- **78 unit tests** across `tests/unit/plugins/evolvers/test_retrieval_based_strengthening_evolver.py` (34 tests) and `tests/unit/observability/test_retrieval_tracking.py` (22 tests), all passing.

#### CORE-BG-1 — Pipeline Stream Producer

- **`smartmemory/streams/pipeline_producer.py`** (new): Emits `{item_id, workspace_id, memory_type, ts}` to `smartmemory:pipeline` (Redis DB 2) after `ingest()` completes. Uses a lazy singleton sync `redis.Redis` client. MAXLEN `~ 10,000` (approximate, O(1)). All exceptions are swallowed — pipeline event emission never breaks `ingest()`.
- **`smartmemory/streams/__init__.py`** (new): Streams subpackage.
- **`smartmemory/smart_memory.py`** (modified): `ingest()` calls `emit_pipeline_event()` after `pipeline_runner.run()` returns. Wrapped in `try/except Exception: pass` — observability failures are non-fatal.

#### CORE-EVO-ENH-1 — ExponentialDecay and InterferenceBasedConsolidation Enhanced Evolvers

- **`ExponentialDecayEvolver`** (`smartmemory/plugins/evolvers/enhanced/exponential_decay.py`): Applies the Ebbinghaus forgetting curve to episodic memories. Computes `retention = exp(-elapsed_days / stability)` where `stability` is a per-memory value read from `metadata["stability"]` (defaults to 30 days). Writes `retention_score` to each item's metadata; archives items below the configurable `archive_threshold` (default 0.1) by setting `metadata["archived"] = True` and `metadata["archive_reason"] = "exponential_decay"`. Registered in `ENHANCED_EVOLVERS` and `_load_builtin_plugins()`.
- **`ExponentialDecayConfig`** (`smartmemory/plugins/evolvers/enhanced/exponential_decay.py`): `default_stability=30.0`, `archive_threshold=0.1`, `max_memories=500`.
- **`InterferenceBasedConsolidationEvolver`** (`smartmemory/plugins/evolvers/enhanced/interference_based_consolidation.py`): Models retroactive and proactive interference between similar episodic and semantic memories. For each embedded memory, searches up to `top_k_neighbors` similar memories and applies a cosine-similarity-based retention penalty to pairs that exceed `similarity_threshold` (default 0.85): `new_score = current_score * (1 - similarity * interference_weight)`. Pair deduplication ensures each `(A, B)` pair is penalised exactly once. Items without embeddings are skipped. Writes `retention_score` and `interference_count` to metadata.
- **`InterferenceBasedConsolidationConfig`** (`smartmemory/plugins/evolvers/enhanced/interference_based_consolidation.py`): `similarity_threshold=0.85`, `interference_weight=0.05`, `top_k_neighbors=5`, `max_memories=200`.
- Both evolvers added to `ENHANCED_EVOLVERS` list (`enhanced/__init__.py`) and to `_load_builtin_plugins()` in `manager.py`.
- Both classes exported from `smartmemory.plugins.evolvers` (`__all__` + import).
- 48 unit tests across `tests/unit/plugins/evolvers/test_exponential_decay_evolver.py` and `tests/unit/plugins/evolvers/test_interference_based_consolidation_evolver.py`.

#### DIST-PLUGIN-1 Prerequisite — `entity_ruler_patterns` Constructor Param

- **`SmartMemory(entity_ruler_patterns=...)` parameter** (`smartmemory/smart_memory.py`): Accepts any object with a `get_patterns() → dict[str, str]` interface (duck-types `PatternManager`). When non-None, overrides the post-ontology-block `pattern_manager` variable so `EntityRulerStage` receives the injected manager. Placement is after the `if self._enable_ontology:` block, so it works in Lite mode where `enable_ontology=False` leaves `pattern_manager=None`. In full mode it overrides the FalkorDB-backed `PatternManager` if explicitly injected. Defaults to `None` (no behaviour change for existing callers).
- **`create_lite_memory(entity_ruler_patterns=None)` param** (`smartmemory/tools/factory.py`): Threads the new constructor param through, enabling the `smartmemory-cc` plugin to inject `LitePatternManager` for zero-infra entity pattern storage.

#### Delete `smartmemory-lite` Package (DIST-LITE-2)

- **`smartmemory-lite` package deleted**: The separate `smartmemory-lite` PyPI package is retired. All functionality now lives in the core `smartmemory` package under `smartmemory.tools.*` and `smartmemory.cli`.
- **`smartmemory.tools.factory`** (`smartmemory/tools/factory.py`): `create_lite_memory()` and `lite_context()` migrated from `smartmemory_lite.factory`. Thin wiring with zero monkey-patching via the 4 new `SmartMemory` constructor params.
- **`smartmemory.tools.markdown_writer`** (`smartmemory/tools/markdown_writer.py`): `_render()` and `write_markdown()` migrated from `smartmemory_lite.markdown_writer`.
- **`smartmemory.cli`** (`smartmemory/cli.py`): `add`, `search`, `rebuild`, `watch` CLI commands migrated from `smartmemory_lite.cli`. The `mcp` command is dropped — use the MCP server in `smart-memory-service` instead. Entry point: `smartmemory = "smartmemory.cli:main"`.
- **`lite` optional dep group** (`pyproject.toml`): `pip install smartmemory[lite]` pulls in `usearch` and `click/rich`. `watch` group adds `watchdog`.

#### Absorb Lite Mode into Core SmartMemory Constructor (DIST-LITE-2)

- **`SmartMemory(vector_backend=...)` parameter** (`smartmemory/smart_memory.py`): Injects a custom vector backend by calling `VectorStore.set_default_backend()` at construction time. Eliminates the corresponding monkey-patch in `smartmemory-lite`.
- **`SmartMemory(cache=...)` parameter** (`smartmemory/smart_memory.py`): Overrides the global cache by calling `set_cache_override()` at construction time. Eliminates the `SMARTMEMORY_CACHE_DISABLED` env-var workaround.
- **`SmartMemory(observability=...)` parameter** (`smartmemory/smart_memory.py`): Always writes `SMARTMEMORY_OBSERVABILITY` to env (`"true"` or `"false"`) — both branches — so a subsequent `SmartMemory(observability=True)` correctly re-enables tracing after a prior lite instance disabled it. Skips `PipelineMetricsEmitter` when `False`.
- **`SmartMemory(pipeline_profile=...)` parameter** (`smartmemory/smart_memory.py`): Accepts a `PipelineConfig` instance whose 4 lite-mode flags are applied in `_build_pipeline_config()`. Eliminates the closure-patching approach in `_apply_lite_pipeline_profile`.
- **`PipelineConfig.lite()` classmethod** (`smartmemory/pipeline/config.py`): Returns a config with coreference, LLM extraction, enrichers (basic only), and Wikidata grounding all disabled. Canonical single source of truth for the lite pipeline profile.
- **`set_cache_override()` / `_CACHE_OVERRIDE`** (`smartmemory/utils/cache.py`): New mechanism to inject a cache backend (e.g. `NoOpCache()`) that `get_cache()` returns regardless of env vars. Clears `_global_cache` on set to prevent stale instances.
- **Call-time env reads in observability modules**: Replaced module-load-time `_ENABLED` / `_OBSERVABILITY_ENABLED` constants with `_is_enabled()` / `_is_observability_enabled()` functions that read `SMARTMEMORY_OBSERVABILITY` from `os.environ` on every call. Enables `SmartMemory(observability=False)` to take effect after module import.
- **`_apply_pipeline_profile()` module helper** (`smartmemory/smart_memory.py`): Pure function that copies 4 lite-mode flags from a profile `PipelineConfig` onto a freshly-built config.

#### SmartMemory Lite — Zero-Infrastructure Local Memory (DIST-LITE-1)

- **`smartmemory-lite` package (new):** `pip install smartmemory-lite` provides the full SmartMemory pipeline (classify → extract → store → embed → search) backed by SQLite + usearch. No Docker, FalkorDB, Redis, or MongoDB required.
- **`SQLiteBackend`** (`smartmemory/graph/backends/sqlite.py`): `SmartGraphBackend` implementation using stdlib `sqlite3`. Adjacency-list model, WAL mode, cascading FK deletes, thread-safe via `threading.Lock`. Full serialize/deserialize for backup/restore. Two-phase `deserialize()` prevents FK violations from rolling back node restores.
- **`UsearchVectorBackend`** (`smartmemory/stores/vector/backends/usearch.py`): ANN search via `usearch.Index` (cosine similarity) + FTS5 full-text search via SQLite. Persists integer key ↔ item_id maps and metadata to a companion JSON file. Reconstructs index from disk on process restart.
- **`SmartGraph(backend=x)` injection** (`smartmemory/graph/smartgraph.py`): New `backend` parameter bypasses `_get_backend_class()` entirely, allowing any `SmartGraphBackend` implementation to be injected without config changes.
- **`SmartMemory(enable_ontology=False)`** (`smartmemory/smart_memory.py`): New flag that skips `OntologyGraph` initialization and replaces `OntologyConstrainStage` with a no-op, preventing FalkorDB connections in Lite mode.
- **`VectorStore.set_default_backend()`** (`smartmemory/stores/vector/vector_store.py`): Class-level backend override. All `VectorStore()` constructions in the process use the injected backend until reset.
- **`NoOpCache`** (`smartmemory/utils/cache.py`): Cache implementation that silently discards all ops. Activated via `SMARTMEMORY_CACHE_DISABLED=true` env var.
- **Non-fatal vector registry** (`smartmemory/stores/vector/backends/base.py`): `_ensure_registry()` no longer raises on `ImportError` for missing backends (e.g. `falkordb` not installed). Logs warning, returns empty registry.
- **`smartmemory-lite` CLI** (`smartmemory_lite/cli.py`): `smartmemory-lite add`, `search`, `rebuild`, `watch`, `mcp` commands. `watch` ingests markdown files from a vault directory via `watchdog`.
- **`smartmemory-lite` MCP server** (`smartmemory_lite/mcp_server.py`): JSON-RPC stdio handler (`LiteMCPHandler`) with `ingest`, `add`, `search`, `get`, `delete` methods. Compatible with Claude Desktop and Claude Code MCP configuration.
- **`smartmemory-lite` markdown writer** (`smartmemory_lite/markdown_writer.py`): Writes `MemoryItem` objects to markdown files with YAML frontmatter (item_id, memory_type, created_at, entities, relations).
- **166 new tests:** 42 SQLiteBackend + 24 coverage, 6 UsearchVectorBackend + 17 coverage, 3 integration (golden flow, no-network ingest, no-network MCP), 10 factory coverage, 5 prerequisite unit test files (P0-1 through P0-5), 14 markdown writer, 17 MCP server, 28 post-release bug-fix regression tests.

### Fixed

#### DIST-LITE-2 — Observability and Resource Cleanup

- **One-way process-global observability** (`smartmemory/smart_memory.py`): `SmartMemory(observability=False)` previously set `SMARTMEMORY_OBSERVABILITY=false` with no symmetric restore path — a later `SmartMemory(observability=True)` left tracing silenced. Fixed by always writing both branches: `os.environ["SMARTMEMORY_OBSERVABILITY"] = "false" if not observability else "true"`. This makes the constructor idempotent and fully reversible.
- **`lite_context()` observability leak** (`smartmemory/tools/factory.py`): `lite_context()` reset the vector backend and cache override but did not restore `SMARTMEMORY_OBSERVABILITY`. Processes that created and closed a lite context continued to have observability silenced. Fixed by saving the env-var value before `create_lite_memory()` and restoring (or removing) it in the `finally` block.
- **`lite_context()` SQLite connection leak** (`smartmemory/tools/factory.py`): `lite_context()` exited without closing the underlying `SQLiteBackend` connection. Resource release depended on GC, leaving WAL flush timing non-deterministic in long-lived processes. Fixed by calling `memory._graph.backend.close()` in the `finally` block (`SQLiteBackend.close()` was already implemented; only the call site was missing).

#### DIST-LITE-2 — smartmemory-lite Package Retired

- **Deleted `smartmemory-lite`**: Now that DIST-LITE-2 absorbed all five lite behaviors into the `SmartMemory` constructor, `smartmemory-lite` has no unique code to maintain. The package has been deleted from the monorepo.
- **CLI migrated to `smartmemory.cli`** (`smartmemory/cli.py`, new): The `add`, `search`, `rebuild`, and `watch` commands from `smartmemory-lite` are now the `smartmemory` CLI entry point. The `mcp` command was dropped — use the MCP server in `smart-memory-service` instead.
- **`markdown_writer` migrated to `smartmemory.tools`** (`smartmemory/tools/markdown_writer.py`, new): Unchanged logic, updated module path.
- **`factory` migrated to `smartmemory.tools`** (`smartmemory/tools/factory.py`, new): `create_lite_memory()` and `lite_context()` now live here. Now includes full cleanup (observability env restore + SQLite close).
- **`lite` optional dep group added** (`pyproject.toml`): `pip install smartmemory[lite]` installs `usearch` and `click`/`rich` for the CLI. `watch` group adds `watchdog`.

#### SmartMemory Lite — Post-Release Bug Fixes (DIST-LITE-1)

- **FTS cross-collection isolation** (`UsearchVectorBackend`): FTS5 virtual table was shared across all collections sharing the same `fts.db` file. Each collection now owns a dedicated virtual table `fts_{sanitized_collection_name}`. `search_by_text()` and `clear()` are now scoped to their collection; clearing one collection no longer wipes another's text index.
- **`search_nodes()` generic property filter** (`SQLiteBackend`): Query keys not in the explicit `if` block were silently ignored, returning all nodes regardless of filter. All keys now route to an appropriate SQL clause — top-level columns (`memory_type`, `valid_from`, etc.) via direct equality, `content` via `LIKE`, and all other keys via `json_extract(properties, ?) = ?` (SQL-injection safe parameterized path). Matches FalkorDB equality-filter semantics.
- **Edge temporal fields data loss** (`SQLiteBackend`): `edges` table had no `valid_from`, `valid_to`, `created_at` columns. `add_edge()` silently discarded those arguments. Schema updated; `_create_tables()` now runs `ALTER TABLE ... ADD COLUMN` migrations for existing databases. `add_edge()`, `serialize()`, and `deserialize()` all updated to persist and restore temporal fields.
- **`get_neighbors()` outgoing-only divergence** (`SQLiteBackend`): Returned only outgoing neighbors (`WHERE source_id=?`), while FalkorDB uses undirected `MATCH (n)-[r]-(m)`. Fixed via `UNION` of both edge legs: outgoing (`source_id=?`) and incoming (`target_id=?`). Now matches FalkorDB bidirectional semantics.
- **`scope_provider` silent isolation gap** (`SQLiteBackend`): Constructor accepted `scope_provider` but never stored or applied it, silently allowing misuse in multi-tenant contexts. Now raises `ValueError` immediately when non-None — hard refusal so the contract violation is impossible to miss.
- **`rebuild` graph mutation** (`smartmemory-lite` CLI): `rebuild` called `memory.add(MemoryItem(...))` which routes through CRUD graph writes, dropping all metadata not explicitly preserved in the `MemoryItem` constructor (labels, source, custom fields). Rewritten to bypass the graph entirely: `create_embeddings(content)` → `VectorStore().upsert(item_id=..., embedding=..., metadata=properties)` where `properties` is the full serialized node properties dict.
- **Backend registry catches too-broad exceptions** (`VectorStore` registry): `except (ImportError, Exception)` narrowed to `except ImportError`. Real backend code bugs (syntax errors, broken imports) now propagate instead of being silently logged as "backend unavailable."
- **`GroundStage` test implementation-detail assertion** (`pipeline_v2/stages/test_ground.py`): `memory._grounding.ground.assert_called_once()` tested mechanism rather than behavior. `GroundStage` uses `WikipediaGrounder` directly and never calls `memory._grounding`. Assertion removed; behavioral assertion (`provenance_candidates` stored in `_context`) is sufficient and passes.

### Changed

#### SSO Migration — Clerk-Based IdP, Legacy Auth Routes Removed (PLAT-SSO-IDP-1)

- **smart-memory-service:** Removed `/auth/login`, `/auth/signup`, `/auth/google/*`, `/auth/password-reset/*` routes. Canonical auth surface is now: `/auth/clerk/session`, `/auth/onboarding/{start,complete,retry}`, `/auth/me`, `/auth/refresh`, `/auth/logout`, `/auth/logout-all`.
- **smart-memory-common:** Added `ClerkSSOService` adapter; OIDC callback wired to `OnboardingService` (Clerk-backed). `sso_subject` field added to `UserModel` for IdP identity linking. JIT provisioning on first Clerk login.
- **All apps (web, studio, insights, maya, admin, viewer):** Switched from email/password login forms to Clerk-hosted IdP. Auth state bootstrapped via `GET /auth/me` (cookie-based, `credentials: 'include'`). Token-URL-param cross-app bridge (`VITE_LEGACY_CROSS_APP_TOKEN_REDIRECT`) deleted.
- **smart-memory-sdk-js:** All frontends switched to `mode: 'sso'`. Removed `endpoints.login`, `AuthCore.login()`, `AuthCore.signup()`. Session bootstrapped via `bootstrapSession()` → `/auth/me` on mount.
- **E2E fixtures:** Replaced localStorage injection with `context.addCookies()` per-app origin. `SM_E2E_TOKEN` + `SM_E2E_TEAM_ID` env vars replace HTTP signup provisioning.
- **Tests:** Rewrote `test_auth.py` (service integration), 8 SDK JS test files, `test_client_full_coverage.py`, and all 6 per-app auth E2E specs for SSO mode.

### Added

#### Enricher & Evolver Event Emission for Live Graph Viewer (CORE-OBS-3)

- **smartgraph.py:** `add_node()`, `add_edge()`, `add_nodes_bulk()`, `add_edges_bulk()`, and `remove_node()` now emit `graph_mutation` events to the Redis Stream. Enrichers and evolvers that create graph mutations are now visible in the Graph Viewer's live operations feed in real-time.
- **evolve.py (service):** Evolution trigger, dream phase, opinion synthesis, observation synthesis, and opinion reinforcement routes now emit `emit_graph_event()` after successful operations.
- **Tests:** 10 unit tests verify event emission, failure resilience, and no double-emission from `add_dual_node`.

### Changed

#### Auth Model Simplification — Teams+Users Only (SVC-6)

- **smart-memory-common:** Removed dual scope resolution path — `scope.py` no longer reads `X-Workspace-Id` or validates against MongoDB workspaces. `core.py` now reads `team_id` from `request.state` (set by scope dependency). Single source of truth for scope resolution.
- **smart-memory-common:** Removed `_can_access_workspace()`, `effective_workspace_id`, `has_workspace_access()`. `ScopePolicy.workspace_required/optional` → `team_required/optional`. Removed `ensure_personal_workspace()` call from middleware.
- **smart-memory-service:** Updated 11 route files from `ScopePolicy.workspace_required` → `team_required`. Removed dead `ensure_workspace_access()` from tenancy.py.
- **Cross-project:** Changed `X-Workspace-Id` → `X-Team-Id` header in JS SDK, E2E fixtures, MCP server, Studio, Hub.
- **Dormant:** Workspace models and repository remain in codebase for future PLAT-WS-1 (Workspaces & Sharing).

### Fixed

#### Semantic Relation Pipeline — 5 Cascading Bug Fix (CORE-REL-1)

- **crud.py:** Always return full dict from `add_dual_node` (entity_node_ids were silently discarded)
- **falkordb.py:** Create semantic edges for ALL entity pairs, not just both-new-entities; CREATE → MERGE prevents duplicates
- **flow.py:** Removed internal relation filter that dropped 100% of semantic relations; added SHA256→graph-ID resolution map
- **store.py (pipeline v2):** Same internal filter bug fix; proper ID resolution via `extraction_id_to_graph_id` map
- **ontology_constrain.py:** Preserve LLM `item_id` during ruler+LLM entity merge so relation endpoint matching works
- **smart_memory.py:** Fixed 3 callers of `crud.add()` that expected string return; removed dead `isinstance` defensive branches
- **Integration tests:** 4 tests covering happy path, all-relations, empty-entities, unresolvable-target edge cases

#### Graph Edge Global Scope Support (CORE-GRAPH-1)

- Added `is_global: bool = False` to `add_edge()` across the full propagation chain (ABC → FalkorDB → SmartGraphEdges → SmartGraph)
- When `is_global=True`, skips `get_write_context()` and uses unscoped MATCH — edges can now reach global nodes (no `workspace_id`)
- Updated `add_edges_bulk()` ABC fallback to forward `is_global` to `add_edge()`
- Updated Wikipedia grounder and grounding pipeline stage to pass `is_global=True` for `GROUNDED_IN` edges
- **Bug fixed:** In multi-tenant mode, `add_edge()` scoped MATCH to `workspace_id`, causing edges to global nodes (e.g. Wikipedia entities) to silently fail (PRE-13)

### Added

#### Admin Console Completion (PLAT-ADMIN-1)

- **smart-memory-admin:** Fixed 3 critical bugs (OAuth token key mismatch, impersonation console.log leak, hardcoded metric fallbacks), setup Vitest + CI pipeline, removed Deployments page, redesigned System page with confirmation dialogs for destructive ops
- **smart-memory-service:** Rewrote `superadmin.py` (735 lines, 17 endpoints with `require_superadmin`): dashboard stats, user/tenant CRUD, feature flags in MongoDB, activity logging with 90-day TTL, system health, billing overview, database maintenance. Added 5 cross-tenant Tier 2 endpoints (orphaned-notes, prune, old-notes, summary, reflect) via `_get_tenant_smart_memory()` with audit trail. Fixed missing `require_admin` on reflect/summarize routes
- **smart-memory-admin:** Wired all 7 pages (Dashboard, Users, Tenants, System, FeatureFlags, Activity, Billing) to real API data. Added Data Management tab in Tenants dialog with scan/preview/execute workflow. Token refresh mutex for concurrent 401s
- **smart-memory-e2e:** Added admin project to Playwright config with superadmin auth fixture, 13 E2E tests (auth, pages, users)
- **Tests:** 60 Vitest tests (api, api-contract, AuthContext, utils, Users, Tenants) + 13 Playwright E2E tests + API contract drift detector (method count assertion)
- **Review fixes:** Removed `updateTenant` method (no backend endpoint), replaced console.log/error/warn with errorTracker in AuthCallback, fixed inconsistent `|| 0` vs `N/A` fallbacks in Billing, added logger.warning to activity log catch block
- **Scope:** `smart-memory-admin`, `smart-memory-service`, `smart-memory-e2e`

#### Flatten Event Emission (CORE-OBS-2)

- `EventSpooler.emit_event()` now accepts trace field kwargs: `name`, `trace_id`, `span_id`, `parent_span_id`, `duration_ms`
- When provided, these are written as top-level Redis Stream fields (not buried in the `data` JSON blob)
- `_emit_span()` updated to forward trace fields as explicit kwargs
- Existing callers unaffected — new kwargs default to `None`, no fields added unless provided
- Insights: removed `normalize_span_event()` shim (INS-SPAN-1); all 4 read paths now extract trace fields directly from top-level Stream fields with backward-compatible fallback to data blob for legacy events
- 9 new core unit tests for trace field emission; 26 new Insights invariant tests for inline extraction (both read paths + backward compat); 15 INS-SPAN-1 shim tests removed
- **Scope:** `smart-memory` (core), `smart-memory-insights` (consumer cleanup)

#### `is_global` Support for Bulk Graph Methods (CORE-BULK-2)

- `add_nodes_bulk()` and `add_edges_bulk()` now accept `is_global: bool = False` parameter
- When `True`, workspace scoping is skipped — nodes/edges are visible across all workspaces
- For edges, `is_global=True` also removes workspace filtering from MATCH clauses so global nodes can be linked
- Propagated through ABC (`SmartGraphBackend`), `FalkorDBBackend`, and `SmartGraph` delegation
- Use for bulk loading shared reference data (ontology types, Wikipedia entities)

#### Insights Span Event Compatibility (INS-SPAN-1)

- `normalize_span_event()` backend normalization shim — promotes trace fields (`name`, `trace_id`, `span_id`, `parent_span_id`, `duration_ms`) from `data` blob to top-level for span events
- Applied to all 4 data paths: `get_recent_events()` (2 paths), WebSocket streaming (2 paths)
- Fixed error detection in `get_recent_errors()` and `get_operations_timeline()` for span events
- `EventData.session_id` now Optional; 6 new trace fields added to model
- Frontend: 21 span name → description mappings, trace context in hover card and expanded view
- 15 invariant tests for the normalization function (100% branch coverage)
- **Scope:** `smart-memory-insights` only (forward-compatible patch; shim removed by CORE-OBS-2)

### Fixed

#### Test Audit Fixes — Batches 1 & 2

- **T1**: Fixed `MEMORY_TYPES` test — added missing "code" type, count 9→10 (`test_memory_item.py`)
- **T2**: Removed 2 stale `@pytest.mark.skip` decorators on decision lifecycle tests that were already implemented (`test_e2e_workflows.py`)
- **T3**: Fixed permanent `skipif(True)` in tenant isolation tests — rewired to use `second_tenant` + `AuthTestClient` pattern (`test_procedures.py`)

### Added

- **T4**: Integration tests for Code Connector API — 18 tests covering POST /code/index, GET /code/search, GET /code/dependencies (`test_code.py`)
- **T5**: Integration tests for Procedure Evolution API (CFS-3b) — 14 tests covering GET /procedures/{id}/evolution, GET /procedures/{id}/evolution/{event_id}, GET /procedures/{id}/confidence-trajectory with contract-validated TrajectoryPoint shape (`test_procedure_evolution.py`)

---

## [0.3.17] - 2026-02-14

### Changed

#### Observability Event Path Unification (CORE-OBS-1)

- **New `trace_span()` API** — Single observability entry point replacing three independent emission paths (`emit_ctx`/`make_emitter`, `IngestionObserver.emit_event`, `PipelineMetricsEmitter`)
- **OTel-compatible span model** — `SpanContext` dataclass with `trace_id`, `span_id`, `parent_span_id`, automatic parent-child nesting via `contextvars.ContextVar`
- **New `smartmemory/observability/tracing.py`** — Core module: `trace_span()` context manager, `current_span()`, `current_trace_id()`, `SpanContext`
- **Migrated primary call sites** — `smart_memory.py` (4 CRUD methods), `graph/core/nodes.py`, `graph/core/edges.py`, `stores/vector/vector_store.py` (5 operations), `memory/ingestion/observer.py`, `pipeline/metrics.py`, `smart-memory-service/service.py`, `smart-memory-service/scripts/run_background_enricher.py` (14 calls)
- **Deprecated all legacy APIs** — `emit_ctx()`, `make_emitter()`, `emit_after()`, `set_obs_context()`, `get_obs_context()`, `update_obs_context()`, `clear_obs_context()`, `with_obs_context()` now emit `DeprecationWarning` with `stacklevel=2`; will be removed in next minor
- **Logging filter bridge** — `LogContextFilter` now injects `trace_id`, `span_id`, `parent_span_id` from active span into log records
- **FastAPI tracing middleware** — Root `trace_span("http.request")` per API request in service.py
- **29 unit tests** for tracing module (nesting, thread safety, disabled mode, error handling, attribute collision protection)

---

## [0.3.16] - 2026-02-14

### Fixed

#### Pre-Existing Issues Resolution (gaps.md PRE-1 through PRE-12)

- `add_edge()` now sanitizes edge types with `re.sub` (PRE-1)
- `SmartGraphBackend` ABC rewritten — correct signatures, `add_nodes_bulk`/`add_edges_bulk` fallback methods (PRE-2, PRE-3)
- `get_scope_filters()` added to `SmartGraph` — eliminates duplication in indexer and MCP tools (PRE-4)
- `extract_node()`/`extract_rel_type()` moved to shared `graph_utils.py` (PRE-5)
- `SmartGraphEdges.clear_cache()` added, `hasattr` guards removed (PRE-6)
- `Triple` forward ref replaced with `Any` in SmartGraph and SmartGraphEdges (PRE-7)
- Exception re-raises now include `from e`/`from exc` (PRE-8)
- Unused variables removed from `falkordb.py` (PRE-9)
- `add_edge()` and `add_edges_bulk()` MATCH clauses now scoped to `workspace_id` in multi-tenant mode (PRE-11)
- `add_node()` now applies `sanitize_label()` (PRE-12)

---

## [0.3.15] - 2026-02-14

### Security

#### Authentication Hardening (AUTH-HARD-1)

- **Single JWT secret source of truth** — New `jwt_singleton.py` with `get_jwt_manager()` replaces scattered JWTManager instantiation across all services
- **Fail-loud production guard** — `RuntimeError` if `JWT_SECRET_KEY` is missing or shorter than 32 characters in production
- **Fixed logout** — `/auth/logout` now actually revokes tokens via `revoke_all_user_tokens()`
- **Fixed `init_superadmin.py`** — Was writing to `studio` database instead of `smartmemory`
- **Fixed bare `except:`** — Middleware no longer silently swallows auth errors
- **Eliminated 3 hardcoded JWT defaults** — Replaced by `DEV_JWT_SECRET` constant for dev/test
- **Unified `AUTH_JWT_SECRET` → `JWT_SECRET_KEY`** — Single env var name across all config files and docker-compose
- **Eliminated dead domain `app.smartmemory.ai`** — All references updated to `www.smartmemory.ai`
- **Fixed `datetime.utcnow()` in auth paths** — Migrated to `datetime.now(timezone.utc)` in jwt.py, jwt_provider.py, scope.py
- **Removed `dummy` auth provider default** — Core config.json now defaults to `jwt`

### Added

- `service_common/auth/constants.py` — `DEV_JWT_SECRET` constant
- `service_common/auth/jwt_singleton.py` — Singleton `JWTManager` with env-based resolution
- `tests/invariants/test_jwt_singleton.py` — 19 invariant tests for singleton lifecycle

---

## [0.3.14] - 2026-02-14

### Added

#### Bulk Graph API (CORE-BULK-1)

- `FalkorDBBackend.add_nodes_bulk()` — UNWIND Cypher batch upserts, grouped by label, chunked at 500 items
- `FalkorDBBackend.add_edges_bulk()` — UNWIND Cypher batch upserts, grouped by edge type, chunked at 500 items
- `FalkorDBBackend._chunked()` — static helper for splitting lists into fixed-size chunks
- `SmartGraph.add_nodes_bulk()` / `add_edges_bulk()` — delegation methods with cache clearing
- New `POST /memory/graph/bulk` endpoint — general-purpose bulk write with optional `delete_prefix` for clean-replace semantics
- Pydantic models: `BulkNode`, `BulkEdge`, `BulkWriteRequest`, `BulkWriteResponse`

### Changed

- `CodeIndexer.index()` — replaced individual `graph.add_node()`/`add_edge()` loops with `add_nodes_bulk()`/`add_edges_bulk()` calls
- `POST /memory/code/index` — refactored to use bulk graph methods instead of individual writes
- Performance: ~14,000 individual graph calls replaced by batched UNWIND queries

---

## [0.3.13] - 2026-02-13

### Added

#### Code Connector Prototype (CORE-CODE-1)

- New `code` memory type added to `MEMORY_TYPES` — source code entities from code indexing
- New `smartmemory.code` module with AST-based Python parser, code indexer, and data models
- `CodeParser` — extracts classes, functions, imports, FastAPI routes, pytest tests from Python files using `ast`
- `CodeIndexer` — orchestrates parsing + graph writes for entire directories with clean-slate re-indexing
- `CodeEntity` / `CodeRelation` models with deterministic IDs (`code::{repo}::{file}::{name}`)
- 3 new MCP tools: `code_index` (trigger indexing), `code_search` (find by name/type), `code_dependencies` (trace dependents/dependencies)
- Updated `contracts/entity-types.json` — added `module`, `class`, `function`, `route`, `test` to `technical_systems` category
- Validated against `smart-memory-service/`: 1,734 entities (116 classes, 258 functions, 266 routes, 945 tests), 1,874 valid edges

#### Ontology Improvements (STU-OL-2, STU-OL-3, STU-OL-4)

**Ontology Version Tracking (OL-2):**
- `PipelineState` gains `ontology_registry_id` and `ontology_version` fields
- `OntologyConstrainStage` captures version from `OntologyRegistry` at pipeline execution time
- `StoreStage` injects ontology version into MemoryItem metadata
- `SmartMemory` exposes `last_ontology_version` and `last_ontology_registry_id` properties

**Property Constraint Enforcement (OL-3):**
- `PropertyConstraint` dataclass — required, type (string/number/date/boolean/enum), cardinality (one/many), kind (soft/hard), enum_values
- `EntityTypeDefinition` gains `property_constraints: Dict[str, PropertyConstraint]`
- `Ontology.to_dict()`/`from_dict()` serialize and parse constraints (backward-compatible)
- `OntologyConstrainStage._validate_constraints()` — checks required, type, enum, cardinality; soft violations warn, hard violations reject
- `SmartMemory` exposes `last_constraint_violations` property

**Unresolved Entity Reporting (OL-4):**
- Structured rejection dicts: `entity_name`, `attempted_type`, `reason`, `confidence`, `source_content`
- Reasons: `unknown_type`, `low_confidence`, `constraint_violation`
- `SmartMemory` exposes `last_unresolved_entities` property
- Ingest API response includes `unresolved_entities` and `constraint_violations` fields

---

## [0.3.12] - 2026-02-12

### Added

#### Evolution Timeline & Recommendation Engine (CORE-CFS-3b)

**Evolution Timeline** (`smartmemory/evolution/`):
- `EvolutionEvent` model — tracks procedure version history (created, updated, refined, merged, superseded)
- `ContentSnapshot` — captures procedure content at each version (content, name, description, skills, tools, steps)
- `EventDiff` — semantic diff engine for detecting changes between versions (added/removed skills, tools, steps)
- `EvolutionTracker` — orchestrates event creation and retrieval
- `EvolutionStore` — MongoDB persistence for evolution events

**Recommendation Engine** (`smartmemory/procedures/`):
- `PatternDetector` — detects repeated patterns in working memory for procedure promotion candidates
- `CandidateScorer` — multi-metric scoring (frequency 0.40, consistency 0.30, recency 0.20, agent_workflow 0.10)
- `CandidateNamer` — auto-generates procedure names and descriptions from cluster content
- `ProcedureCandidate` model with stable cluster IDs (SHA-256 hash of item IDs for promote/dismiss flow)

### Fixed

- **Cluster ID stability**: Changed from random UUIDs to deterministic hashes, enabling reliable promote/dismiss flow
- **Deprecated datetime.utcnow()**: Updated to timezone-aware `datetime.now(timezone.utc)`

---

## [0.3.11] - 2026-02-12

### Added

#### Self-Healing Procedures (CORE-CFS-4)

**Schema Diff Engine** (`smartmemory/schema_diff.py`):
- `diff_schemas()` — JSON Schema-level diff detecting added/removed/modified/type-changed properties
- `diff_tool_schemas()` — Multi-tool diff aggregating per-tool changes with breaking change classification
- `SchemaChange` and `SchemaDiffResult` dataclasses for structured diff output

**Schema Snapshots & Drift Events** (`smartmemory/models/`):
- `SchemaSnapshot` model — captures tool signatures with SHA-256 hash for change detection
- `DriftEvent` model — records detected schema drift with breaking/non-breaking change counts

**Schema Providers** (`smartmemory/schema_providers.py`):
- `SchemaProvider` protocol — pluggable interface for resolving current tool schemas
- `StaticSchemaProvider` — dictionary-backed provider for testing and static configurations

**Drift Detector** (`smartmemory/drift_detector.py`):
- `DriftDetector` — stateless detector comparing snapshots against current schemas
- `DriftDetectorConfig` — configuration for enable/disable and confidence multiplier
- Confidence degradation: `apply_confidence()` reduces match confidence when drift detected

**Procedure Match Extensions**:
- Added `drift_detected`, `drift_event_id`, `effective_confidence` fields to `ProcedureMatchResult`
- `SmartMemory.drift_detector` property for accessing the detector instance

---

## [0.3.10] - 2026-02-12

### Added

#### Procedure Analytics Dashboard (CORE-CFS-3)

**Backend API:**
- **`GET /memory/procedures`** (`smart-memory-service/memory_service/api/routes/procedures.py`): Lists all procedural memories with aggregated match statistics (total matches, success rate, avg confidence, tokens saved). Supports pagination (limit, offset) and sorting (by match_count, success_rate, created_at, name)
- **`GET /memory/procedures/{id}`** (`procedures.py`): Returns procedure detail with recent match history. Query params: `include_matches` (default true), `match_limit` (default 20)
- **Contract file** (`contracts/procedures.json`): API contract defining Procedure, ProcedureDetail, MatchStats, MatchRecord types

**Python SDK:**
- **`list_procedures(limit, offset, sort_by, sort_order)`** (`smart-memory-client/smartmemory_client/client.py`): List procedures with pagination and sorting
- **`get_procedure(procedure_id, include_matches, match_limit)`** (`client.py`): Get procedure detail with optional match history

**Frontend (smart-memory-insights):**
- **Procedures page** (`web/src/pages/Procedures.jsx`): New analytics page with 30-second auto-refresh
- **ProcedureStatsCards** (`web/src/components/procedures/ProcedureStatsCards.jsx`): Stats cards showing Total Procedures, Total Matches, Success Rate, Tokens Saved with loading skeletons
- **ProcedureMatchChart** (`web/src/components/procedures/ProcedureMatchChart.jsx`): Dual-axis time-series chart (matches on left, tokens saved on right) using recharts
- **ProcedureCatalog** (`web/src/components/procedures/ProcedureCatalog.jsx`): Searchable, sortable table with 300ms debounced search, pagination (Load More), and empty states
- **ProcedureDetail** (`web/src/components/procedures/ProcedureDetail.jsx`): Slide-out Sheet panel showing procedure content, stats, recent matches with feedback buttons
- **Navigation item** (`web/src/pages/Layout.jsx`): Added "Procedures" to sidebar with Workflow icon

**JS Client:**
- **`getProcedures(limit, offset, sortBy, sortOrder)`** (`web/src/api/smartMemoryClient.js`): Fetch procedures list
- **`getProcedure(procedureId)`** (`smartMemoryClient.js`): Fetch procedure detail
- **`getProcedureMatchHistory(limit, procedureId)`** (`smartMemoryClient.js`): Fetch match history for chart data

---

## [0.3.9] - 2026-02-11

### Security

#### Temporal Query Tenant Isolation (SEC-TEMPORAL-1)

- **Fixed cross-tenant data leak in `TemporalQueries`** (`smartmemory/temporal/queries.py`): Added `_get_node_scoped()` helper method that enforces workspace isolation when retrieving nodes
- **`compare_versions()` now respects tenant boundaries**: Uses scoped node retrieval to prevent cross-workspace version comparison
- **`rollback()` now respects tenant boundaries**: Uses scoped node retrieval to prevent cross-workspace item access and modification
- **OSS mode backward compatibility**: When no scope provider is configured, temporal queries work as before (no restrictions)

### Added

#### Token Cost Instrumentation (CFS-1)

- **`PipelineTokenTracker`** (`smartmemory/pipeline/token_tracker.py`): Per-request tracker that rides on `PipelineState`, recording tokens spent and avoided per pipeline stage with model attribution and cost estimation
- **`StageTokenRecord`** dataclass for individual token events with prompt/completion breakdown, model, and reason fields
- **`token_tracker` field on `PipelineState`**: Automatically created by `PipelineRunner` at pipeline start, populated by stages during execution
- **LLM extract stage tracking**: Records spent tokens after LLM calls, avoided tokens on cache hits and stage disabled
- **Ground stage tracking**: Records avoided tokens when Wikipedia entities resolve from graph instead of API
- **Ontology constrain tracking**: Records avoided tokens on entity-pair cache hits
- **`get_last_usage()` in DSPy adapter**: Thread-local consume-once accessor for LLM token usage from DSPy calls
- **`SmartMemory.last_token_summary` property**: Access the most recent pipeline run's token summary
- **Extended `COST_PER_1K_TOKENS`** in `utils/token_tracking.py`: Added Groq Llama, Gemini Flash, Claude Haiku pricing
- **Embedding token tracking (CFS-1a)**: Added `EmbeddingUsage` dataclass and `get_last_embedding_usage()` thread-local accessor in `plugins/embedding.py`; store stage now tracks embedding tokens as spent (API calls) or avoided (cache hits)
- **Enricher LLM token tracking (CFS-1b)**: Added `usage_tracking.py` module in `plugins/enrichers/` with thread-local accumulator pattern; `TemporalEnricher` and `LinkExpansionEnricher` now record OpenAI token usage; `EnrichStage` collects accumulated usage in finally block for cost attribution

---

## [0.3.8] - 2026-02-11

### Added

#### Graph Integrity — Run Tracking & Entity Type Migration (CORE-GRAPH-INT-1)

- **`run_id` injection in StoreStage** (`smartmemory/pipeline/stages/store.py`): Pipeline now injects `run_id` from raw_metadata into stored item metadata, enabling run-based entity cleanup
- **`SmartGraph.delete_by_run_id(run_id)`** (`smartmemory/graph/smartgraph.py`): New method to delete all nodes created by a specific pipeline run, with workspace scope filtering
- **`SmartGraph.rename_entity_type(old, new)`** (`smartmemory/graph/smartgraph.py`): Bulk-rename entity types across all matching nodes in the graph
- **`SmartGraph.merge_entity_types(sources, target)`** (`smartmemory/graph/smartgraph.py`): Merge multiple entity types into a single target type
- **`SmartMemory.delete_run(run_id)`** (`smartmemory/smart_memory.py`): Public API to delete all entities from a specific pipeline run
- **`SmartMemory.rename_entity_type(old, new)`** (`smartmemory/smart_memory.py`): Public API for ontology evolution — rename entity types in-place
- **`SmartMemory.merge_entity_types(sources, target)`** (`smartmemory/smart_memory.py`): Public API to consolidate fragmented entity types

---

## [0.3.7] - 2026-02-10

### Fixed

- **Evolution typed config**: Fixed `EvolutionOrchestrator.commit_working_to_episodic()` and `commit_working_to_procedural()` to pass typed config objects (`WorkingToEpisodicConfig`, `WorkingToProceduralConfig`) instead of raw dicts — eliminates TypeError from evolvers expecting typed configs

---

## [0.3.6] - 2026-02-08

### Changed

#### Corpus → Library Rename (STU-UX-13 V1)
- **Renamed `Corpus` model to `Library`** (`smartmemory/models/library.py`): New primary model with `retention_policy`, `retention_days` fields
- **Renamed `corpus_id` to `library_id`** in `Document` model, added `content_hash`, `storage_phase`, `evict_at` fields
- **Backward-compat shim** (`smartmemory/models/corpus.py`): Re-exports `Library` as `Corpus` for existing imports
- **Updated `CorpusStore`** (`smartmemory/stores/corpus/store.py`): Now operates on `Library` objects, supports both `library_id` and `corpus_id` filters
- **Updated `models/__init__.py`**: Exports `Library` and `Document`

---

## [0.3.5] - 2026-02-07

### Added

#### Batch Ontology Update — Core Metadata (WS-8 Phase 1)
- **`extraction_status` pipeline field** (`smartmemory/pipeline/state.py`): New `extraction_status` field on `PipelineState` tracking whether LLM extraction ran (`ruler_only`, `llm_enriched`, `llm_failed`)
- **`LLMExtractStage` sets extraction_status** (`smartmemory/pipeline/stages/llm_extract.py`): Status set on all code paths — disabled, empty text, success, and failure
- **`StoreStage` injects extraction_status into metadata** (`smartmemory/pipeline/stages/store.py`): `extraction_status` flows through to FalkorDB node properties for batch worker queries
- **`ensure_extraction_indexes()`** (`smartmemory/graph/indexes.py`): Standalone index utility for creating FalkorDB index on `extraction_status`, callable from service worker or tests

#### Auto-assign Team and Workspace on Signup
- **`TeamModel`** (`service_common/models/auth.py`): New dataclass for persisting teams in MongoDB `teams` collection
- **`UserModel.default_team_id`** (`service_common/models/auth.py`): New optional field storing the user's default team assignment
- **Team persistence on signup** (`service_common/services/auth_service.py`): Default team is now created and persisted in MongoDB during user registration
- **Team context in JWT auth** (`service_common/auth/jwt_provider.py`): `ServiceUser` is now populated with `team_memberships`, `current_team_id`, and `current_team_role` at token authentication time (previously only patched in by `get_service_user()`)
- **Workspace-team linking** (`service_common/repositories/workspace_repo.py`): `ensure_personal_workspace()` now accepts `default_team_id` and grants team admin access to the personal workspace
- **API key auth includes team context** (`service_common/auth/jwt_provider.py`): `_authenticate_api_key` now populates team memberships on ServiceUser (consistent with JWT flow)
- **`/auth/me` returns `default_team_id`** (`memory_service/api/routes/auth.py`): UserResponse now always includes the user's default team ID

#### Team/Workspace Roadmap (WS-9)
- **WS-9 workstream** (`docs/plans/2026-02-04-reasoning-system-roadmap.md`): Added Team & Workspace Management roadmap entry — auto-assignment done, remaining CRUD APIs, invitation flow, multi-team, admin UI

### Removed
- **Backward compatibility fallbacks**: Removed all deterministic `team_{tenant_suffix}` fallback computations from `jwt_provider.py`, `auth_service.py`, `middleware.py`, `auth.py` (service + maya), and `core.py` — no users exist, clean slate
- **Workspace backfill logic**: Removed `elif` branch in `ensure_personal_workspace()` that backfilled team access on existing workspaces
- **In-memory team patching in `get_service_user()`**: Removed fallback that computed team membership when `ServiceUser.team_memberships` was empty — now handled at authentication time by `JWTAuthProvider`

### Changed

#### Test Suite Sweep
- **Removed dead code**: Deleted 4 debug/benchmark scripts from `tests/` (zero test functions), removed empty `tests/e2e/` scaffold (11 dirs with only `__init__.py`)
- **Trimmed conftest**: Removed ~150 lines of unused fixtures from `tests/conftest.py` (sample_memory_items, sample_entities, sample_relations, sample_embeddings, conversation_context, conversation_manager, mock_smartmemory_dependencies, clean_memory, TestDataFactory)
- **Relocated orphaned tests**: Moved `tests/plugins/evolvers/test_opinion_synthesis.py` and `tests/plugins/extractors/test_reasoning.py` to proper `tests/unit/plugins/` locations
- **Marked deprecated extractors**: SpacyExtractor and RelikExtractor test classes marked `@pytest.mark.skip(reason="deprecated extractor")`
- **Coverage config**: Added `--cov=smartmemory --cov-report=term-missing`, marker registration, and `pytest-cov` dev dependency to `pyproject.toml`
- **Removed duplicate tests**: Removed TestOpinionMetadata, TestObservationMetadata, TestDisposition from `test_opinion_synthesis.py` (covered by `tests/unit/models/test_opinion.py`)
- **Removed tautology tests**: Deleted TestExtractorComparison class (3 no-op self-equality assertions)
- **Removed redundant conftest markers**: Marker assignments now handled by `pyproject.toml`

### Added

#### Ontology Layers: Public Base + Private Overlay
- **`OntologySubscription` model** (`smartmemory/ontology/models.py`): Tracks base registry subscription with pinned version and hidden types; serializes to/from dict with backward compatibility
- **`LayerDiff` model** (`smartmemory/ontology/models.py`): Dataclass representing diff between base and overlay layers (base_only, overlay_only, overridden, hidden)
- **`LayeredOntology` class** (`smartmemory/ontology/layered.py`): Merged read view of base + overlay ontologies — entity-level override (overlay wins), rules not inherited from base, hidden types excluded, provenance tracking (local/base/override/hidden), detach to flatten
- **`LayeredOntologyService`** (`memory_service/services/layer_service.py`): Subscription lifecycle orchestration — subscribe, unsubscribe (flatten), pin/unpin version, hide/unhide types, diff computation (service layer, not core)
- **8 layer API endpoints** (`memory_service/api/routes/ontology_layers.py`): `POST/DELETE /ontology/registry/{id}/subscribe`, `PUT/DELETE /ontology/registry/{id}/subscribe/pin`, `POST /ontology/registry/{id}/subscribe/hidden`, `DELETE /ontology/registry/{id}/subscribe/hidden/{type_name}`, `GET /ontology/registry/{id}/layers`, `GET /ontology/registry/{id}/layer-diff`
- **75 tests**: 59 unit tests (models, LayeredOntology, LayeredOntologyService) + 16 API endpoint tests

- **Archive API endpoints**: `POST /api/archive/store` and `GET /api/archive/{uri}` for durable conversation artifact storage
- **Graph path endpoint**: `GET /api/graph/path` for shortest-path queries between knowledge graph nodes
- **`SmartMemory.find_shortest_path()`**: Public method for graph path traversal using FalkorDB `shortestPath()` Cypher
- **`SecureSmartMemory` proxy methods**: `archive_put`, `archive_get`, `find_shortest_path` delegate to core with tenant scoping

- **Model tests**: `test_memory_item.py`, `test_entity.py`, `test_opinion.py`, `test_reasoning.py` — 82 tests covering creation, validation, serialization, and key behaviors
- **Observability tests**: `test_events.py`, `test_json_formatter.py`, `test_logging_filter.py` — 27 tests covering EventSpooler, JsonFormatter, LogContextFilter (all mocked, no Docker)
- **Evolver tests**: `test_episodic_to_semantic.py`, `test_working_to_episodic.py`, `test_episodic_decay.py`, `test_decision_confidence.py` — 71 tests covering metadata, config, and evolution logic

### Security

- **Workspace access fail-closed** (`service_common/auth/scope.py`): `_can_access_workspace` now returns `False` on exception instead of `True` — prevents tenant isolation bypass when MongoDB is unreachable
- **Team ID exact match validation** (`service_common/auth/scope.py`): Changed substring `in` check to exact `==` for team ID validation — prevents cross-tenant access via crafted team IDs
- **OAuth random password** (`memory_service/api/routes/auth.py`, `maya/auth/routes.py`): OAuth users now get `secrets.token_urlsafe(32)` instead of static `"oauth-no-password"` — prevents password-based login to OAuth accounts
- **Open redirect prevention** (`memory_service/api/routes/auth.py`): `frontend_callback` parameter validated against allowed origins (FRONTEND_URL, MAYA_FRONTEND_URL, CORS_ORIGINS, localhost) — prevents token theft via crafted OAuth login link
- **Password reset method fix** (`memory_service/api/routes/auth.py`): Fixed `invalidate_all_refresh_tokens` → `revoke_all_user_tokens` (was calling non-existent method)
- **Ontology tenant isolation**: Added `validate_registry_ownership()` to 5 registry endpoints that were missing tenant checks (`apply_changeset`, `list_registry_snapshots`, `get_registry_changelog`, `rollback_registry`, `import_registry_snapshot`). Fixed `export_registry_snapshot` which had no authentication at all.
- **WebSocket authentication**: Added JWT auth to `/ws/feedback` endpoint via `?token=<jwt>` query param. Invalid/missing tokens rejected with close code 1008 (Policy Violation).
- **Ontology access model**: Formalized workspace-shared access model — `created_by` is audit metadata, not an access gate. Removed 6 TODO comments and converted warning logs to informational audit logs.

### Added

#### Decision Confidence Evolver v2.0
- **Evidence-based reinforcement/contradiction** (`smartmemory/plugins/evolvers/decision_confidence.py`): Enhanced from decay-only to full evidence matching — keyword (content, domain, tags), semantic (cosine similarity), and contradiction signal detection against recent episodic/semantic/opinion memories
- **Decision model integration**: Uses `Decision.reinforce()` and `Decision.contradict()` for confidence math with diminishing returns and proportional penalties, persists full `Decision.to_dict()` state
- **Registry registration** (`smartmemory/evolution/registry.py`): Registered as `decision_confidence` so it runs in the evolution pipeline
- **Duplicate episodic prevention**: `fetched_episodic` flag skips episodic in standard search loop when the dedicated episodic API succeeds
- **47 unit tests** (`tests/unit/decisions/test_decision_evolver.py`): Covering reinforcement, contradiction, decay, retraction, evidence matching (keyword/domain/tag/semantic), full cycle, staleness, and registry integration

#### Ontology Templates
- **Template catalog** (`smartmemory/ontology/template_service.py`): `TemplateService` — browse, preview, clone, save-as-template, delete custom templates. Built-in templates cached in memory, custom templates stored via `OntologyStorage` with `is_template` flag
- **Built-in templates** (`smartmemory/ontology/templates/`): 3 curated ontology templates — General Purpose (12 entity types, 8 relationships), Software Engineering (15 entity types, 10 relationships), Business & Finance (14 entity types, 9 relationships)
- **Template data models** (`smartmemory/ontology/models.py`): `TemplateInfo` and `TemplatePreview` dataclasses for catalog browsing and preview
- **Template API endpoints** (`memory_service/api/routes/ontology.py`): 5 endpoints — `GET /templates` (list), `GET /templates/{id}` (preview), `POST /templates/clone` (clone into workspace), `POST /templates` (save as template), `DELETE /templates/{id}` (delete custom)
- **Ontology model extensions**: `is_template` and `source_template` fields on `Ontology`, serialized via `to_dict`/`from_dict`, included in `FileSystemOntologyStorage.list_ontologies`

### Fixed

- **Ontology deserialization bug**: Fixed operator precedence in `Ontology.from_dict()` — `data.get('x') or {}.items()` was iterating raw dict keys instead of key-value pairs. Added parentheses: `(data.get('x') or {}).items()`

#### Studio Pipeline UI (Phase 7)
- **Learning Page backend** (`smart-memory-studio/server/memory_studio/api/routes/learning.py`): 8 API endpoints — stats, convergence, promotions (with approve/reject), patterns, types, activity feed — all proxying to core OntologyGraph with tenant isolation
- **Learning Page frontend** (`smart-memory-studio/web/src/pages/Learning.jsx`): Full dashboard with stats cards, promotion queue (approve/reject actions), type registry (searchable/sortable), pattern browser (layer filtering), activity feed (auto-refresh 30s)
- **Learning hooks** (`smart-memory-studio/web/src/hooks/learning/`): 5 data hooks — useLearningStats, useLearningPromotions, useLearningPatterns, useLearningTypes, useLearningActivity
- **LearningService** (`smart-memory-studio/web/src/services/LearningService.js`): API client for all learning endpoints
- **PipelineConfigEditor** (`smart-memory-studio/web/src/components/pipeline/PipelineConfigEditor.jsx`): Accordion-based config editor matching PipelineConfig dataclass, save/load named profiles via Profiles API
- **ConfigField** + **StageConfigSection**: Smart form field renderer (switch/number/text/select by type) and collapsible per-stage sections with reset-to-default
- **BreakpointRunner** (`smart-memory-studio/web/src/components/pipeline/BreakpointRunner.jsx`): Debug pipeline runner — set breakpoints on any of 11 stages, run-to/resume/undo with intermediate state inspection
- **PipelineStepper** + **StateInspector**: Visual step indicator with status icons and tabbed state viewer (entities, relations, text, timings)
- **PromptsPanel polish**: Character count, copy-to-clipboard button, last-modified timestamp, improved empty state messaging
- **Navigation**: Learning page with TrendingUp icon added to Studio nav

#### Insights + Observability & Decision Memory UI (Phase 6)
- **`MetricsConsumer`** (`smartmemory/pipeline/metrics_consumer.py`): Reads pipeline metric events from Redis Streams, aggregates into 5-min time buckets, writes pre-aggregated metrics to Redis Hashes for fast dashboard reads
- **Pipeline Metrics API** (`smart-memory-insights/server/observability/api.py`): `GET /api/pipeline/metrics`, `/api/pipeline/bottlenecks` — per-stage latency, throughput, error rates from MetricsConsumer aggregated data
- **Ontology Metrics API**: `GET /api/ontology/status`, `/api/ontology/growth`, `/api/ontology/promotion-rates` — direct FalkorDB Cypher queries for type/pattern counts, convergence estimates, promotion rates
- **Extraction Quality API**: `GET /api/extraction/quality`, `/api/extraction/attribution` — confidence distribution histograms, provisional ratio, ruler vs LLM attribution from graph entity data
- **Pipeline Performance dashboard** (`smart-memory-insights/web/src/pages/Pipeline.jsx`): Summary cards, per-stage latency bar chart, throughput area chart, error rate chart with 1H/6H/24H period selector
- **Ontology Health dashboard** (`smart-memory-insights/web/src/pages/Ontology.jsx`): Summary cards, type status donut, pattern layers donut, convergence curve, pattern growth area chart, promotion rates bar chart
- **Extraction Quality dashboard update** (`smart-memory-insights/web/src/pages/Extraction.jsx`): ConfidenceDistribution histogram, AttributionChart pie, TypeRatioChart bar replacing previous stub data
- **Decision Memory client SDK** (`smart-memory-client`): `create_decision()`, `get_decision()`, `list_decisions()`, `supersede_decision()`, `retract_decision()`, `reinforce_decision()`, `get_provenance_chain()`, `get_causal_chain()` methods
- **Decision Memory web UI** (`smart-memory-web`): DecisionCard, DecisionList, ProvenanceChain components; standalone `/Decisions` route with Scale icon + amber accent in navigation
- **Insights navigation**: Pipeline and Ontology pages added to sidebar and router

#### Pipeline v2 Service + API (Phase 5)
- **Pipeline Config CRUD** (`smart-memory-service/memory_service/api/routes/pipeline.py`): Named pipeline config management — `GET/POST/PUT/DELETE /memory/pipeline/configs` with FalkorDB storage as `PipelineConfig` nodes in the ontology graph
- **`OntologyGraph` pipeline config storage**: `save_pipeline_config()`, `get_pipeline_config()`, `list_pipeline_configs()`, `delete_pipeline_config()` for persistent named configs per workspace
- **Pattern Admin routes** (`smart-memory-service/memory_service/api/routes/ontology.py`): `GET/POST/DELETE /memory/ontology/patterns` for entity pattern management with stats, Redis pub/sub reload notifications
- **`OntologyGraph.delete_entity_pattern()`**: Delete EntityPattern nodes with workspace scoping
- **Ontology Status endpoint** (`GET /memory/ontology/status`): Convergence metrics — pattern counts by source layer, entity type status distribution (seed/provisional/confirmed)
- **Ontology Import/Export** (`POST /memory/ontology/import`, `GET /memory/ontology/export`): Portable ontology data for backup/migration with entity type and pattern support
- **`EventBusTransport`** (`smartmemory/pipeline/transport/event_bus.py`): Redis Streams transport for async pipeline execution with per-stage consumer groups, status polling, and result retrieval
- **`StageConsumer`**: Worker class for consuming and executing pipeline stages from Redis Streams
- **Async ingest mode**: `POST /memory/ingest?mode=async` submits via `EventBusTransport`, returns `run_id` for polling; `GET /memory/ingest/async/{run_id}` for status
- **`StageRetryPolicy`** (`smartmemory/pipeline/config.py`): Per-stage retry policy with configurable `backoff_multiplier`, overrides global `RetryConfig`
- **`PipelineConfig.stage_retry_policies`**: Dict mapping stage names to `StageRetryPolicy` for fine-grained retry control

### Changed
- **`PipelineRunner._execute_stage()`** now checks `config.stage_retry_policies` for per-stage overrides, calls `stage.undo()` on retry exhaustion before raising or skipping
- **`transport.py` → `transport/` package**: Converted single file to package to accommodate `event_bus.py`; backward-compatible imports preserved

#### Pipeline v2 Self-Learning Loop (Phase 4)
- **`PromotionEvaluator`** (`smartmemory/ontology/promotion.py`): Six-gate evaluation pipeline for entity type promotion — min name length, common word blocklist (~200 words), min confidence, min frequency, type consistency, optional LLM reasoning validation
- **`PromotionWorker`** (`smartmemory/ontology/promotion_worker.py`): Background Redis Stream consumer for async promotion candidate processing with DLQ error handling
- **`PatternManager`** (`smartmemory/ontology/pattern_manager.py`): Hot-reloadable dictionary of learned entity patterns with Redis pub/sub reload notifications and common word filtering
- **`EntityPairCache`** (`smartmemory/ontology/entity_pair_cache.py`): Redis read-through cache for known entity-pair relations with 30-min TTL, graph fallback, and invalidation
- **`ReasoningValidator`** (`smartmemory/ontology/reasoning_validator.py`): LLM-based validation for promotion candidates with ReasoningTrace storage as `reasoning` memory type
- **`OntologyGraph` frequency tracking**: `increment_frequency()`, `get_frequency()` for atomic type frequency and running average confidence tracking
- **`OntologyGraph` entity patterns**: `add_entity_pattern()`, `get_entity_patterns()`, `get_type_assignments()` for three-layer pattern storage (seed / promoted / tenant)
- **`OntologyGraph` pattern layers**: `seed_entity_patterns()` for bootstrap, `get_pattern_stats()` for layer counts
- **`RedisStreamQueue.for_promote()`**: Factory method for promotion job queue (follows `for_enrich()` pattern)
- **`_ngram_scan()`**: N-gram dictionary scan (up to 4-grams) for entity pattern matching in `EntityRulerStage`

### Changed
- **`OntologyConstrainStage`** now tracks entity type frequency on every accepted entity, enqueues promotion candidates to Redis stream (with inline fallback), and injects cached entity-pair relations
- **`EntityRulerStage`** now accepts optional `PatternManager` for learned pattern dictionary scan after spaCy NER
- **`PromotionConfig`** extended with `min_type_consistency` (0.8) and `min_name_length` (3) fields
- **`SmartMemory._create_pipeline_runner()`** wires `PatternManager`, `EntityPairCache`, and promotion queue into the pipeline

#### Pipeline v2 Storage & Post-Processing (Phase 3)
- **`PipelineMetricsEmitter`** (`smartmemory/pipeline/metrics.py`): Fire-and-forget pipeline metrics emission via Redis Streams. Emits `stage_complete` events per stage and `pipeline_complete` summary with timing breakdown, entity/relation counts, and workspace context. Stream: `smartmemory:metrics:pipeline`
- **`SmartMemory.create_pipeline_runner()`**: Public factory method exposing the v2 `PipelineRunner` for Studio and other consumers that need `run_to()`/`run_from()` breakpoint execution
- **Studio `get_v2_runner()`** (`pipeline_registry.py`): Cached v2 `PipelineRunner` per tenant for Studio backend, backed by `SecureSmartMemory`
- **Studio `run_extraction_v2()`** (`extraction.py`): Extraction preview via `PipelineRunner.run_to("ontology_constrain")` — replaces old pipeline path for extraction previews

### Changed
- **`PipelineRunner`** accepts optional `metrics_emitter` parameter — calls `on_stage_complete()` after each stage and `on_pipeline_complete()` after full run
- **Extraction preview** in Studio (`transaction.py`) now uses v2 `PipelineRunner` with fallback to v1 `ExtractorPipeline`
- **Normalization deduplication**: `EnrichmentPipeline` and `StorageEngine` now use canonical `sanitize_relation_type()` from `memory.ingestion.utils` instead of weak local `_sanitize_relation_type()` (space→underscore only). Relation types now get regex normalization, uppercase, FalkorDB-safe prefix, and 50-char limit

### Deprecated
- **`ExtractorPipeline`** (`smartmemory/memory/pipeline/extractor.py`): Emits `DeprecationWarning` on instantiation. Use `smartmemory.pipeline.stages` via `PipelineRunner` instead. Full removal deferred to Phase 7

#### Pipeline v2 Native Extraction (Phase 2)
- **4 native extraction stages** replace the single `ExtractStage` wrapper — pipeline grows from 8 to 11 stages:
  - `SimplifyStage` — splits complex sentences into atomic statements using spaCy dependency parsing with configurable transforms (clause splitting, relative clause extraction, appositive extraction)
  - `EntityRulerStage` — rule-based entity extraction using spaCy NER with label mapping, deduplication, and confidence filtering
  - `LLMExtractStage` — LLM-based entity/relation extraction via `LLMSingleExtractor` with configurable truncation limits
  - `OntologyConstrainStage` — merges ruler + LLM entities, validates types against `OntologyGraph`, filters relations by accepted endpoints, auto-promotes provisional types
- **Extended `PipelineConfig`** with Phase 2 fields:
  - `SimplifyConfig` — 4 boolean transform flags, `min_token_count`
  - `EntityRulerConfig` — `pattern_sources`, `min_confidence`, `spacy_model`
  - `LLMExtractConfig` — `max_relations`
  - `ConstrainConfig` — `domain_range_validation`
  - `PromotionConfig` — `reasoning_validation`, `min_frequency`, `min_confidence`
- **`PipelineState.simplified_sentences`** — renamed from `simplified_text` (Optional[str]) to `List[str]` for multi-sentence output
- **Studio backend models** — `SimplifyRequest`, `EntityRulerRequest`, `OntologyConstrainRequest` dataclasses added; pipeline component descriptions updated

#### Pipeline v2 Foundation (Phase 1)
- **New unified pipeline architecture** (`smartmemory/pipeline/`): Replaces three separate orchestrators with a single composable pipeline built on the StageCommand protocol
  - `StageCommand` protocol (structural subtyping) with `execute()` and `undo()` methods
  - `PipelineState` immutable-by-convention dataclass with full serialization (`to_dict()`/`from_dict()`)
  - `PipelineConfig` nested dataclass hierarchy with named factories (`default()`, `preview()`)
  - `Transport` protocol with `InProcessTransport` for in-process execution
  - `PipelineRunner` with `run()`, `run_to()`, `run_from()`, `undo_to()` — supports breakpoints, resumption, rollback, and per-stage retry with exponential backoff
- **11 stage wrappers** (`smartmemory/pipeline/stages/`): classify, coreference, simplify, entity_ruler, llm_extract, ontology_constrain, store, link, enrich, ground, evolve
- **OntologyGraph** (`smartmemory/graph/ontology_graph.py`): Dedicated FalkorDB graph for entity type definitions with three-tier status (seed → provisional → confirmed) and 14 seed types

### Changed
- `SmartMemory.ingest()` now delegates to `PipelineRunner.run()` instead of `MemoryIngestionFlow.run()` — identical output, new internal architecture
- Pipeline stage count increased from 8 to 11 with native extraction stages

### Removed
- `ExtractStage` (`smartmemory/pipeline/stages/extract.py`) — replaced by 4 native extraction stages (simplify, entity_ruler, llm_extract, ontology_constrain)
- `ExtractionPipeline` and `IngestionRegistry` no longer needed for pipeline v2 extraction
- `FastIngestionFlow` (`smartmemory/memory/fast_ingestion_flow.py`, 502 LOC) — async ingestion is now a config flag (`mode="async"`) on the unified pipeline

---

## [0.3.2] - 2026-02-05

### Changed

#### Extraction Pipeline
- **Default extractor changed to Groq** (`GroqExtractor`): Llama-3.3-70b-versatile via Groq API — 100% E-F1, 89.3% R-F1, 878ms. Requires `GROQ_API_KEY` env var.
- **New `GroqExtractor` class** in `llm_single.py`: Zero-arg constructor wrapper for registry lazy loading.
- **Fallback chain updated**: groq → llm → llm_single → conversation_aware_llm → spaCy. SpaCy extractor (no API keys needed) re-registered as last-resort fallback.
- **Direct module imports** in registry: Extractors now imported from specific module files (not `__init__.py`) to avoid transitive dependency failures (e.g., GLiNER not installed).
- **Robust fallback iteration**: Extraction pipeline now tries ALL fallbacks (not just the first) when the primary extractor fails to instantiate.
- **`select_default_extractor()` never returns `None`**: Raises `ValueError` with actionable message if no extractors are available.

### Added

#### Graph Validation & Health (Wave 2)
- **New package**: `smartmemory.validation` - Runtime schema validation for memories and edges
  - `MemoryValidator` - Validates memory items against schema constraints (required fields, content length, type validity, metadata types)
  - `EdgeValidator` - Validates edges against registered schemas (allowed source/target types, required metadata, cardinality limits)
- **New package**: `smartmemory.metrics` - Graph health metrics collection
  - `GraphHealthChecker` - Collects orphan ratio, type distribution, edge distribution, provenance coverage via Cypher queries
  - `HealthReport` dataclass with `is_healthy` property (thresholds: orphan < 20%, provenance > 50%)
- **New package**: `smartmemory.inference` - Automatic graph inference engine
  - `InferenceEngine` - Runs pattern-matching rules to create inferred edges with provenance metadata
  - `InferenceRule` dataclass with Cypher pattern, edge type, and confidence
  - 3 built-in rules: causal transitivity, contradiction symmetry, topic inheritance

#### Symbolic Reasoning Layer (Wave 2)
- **New module**: `smartmemory.reasoning.residuation` - Pause reasoning when data is incomplete
  - `ResiduationManager` - Manages pending requirements on decisions; auto-resumes when data arrives via `check_and_resume()`
  - `PendingRequirement` model added to `Decision` with description, created_at, resolved flag
- **New module**: `smartmemory.reasoning.query_router` - Route queries to cheapest effective retrieval
  - `QueryRouter` - Classifies queries as SYMBOLIC (graph Cypher), SEMANTIC (vector search), or HYBRID (both)
  - Pattern-based classification with priority: hybrid > semantic > symbolic
- **New module**: `smartmemory.reasoning.proof_tree` - Auditable reasoning chains
  - `ProofTreeBuilder` - Builds proof trees from graph traversal tracing evidence back to sources
  - `ProofTree` and `ProofNode` with `render_text()` for human-readable proof output
- **New module**: `smartmemory.reasoning.fuzzy_confidence` - Multi-dimensional confidence scoring
  - `FuzzyConfidenceCalculator` - Scores decisions on 4 dimensions: evidence, recency, consensus, directness
  - `ConfidenceScore` with per-dimension breakdown and weighted composite

#### Extended Decision Model (Wave 2)
- `Decision.status` now supports `"pending"` for residuation (decisions awaiting data)
- `Decision.pending_requirements` list for tracking what data is needed
- `PendingRequirement` dataclass with description, created_at, resolved fields
- New edge types: `INFERRED_FROM` (inference provenance), `REQUIRES` (residuation dependencies)

### Changed

#### Evolver Inheritance Cleanup
- Eliminate dual-inheritance in all 12 evolvers: inherit only from `EvolverPlugin` (not `Evolver` + `EvolverPlugin`)
- `Evolver` base class in `plugins/evolvers/base.py` is now a backwards-compatible alias for `EvolverPlugin`

#### AssertionChallenger Strategy Pattern Extraction
- Extract 4 contradiction detection strategies into `reasoning/detection/` package: `LLMDetector`, `GraphDetector`, `EmbeddingDetector`, `HeuristicDetector`
- Extract 4 conflict resolution strategies into `reasoning/resolution/` package: `WikipediaResolver`, `LLMResolver`, `GroundingResolver`, `RecencyResolver`
- Add `DetectionCascade` and `ResolutionCascade` orchestrators for ordered strategy execution
- Extract confidence operations into `ConfidenceManager` class (`reasoning/confidence.py`)
- Slim `AssertionChallenger` from 1,249 to ~200 lines while preserving all public API methods

#### SmartMemory Decomposition
- Extract monitoring operations into `MonitoringManager` (7 methods: summary, orphaned_notes, prune, find_old_notes, self_monitor, reflect, summarize)
- Extract evolution operations into `EvolutionManager` (4 methods: run_evolution_cycle, commit_working_to_episodic, commit_working_to_procedural, run_clustering)
- Extract debug operations into `DebugManager` (4 methods: debug_search, get_all_items_debug, fix_search_if_broken, clear)
- Extract enrichment operations into `EnrichmentManager` (4 methods: enrich, ground, ground_context, resolve_external)
- SmartMemory reduced from 961 to ~750 lines; all public methods preserved as one-line delegates

#### Constructor Injection
- Add optional dependency injection parameters to `SmartMemory.__init__` for all 13 dependencies (graph, crud, search, linking, enrichment, grounding, personalization, monitoring, evolution, clustering, external_resolver, version_tracker, temporal)
- All parameters default to `None` (existing behavior preserved); when provided, injected instances are used instead of creating defaults
- Enables unit testing with mocks without requiring FalkorDB/Redis infrastructure

---

## [0.3.1] - 2026-02-05

### Fixed
- Replace all bare `except:` clauses with `except Exception:` across 6 core files to avoid catching SystemExit/KeyboardInterrupt
- Fix undefined `setFilteredCategories`/`setFilteredNodeTypes` in KnowledgeGraph.jsx (delegate to prop callbacks)
- Fix rules-of-hooks violation in TripleResultsViewer.jsx (conditional `useMemoryStore` call moved to top level)
- Fix undefined `predicateTypes` variable in TripleResultsViewer.jsx (fall back to store value)
- Fix undefined `properties` variable in insights database.py (6 references)
- Add missing `SAMPLE_USER_PROFILE` fixture in maya test_proactive_scheduler.py
- Replace bare `except:` in zettelkasten.py (6 instances), mcp_memory_manager.py (2), background_tasks.py (1)

### Changed
- Standardize ruff config across all Python projects: select `E`, `F`, `B`; ignore `B008`, `E501`
- Fix studio ruff config: move `select` under `[tool.ruff.lint]`, line-length 100 -> 120
- Create ruff configs for maya and insights (previously unconfigured)
- Add `react/prop-types: "off"` to all ESLint configs (web, studio, insights)
- Add `process: "readonly"` global to all ESLint configs

### Version
- Bump all repos to 0.3.1

---

## [0.3.0] - 2026-02-05

### Added

#### Decision Memory System
- **New memory type**: `decision` - First-class decisions with confidence tracking and lifecycle management
- **New model**: `Decision` - Dataclass with provenance, confidence (reinforce/contradict with diminishing returns), and lifecycle (active/superseded/retracted)
- **New module**: `smartmemory.decisions` with:
  - `DecisionManager` - Create, supersede, retract, reinforce, contradict decisions with graph edge management
  - `DecisionQueries` - Filtered retrieval, provenance chains, recursive causal chain traversal
- **New extractor**: `DecisionExtractor` - Regex-based extraction from text + `extract_from_trace()` for ReasoningTrace integration
- **New edge types**: `PRODUCED`, `DERIVED_FROM`, `SUPERSEDES`, `CONTRADICTS`, `INFLUENCES` registered in schema validator
- **Decision types**: inference, preference, classification, choice, belief, policy
- **Conflict detection**: Semantic search + content overlap heuristic for finding contradicting decisions
- **Keyword classification**: Automatic decision type classification from content (no LLM required)
- **Provenance tracking**: Full chain from evidence → reasoning trace → decision → superseded decisions
- **Causal chains**: Recursive traversal of DERIVED_FROM, CAUSED_BY, CAUSES, INFLUENCES, PRODUCED edges with configurable depth
- **New evolver**: `DecisionConfidenceEvolver` - Confidence decay for stale decisions with automatic retraction below threshold
- **Graceful degradation**: All components work without graph (skip edge operations, return empty lists)
- **Service API**: 11 REST endpoints at `/memory/decisions/*` for full decision lifecycle (create, get, list, search, supersede, retract, reinforce, contradict, provenance, causal-chain, conflicts)
- **Maya commands**: `/decide`, `/beliefs`, `/why` with aliases (`/decision`, `/decisions`, `/provenance`, `/explain`)
- **Tests**: 149 unit tests covering model, manager, queries, extractor, and evolver

---

## [Unreleased]

### Added

#### Link Expansion Enricher
- **New enricher**: `LinkExpansionEnricher` - Expands URLs in memory items into rich graph structures
- **URL detection**: Regex-first extraction with fallback to extraction stage URLs
- **Metadata extraction**: Title, description, OG tags, author, published date via BeautifulSoup
- **Entity extraction**: Heuristic extraction from JSON-LD (Schema.org) structured data
- **LLM analysis (optional)**: Summary and deeper entity extraction when `enable_llm=True`
- **Graph integration**: Creates `WebResource` nodes linked to `Entity` nodes via `MENTIONS` edges
- **Error handling**: Failed fetches create nodes with `status='failed'` for retry capability
- **Config**: `LinkExpansionEnricherConfig` with timeout, max URLs, user agent settings

#### Per-Enricher Configuration Support
- **New field**: `EnrichmentConfig.enricher_configs` - Dict mapping enricher names to config dicts
- **Config passthrough**: Enrichment stage now passes per-enricher config to enricher instances
- **Auto config class discovery**: Automatically finds `*Config` class for typed config instantiation
- **Backward compatible**: Enrichers without config or with mismatched config continue to work

#### Claude CLI Integration
- **External package**: `claude-cli` extracted to `regression-io/claude-cli` (private)
- **Optional dependency**: `pip install smartmemory[claude-cli]`
- **No API key required**: Uses Claude subscription authentication via subprocess
- **Simple API**: `claude = Claude(); answer = claude("prompt")`
- **Structured output**: `claude.structured(prompt, schema=MyModel)` with Pydantic
- **Framework adapters**: LangChain and DSPy adapters included
- **Integration**: Available via `LLMClient(provider='claude-cli')`
- **Note**: Experimental, for internal testing

#### Security & Authentication Documentation
- **New documentation**: `docs/SECURITY_AND_AUTH.md` - Comprehensive guide to SmartMemory's security architecture
- **Updated**: `README.md` - Added security/auth examples and references
- **Updated**: `docs/ARCHITECTURE.md` - Enhanced multi-tenancy section with isolation levels

#### System 2 Memory: Reasoning Traces
- **New memory type**: `reasoning` - Captures chain-of-thought reasoning traces
- **New extractor**: `ReasoningExtractor` - Extracts reasoning from Thought:/Action:/Observation: markers or via LLM detection
- **New models**: `ReasoningStep`, `ReasoningTrace`, `ReasoningEvaluation`, `TaskContext`
- **New edge types**: `CAUSES` and `CAUSED_BY` for linking reasoning to artifacts
- **Use case**: Query "why" decisions were made, not just the outcomes

#### Synthesis Memory: Opinions & Observations
- **New memory type**: `opinion` - Beliefs with confidence scores that can be reinforced or contradicted
- **New memory type**: `observation` - Synthesized entity summaries from scattered facts
- **New models**: `OpinionMetadata`, `ObservationMetadata`, `Disposition`
- **New evolvers**:
  - `OpinionSynthesisEvolver` - Forms opinions from episodic patterns
  - `ObservationSynthesisEvolver` - Creates entity summaries from facts
  - `OpinionReinforcementEvolver` - Updates confidence based on new evidence

#### Coreference Resolution Stage (from feed integration)
- **New pipeline stage**: `CoreferenceStage` - Resolves pronouns and vague references to explicit entity names
- **Example**: "Apple announced... The company exceeded..." → "Apple announced... Apple exceeded..."
- **Enabled by default**: Runs automatically in `ingest()` pipeline before entity extraction
- **Uses fastcoref**: High-quality neural coreference resolution
- **Optional dependency**: `pip install smartmemory[coreference]`

#### Conversation-Aware Extraction with Coreference
- **Enhanced extractor**: `ConversationAwareLLMExtractor` now uses fastcoref chains for entity resolution
- **Coreference chains in context**: `ConversationContext` now includes `coreference_chains` field
- **Auto-selection**: Pipeline auto-selects `conversation_aware_llm` extractor when conversation context is present
- **Resolution priority**: Uses fastcoref chains (high quality) before falling back to heuristic resolution
- **LLM context enhancement**: Coreference mappings included in extraction prompts for better entity recognition
- **Configuration**: `CoreferenceConfig` with resolver, device, enabled settings
- **Location**: `smartmemory.memory.pipeline.stages.coreference`
- **Metadata stored**: Original content and coreference chains preserved in item metadata
- **Use case**: Improves entity extraction quality by making implicit references explicit

#### New Exports
- `smartmemory.memory.pipeline.stages`: Added `CoreferenceStage`, `CoreferenceResult`
- `smartmemory.memory.pipeline.config`: Added `CoreferenceConfig`
- `smartmemory.models`: Added `ReasoningStep`, `ReasoningTrace`, `ReasoningEvaluation`, `TaskContext`, `OpinionMetadata`, `ObservationMetadata`, `Disposition`
- `smartmemory.plugins.extractors`: Added `ReasoningExtractor`
- `smartmemory.plugins.evolvers`: Added `OpinionSynthesisEvolver`, `ObservationSynthesisEvolver`, `OpinionReinforcementEvolver`

#### Dependencies
- Added optional dependency group `coreference` with `fastcoref>=2.1.0`

---

## [0.2.6] - 2025-11-24

### 🎯 Major: API Clarification - `ingest()` vs `add()`

**Breaking Changes**: Renamed methods to align behavior with intent

#### Changed

- **`add()` renamed to `ingest()`**: Full agentic pipeline (extract → store → link → enrich → evolve)
- **`_add_basic()` renamed to `add()`**: Simple storage (normalize → store → embed)
- **Removed `ingest_old()`**: Consolidated async queueing into `ingest(sync=False)`

#### API Design

```python
# Full pipeline - use for user-facing ingestion
item_id = memory.ingest("content")  # sync=True by default
result = memory.ingest("content", sync=False)  # async: {"item_id": str, "queued": bool}

# Simple storage - use for internal operations, derived items
item_id = memory.add(item)
```

#### Migration Guide

**Before (v0.2.6)**:
```python
memory.add(item)  # Ran full pipeline (confusing!)
memory._add_basic(item)  # Simple storage (private method)
```

**After (v0.2.6)**:
```python
memory.ingest(item)  # Full pipeline (clear intent)
memory.add(item)     # Simple storage (public, clear intent)
```

#### Internal Callers Updated

- `cli.py`: Now uses `ingest()` for CLI add command
- `mcp_handler.py`: Now uses `ingest()` for external MCP calls
- `evolution.py`: Uses `add()` for evolved items (no re-evolution)
- `enrichment.py`: Uses `add()` for derived items (no re-pipeline)

---

## [0.2.6] - 2025-11-23

### 🎯 Major: Complete Scoping Architecture Refactor

**Breaking Changes**: Method signatures changed - removed all hardcoded scoping parameters

#### Changed

- **`SmartMemory.__init__()`**: Now accepts optional `scope_provider` parameter (defaults to `DefaultScopeProvider()` for OSS usage)
- **`SmartMemory.search()`**: Removed `user_id` parameter - uses `ScopeProvider` exclusively
- **`SmartMemory.personalize()`**: Removed `user_id` parameter - operates on current user via `ScopeProvider`
- **`SmartMemory.run_clustering()`**: Removed `workspace_id`, `tenant_id`, `user_id` parameters
- **`SmartMemory.run_evolution_cycle()`**: Removed `workspace_id`, `tenant_id`, `user_id` parameters
- **`SmartMemory.get_all_items_debug()`**: Removed `tenant_id`, `user_id` parameters - uses `ScopeProvider`
- **`VectorStore.add()`**: Removed `workspace_id` parameter
- **`VectorStore.upsert()`**: Removed `workspace_id` parameter
- **`VectorStore.search()`**: Removed `workspace_id` parameter override
- **Evolution methods**: All evolution-related methods now use `ScopeProvider` automatically
- **Clustering methods**: All clustering methods now use `ScopeProvider` automatically

#### Added

- **`DefaultScopeProvider`**: Returns empty filters for unrestricted OSS usage
- **Lazy import pattern**: Avoids circular dependencies when importing `DefaultScopeProvider`
- **Complete documentation**: `SCOPING_ARCHITECTURE.md` with line-by-line trace of scoping flow
- **OSS simplicity**: `SmartMemory()` works out-of-the-box without configuration
- **Service security**: Service layer always provides secure `ScopeProvider` with tenant isolation

#### Benefits

- ✅ **Zero configuration** for OSS single-user applications
- ✅ **Automatic tenant isolation** in service layer
- ✅ **No hardcoded parameters** - clean method signatures
- ✅ **Single source of truth** - all scoping through `ScopeProvider`
- ✅ **Core library agnostic** - no knowledge of multi-tenancy concepts
- ✅ **Backward compatible** - service layer unchanged (always provided `ScopeProvider`)

#### Migration Guide

**Before (v0.1.16)**:
```python
# Had to pass user_id everywhere
memory.search("query", user_id="user123")
memory.run_clustering(tenant_id="tenant456", workspace_id="ws789")
```

**After (v0.2.6)**:
```python
# OSS: No parameters needed
memory = SmartMemory()
memory.search("query")
memory.run_clustering()

# Service: ScopeProvider injected automatically
memory = SmartMemory(scope_provider=my_scope_provider)
memory.search("query")  # Automatically filtered
```

### Added

#### Similarity Graph Traversal (SSG) Retrieval
- Novel graph-based semantic search algorithms for enhanced multi-hop reasoning
- Two high-performance algorithms implemented:
  - `query_traversal`: Best for general queries (100% test pass, 0.91 precision/recall)
  - `triangulation_fulldim`: Best for high-precision factual queries
- Features:
  - Hybrid neighbor discovery (graph relationships + vector similarity)
  - Early stopping to prevent over-retrieval
  - Similarity caching for performance
  - Multi-tenant workspace filtering
  - Configurable via `config.json`
- New module: `smartmemory/retrieval/ssg_traversal.py`
- Configuration section added to `config.json`
- Integration with `SmartGraphSearch` via `_search_with_ssg_traversal()`
- Comprehensive test suite (18 unit tests, all passing)

#### API Examples

```python
from smartmemory import SmartMemory
from smartmemory.retrieval.ssg_traversal import SimilarityGraphTraversal

# Initialize
sm = SmartMemory()
ssg = SimilarityGraphTraversal(sm)

# Query traversal (best for general queries)
results = ssg.query_traversal(
    query="How do neural networks work?",
    max_results=15,
    workspace_id="workspace_123"
)

# Triangulation (best for precision)
results = ssg.triangulation_fulldim(
    query="What is the capital of France?",
    max_results=10,
    workspace_id="workspace_123"
)

# Integrated search with automatic fallback
results = sm.search("neural networks", use_ssg=True)
```

#### Reference
Eric Lester. (2025). Novel Semantic Similarity Graph Traversal Algorithms for Semantic Retrieval Augmented Generation Systems. https://github.com/glacier-creative-git/semantic-similarity-graph-traversal-semantic-rag-research

---

## [0.1.5] - 2025-10-04

### Added

#### Version Tracking System
- Bi-temporal version tracking with valid time and transaction time
- Automatic version numbering for memory updates
- Version comparison and diff functionality
- Graph-backed persistence
- Version caching

#### Temporal Search
- Time-range search functionality
- Version-aware search results
- Temporal metadata in results
- Integration with version tracker

#### Temporal Relationship Queries
- Query relationships at specific points in time
- Relationship history tracking
- Temporal pattern detection
- Co-occurring relationship detection

#### Bi-Temporal Joins
- Temporal overlap joins
- Concurrent event detection
- Temporal correlation queries
- Multiple join strategies

#### Performance Optimizations
- Temporal indexes for time-based lookups
- Query optimization and planning
- Batch operations
- Result caching
- Performance statistics

### API Examples

```python
# Version tracking
version = memory.temporal.version_tracker.create_version(
    item_id="memory123",
    content="Updated content",
    changed_by="user_id",
    change_reason="Correction"
)

# Temporal search
results = memory.temporal.search_temporal(
    "Python programming",
    start_time="2024-09-01",
    end_time="2024-09-30"
)

# Relationship queries
rels = memory.temporal.relationships.get_relationships_at_time(
    "memory123",
    datetime(2024, 9, 15)
)

# Bi-temporal joins
overlaps = memory.temporal.relationships.temporal_join(
    ["memory1", "memory2", "memory3"],
    start_time=datetime(2024, 9, 1),
    end_time=datetime(2024, 9, 30),
    join_type='overlap'
)

# Performance optimization
index = TemporalIndex()
optimizer = TemporalQueryOptimizer(index)
plan = optimizer.optimize_time_range_query(start, end)
```

### Tests
- 115 tests covering temporal features
- 25 unit tests for version tracking and relationships
- 90 integration tests for temporal search and joins

---

## [0.1.4] - 2025-10-04

### Added

- Temporal query system with 8 methods for time-travel and audit trails
- TemporalQueries API (`memory.temporal`):
  - `get_history()` - Version history of memories
  - `at_time()` - Query memories at specific points in time
  - `get_changes()` - Track changes to memory items
  - `compare_versions()` - Compare memory versions
  - `rollback()` - Rollback with dry-run preview
  - `get_audit_trail()` - Audit logs for compliance
  - `find_memories_changed_since()` - Find recent changes
  - `get_timeline()` - Timeline visualization

- Time-travel context manager (`memory.time_travel()`):
  - Execute queries in temporal context
  - Nested time-travel support

- Documentation:
  - `docs/BITEMPORAL_GUIDE.md` - Complete guide
  - `docs/IMPLEMENTATION_GUIDE.md` - Developer reference
  - `.cascade/smartmemory_implementation_memory.md` - Implementation patterns

- Examples:
  - `examples/temporal_queries_basic.py` - Feature demonstrations
  - `examples/temporal_audit_trail.py` - Compliance scenario
  - `examples/temporal_debugging.py` - Debugging use case

- Test suite (84 tests):
  - 39 unit tests for TemporalQueries
  - 24 unit tests for time-travel context
  - 21 integration tests for compliance

---

## [0.1.3] - 2025-10-04

### Added

- Zettelkasten implementation with bidirectional linking, emergent structure detection, and discovery
- Wikilink parser with automatic bidirectional linking
  - `[[Note Title]]` - Wikilinks with automatic link creation
  - `[[Note|Alias]]` - Wikilink aliases
  - `((Concept))` - Concept mentions
  - `#hashtag` - Hashtag extraction
- Documentation (`docs/ZETTELKASTEN.md`)
- Example scripts:
  - `examples/zettelkasten_example.py` - System demonstration
  - `examples/wikilink_demo.py` - Wikilink showcase
- CLI commands for Zettelkasten operations:
  - `smartmemory zettel add` - Add notes with wikilinks
  - `smartmemory zettel overview` - System overview
  - `smartmemory zettel backlinks` - Show backlinks
  - `smartmemory zettel connections` - Show all connections
  - `smartmemory zettel suggest` - Get AI suggestions
  - `smartmemory zettel clusters` - Detect knowledge clusters
  - `smartmemory zettel parse` - Parse wikilinks from content
- Unit tests for wikilink parser (18 tests)

### Fixed
- Fixed `FalkorDB.get_neighbors()` to handle Node objects properly
- Fixed `ZettelBacklinkSystem` to handle MemoryItem returns
- Fixed `ZettelEmergentStructure._get_all_notes()` to accept label='Note'
- Fixed `ZettelEmergentStructure._get_note_links()` for cluster detection
- Fixed wikilink resolution timing (now creates links after note is added)

### Documentation
- Zettelkasten user guide with API reference
- Updated README with Zettelkasten sections

---

## [0.1.2] - 2025-10-04

### Changed
- ChromaDB is now optional - Moved to optional dependency `[chromadb]`
- FalkorDB is default vector backend - Handles both graph and vector storage
- Python requirement updated to >=3.10
- Version externalized to `__version__.py`
- Removed hardcoded paths from tests

### Fixed
- Fixed Python 3.10+ syntax compatibility (`str | None` → `Optional[str]`)
- Fixed vector backend registry to handle missing ChromaDB gracefully
- Fixed all evolver tests with proper typed configs
- Fixed all enricher tests with correct API expectations
- Fixed test isolation issues

### Added
- Zettelkasten Memory System
  - Bidirectional linking with automatic backlinks
  - Knowledge cluster detection
  - Discovery engine for connections
  - Knowledge path finding
  - Missing connection suggestions
  - Random walk exploration
  - System health analytics
- `smartmemory/memory/types/zettel_memory.py` - Zettelkasten memory type
- `smartmemory/memory/types/zettel_extensions.py` - Discovery and structure detection
- `smartmemory/graph/types/zettel.py` - Zettelkasten graph backend
- `smartmemory/stores/converters/zettel_converter.py` - Zettel data conversion
- `smartmemory/plugins/evolvers/episodic_to_zettel.py` - Episodic → Zettel evolver
- `smartmemory/plugins/evolvers/zettel_prune.py` - Zettel pruning evolver
- `docs/ZETTELKASTEN.md` - Zettelkasten documentation
- `examples/zettelkasten_example.py` - Zettelkasten demo
- `tests/integration/zettelkasten/` - Test suite
- `smartmemory/__version__.py` - Single source of truth for version
- `docs/BACKEND_PLUGIN_DESIGN.md` - Design for future vectorstore plugins
- `VERSION_AND_COMPATIBILITY.md` - Version and compatibility documentation
- Better error messages for missing backends

### Documentation
- Updated README with Zettelkasten memory type
- Zettelkasten guide with API reference
- Updated dependency documentation

---

## [0.1.1] - 2025-10-03

### Added
- Plugin system with 19 built-in plugins
- Security system with 4 security profiles
- Plugin security features: Sandboxing, permissions, resource limits
- External plugin support via entry points
- CLI tools (optional install with `[cli]`)
- `docs/PLUGIN_SECURITY.md` - Security documentation
- 4 plugin examples

### Changed
- Plugins converted to class-based architecture
- Plugin discovery and registration system
- Security profiles: trusted, standard, restricted, untrusted

### Fixed
- Plugin loading and initialization
- Security validation and enforcement

---

## [0.1.0] - 2025-10-01

### Added
- Initial release
- Multi-type memory system (working, semantic, episodic, procedural)
- FalkorDB graph backend
- ChromaDB vector backend
- Basic plugin system
- Redis caching
- LLM integration via LiteLLM

---

## Version Numbering

SmartMemory follows [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 0.1.5)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Current Status:** Pre-1.0 (Beta)
- API may change between minor versions
- Production-ready but evolving
