# SmartMemory - Multi-Layered AI Memory System

[![Docs](https://img.shields.io/badge/docs-smartmemory.ai-blue)](https://docs.smartmemory.ai/smartmemory/intro)
[![PyPI version](https://badge.fury.io/py/smartmemory-core.svg)](https://pypi.org/project/smartmemory-core/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Read the docs](https://docs.smartmemory.ai/smartmemory/intro)** | **[Maya sample app](https://docs.smartmemory.ai/maya)**

SmartMemory is a comprehensive AI memory system that provides persistent, multi-layered memory storage and retrieval for AI applications. It combines graph databases, vector stores, and intelligent processing pipelines to create a unified memory architecture.

## 🚀 Quick Install

```bash
pip install smartmemory[local]           # Local memory + MCP server + viewer + CLI (recommended)
pip install smartmemory                  # Remote client only (connects to a SmartMemory service)
pip install smartmemory-core[lite]       # Core library only, local mode (for developers)
pip install smartmemory-core[server]     # Core library only, server mode (FalkorDB + Redis)
```

> **`smartmemory`** is the distribution package — MCP server, graph viewer, and CLI.
> **`smartmemory-core`** is the core library for developers building on top of SmartMemory.
> `smartmemory[local]` bundles `smartmemory-core[lite]` for local SQLite storage. Without `[local]`, it's a remote client only.

### SmartMemory Lite — No Docker Required

```python
from smartmemory.tools.factory import create_lite_memory, lite_context

# Simple usage — full LLM extraction runs if OPENAI_API_KEY is set
memory = create_lite_memory()
item_id = memory.ingest("Alice leads Project Atlas")
results = memory.search("who leads Atlas", top_k=5)

# Preferred in scripts — cleans up globals and closes SQLite on exit
with lite_context() as memory:
    item_id = memory.ingest("Alice leads Project Atlas")
    results = memory.search("who leads Atlas")

# Force no LLM calls (even if OPENAI_API_KEY is set)
from smartmemory.pipeline.config import PipelineConfig
memory = create_lite_memory(pipeline_profile=PipelineConfig.lite(llm_enabled=False))
```

Or via CLI:
```bash
smartmemory-core add "Alice leads Project Atlas"
smartmemory-core search "who leads Atlas"
smartmemory-core rebuild       # Reindex vector store from graph data
smartmemory-core watch /path/to/vault  # Auto-ingest new/changed .md files
```

## Architecture Overview

SmartMemory implements a multi-layered memory architecture with the following components:

### Core Components

- **SmartMemory**: Main unified memory interface (`smartmemory.smart_memory.SmartMemory`)
- **SmartGraph**: Graph database backend using FalkorDB for relationship storage
- **Memory Types**: Specialized memory stores for different data types
- **Pipeline Stages**: Processing stages for ingestion, enrichment, and evolution
- **Plugin System**: Extensible architecture for custom evolvers and enrichers

### Memory Types

- **Working Memory**: Short-term context buffer (in-memory, capacity=10)
- **Semantic Memory**: Facts and concepts with vector embeddings
- **Episodic Memory**: Personal experiences and learning history
- **Procedural Memory**: Skills, strategies, and learned patterns
- **Zettelkasten Memory**: Bidirectional note-taking system with AI-powered knowledge discovery
- **Reasoning Memory**: Chain-of-thought traces capturing "why" decisions were made (System 2)
- **Opinion Memory**: Beliefs with confidence scores, reinforced or contradicted over time
- **Observation Memory**: Synthesized entity summaries from scattered facts
- **Decision Memory**: First-class decisions with confidence tracking, provenance chains, and lifecycle management

### Storage Backends

- **Lite mode**: SQLite graph + usearch vectors — no Docker, no external services
- **Server mode**: FalkorDB (graph + vectors) + Redis (caching) — full-featured, requires Docker

### Processing Pipeline

`ingest()` runs an 11-stage pipeline:

```
classify → coreference → simplify → entity_ruler → llm_extract → ontology_constrain → store → link → enrich → ground → evolve
```

Each stage implements the `StageCommand` protocol (`execute(state, config) → state`, `undo(state) → state`). The pipeline supports breakpoint execution (`run_to()`, `run_from()`, `undo_to()`) for debugging and resumption.

`add()` is simple storage: normalize → store → embed (use for internal/derived items).

## Key Features

- **9 Memory Types**: Working, Semantic, Episodic, Procedural, Zettelkasten, Reasoning, Opinion, Observation, Decision
- **11-Stage NLP Pipeline**: classify → coreference → simplify → entity_ruler → llm_extract → ontology_constrain → store → link → enrich → ground → evolve
- **Self-Learning EntityRuler**: Pattern-matching NER that improves with use — LLM discoveries feed back into rules (96.9% entity F1 at 4ms)
- **Evolver Framework**: Core auto-registered evolvers plus specialist lifecycle evolvers for decay, consolidation, opinion synthesis, retrieval-based strengthening, Hebbian co-retrieval, and stale memory detection
- **Code Indexer**: AST-based Python + TypeScript parser with cross-file call resolution, semantic code search, and memory↔code graph bridging
- **Zero-Infra Lite Mode**: SQLite + usearch backend — `pip install smartmemory-core[lite]` and go
- **Server Mode**: FalkorDB graph + Redis caching for production-scale deployments
- **Hybrid Search**: Graph-structured search + BM25/embedding RRF fusion with query decomposition for compound queries
- **20 Auto-Registered Plugins**: 4 extractors, 5 enrichers, 10 evolvers, and 1 grounder loaded by default, with additional specialist plugins available for opt-in use
- **Plugin Security**: Sandboxing, permissions, and resource limits for safe plugin execution
- **Flexible Scoping**: Optional `ScopeProvider` for multi-tenancy or unrestricted OSS usage

## 📦 Installation

### From PyPI (Recommended)

```bash
# Lite mode (zero infra — SQLite + usearch, no Docker required)
pip install smartmemory-core[lite]

# Server mode (FalkorDB + Redis, requires Docker or manual install)
pip install smartmemory-core[server]

# Optional features
pip install smartmemory-core[cli]           # CLI tools
pip install smartmemory-core[watch]         # Vault watcher for Markdown files
pip install smartmemory-core[wikipedia]     # Wikipedia enrichment
pip install smartmemory-core[all]           # All optional features
```

### From Source (Development)

```bash
git clone https://github.com/smart-memory/smart-memory-core.git
cd smart-memory-core
pip install -e ".[dev,lite,cli,watch]"

# Install spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### Infrastructure

**Lite mode** (`smartmemory-core[lite]`): No external services needed. SQLite and usearch are bundled.

**Server mode** (`smartmemory-core[server]`): Requires FalkorDB and Redis:

```bash
# Docker Compose (recommended) — from repository root
docker-compose up -d
# Starts FalkorDB on port 9010, Redis on port 9012

# Or manually
docker run -d -p 9010:6379 falkordb/falkordb:latest
docker run -d -p 9012:6379 redis:7-alpine

# Verify
redis-cli -p 9010 PING   # FalkorDB
redis-cli -p 9012 PING   # Redis
```

## 🎯 Quick Start

### Basic Usage (Lite Mode)

```python
from smartmemory.tools.factory import create_lite_memory

# No Docker, no config — just works
memory = create_lite_memory()

# Ingest a memory (full pipeline: extract → store → link → enrich → evolve)
item_id = memory.ingest("User prefers Python for data analysis tasks")

# Or use add() for simple storage without pipeline
item = MemoryItem(
    content="Quick note about Python",
    memory_type="semantic",
    metadata={'topic': 'preferences'}
)
memory.add(item)

# Search memories (automatically scoped)
results = memory.search("Python programming", top_k=5)
for result in results:
    print(f"Content: {result.content}")
    print(f"Type: {result.memory_type}")

# Get memory summary
summary = memory.get_all_items_debug()
print(f"Total memories: {summary['total_items']}")
```

### Using Different Memory Types

```python
from smartmemory import SmartMemory, MemoryItem

# Initialize SmartMemory
memory = SmartMemory()

# Add working memory (short-term context)
working_item = MemoryItem(
    content="Current conversation context",
    memory_type="working"
)
memory.add(working_item)

# Add semantic memory (facts and concepts)
semantic_item = MemoryItem(
    content="Python is a high-level programming language",
    memory_type="semantic"
)
memory.add(semantic_item)

# Add episodic memory (experiences)
episodic_item = MemoryItem(
    content="User completed Python tutorial on 2024-01-15",
    memory_type="episodic"
)
memory.add(episodic_item)

# Add procedural memory (skills and procedures)
procedural_item = MemoryItem(
    content="To sort a list in Python: use list.sort() or sorted(list)",
    memory_type="procedural"
)
memory.add(procedural_item)

# Add Zettelkasten note (interconnected knowledge)
zettel_item = MemoryItem(
    content="# Machine Learning\n\nML learns from data using algorithms.",
    memory_type="zettel",
    metadata={'title': 'ML Basics', 'tags': ['ai', 'ml']}
)
memory.add(zettel_item)
```

### CLI Usage (Optional)

```bash
# Install CLI tools
pip install smartmemory-core[cli]

# Add a memory
smartmemory-core add "Python is great for AI"

# Add without creating a markdown file
smartmemory-core add "Quick note" --no-markdown

# Search memories
smartmemory-core search "Python programming" --top-k 5

# Search with JSON output
smartmemory-core search "Python programming" --json

# Rebuild the vector index from graph data
smartmemory-core rebuild

# Auto-ingest new/changed Markdown files from a vault directory
pip install smartmemory-core[watch]
smartmemory-core watch /path/to/vault
```

## Use Cases

### Conversational AI Systems
- Maintain context across multiple conversation sessions
- Learn user preferences and adapt responses
- Build comprehensive user profiles over time

### Educational Applications
- Track learning progress and adapt teaching strategies
- Remember previous topics and build upon them
- Personalize content based on individual learning patterns

### Knowledge Management
- Store and retrieve complex information relationships
- Connect related concepts across different domains
- Evolve understanding through continuous learning
- Build a personal knowledge base with Zettelkasten method

### Personal AI Assistants
- Remember user preferences and past interactions
- Provide contextually relevant recommendations
- Learn from user feedback to improve responses

## Examples

The `examples/` directory contains a broader set of demonstration scripts than the highlights below. See [`examples/README.md`](examples/README.md) for the full catalog and setup notes.

- `memory_system_usage_example.py`: Basic memory operations (ingest, search, delete)
- `factory_usage_example.py`: Factory helpers for creating memory and store instances
- `pipeline_v2_example.py`: Breakpoints, resumption, rollback, and stage timing inspection
- `reasoning_trace_example.py`: System 2 reasoning traces and storage patterns
- `self_learning_ontology_example.py`: Ontology promotion and governance workflow
- `working_holistic_example.py`: Multi-memory demo across semantic, episodic, procedural, and working memory

## Configuration

SmartMemory uses environment variables for configuration:

### Environment Variables

Key environment variables:
- `OPENAI_API_KEY`: OpenAI API key for embeddings and LLM extraction (auto-detected in lite mode)
- `GROQ_API_KEY`: Groq API key — alternative to OpenAI for LLM extraction (auto-detected in lite mode)

**Server mode only:**
- `FALKORDB_HOST`: FalkorDB server host (default: localhost)
- `FALKORDB_PORT`: FalkorDB server port (default: 9010)
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 9012)

```bash
# Lite mode — only API key needed (optional, enables LLM extraction)
export OPENAI_API_KEY=your-api-key-here

# Server mode — also needs database hosts
export FALKORDB_HOST=localhost
export FALKORDB_PORT=9010
export REDIS_HOST=localhost
export REDIS_PORT=9012
```

## Memory Evolution

SmartMemory includes built-in evolvers that automatically transform memories. In lite mode, evolution runs incrementally in the background — memories evolve as they're added, not just at the end of each pipeline run.

### Available Evolvers

**Core evolvers** — memory type transitions and lifecycle:
- **WorkingToEpisodicEvolver**: Converts working memory to episodic when buffer is full
- **WorkingToProceduralEvolver**: Extracts repeated patterns as procedures
- **EpisodicToSemanticEvolver**: Promotes stable facts to semantic memory
- **EpisodicToZettelEvolver**: Converts episodic events to Zettelkasten notes
- **EpisodicDecayEvolver**: Archives old episodic memories
- **SemanticDecayEvolver**: Prunes low-relevance semantic facts
- **ZettelPruneEvolver**: Merges duplicate or low-quality notes
- **DecisionConfidenceEvolver**: Decays confidence on stale decisions, auto-retracts below threshold
- **OpinionSynthesisEvolver**: Synthesizes opinions from accumulated observations
- **ObservationSynthesisEvolver**: Creates entity summaries from scattered facts
- **OpinionReinforcementEvolver**: Adjusts opinion confidence based on new evidence
- **StaleMemoryEvolver**: Flags memories as stale when referenced source code changes

**Enhanced evolvers** — neuroscience-inspired dynamics:
- **ExponentialDecayEvolver**: Time-based activation decay with configurable half-life
- **RetrievalBasedStrengtheningEvolver**: Memories accessed more frequently become harder to forget
- **HebbianCoRetrievalEvolver**: Reinforces edges between memories retrieved together ("neurons that fire together wire together")
- **InterferenceBasedConsolidationEvolver**: Similar competing memories interfere, strengthening the dominant one
- **EnhancedWorkingToEpisodicEvolver**: Context-aware working→episodic transition with richer metadata

Evolvers run automatically as part of the memory lifecycle. See the [examples](examples/) directory for evolution demonstrations.

## Plugin System

SmartMemory features a **unified, extensible plugin architecture** that allows you to customize and extend functionality. All plugins follow a consistent class-based pattern.

### Built-in Plugins

SmartMemory includes **20 auto-registered plugins** with additional specialist plugins available for opt-in use:

**Auto-registered by default** (loaded by `PluginManager`):
- **4 Extractors**: `LLMExtractor`, `LLMSingleExtractor`, `ConversationAwareLLMExtractor`, `SpacyExtractor`
- **5 Enrichers**: `BasicEnricher`, `SentimentEnricher`, `TemporalEnricher`, `ExtractSkillsToolsEnricher`, `TopicEnricher`
- **10 Evolvers**: `WorkingToEpisodicEvolver`, `WorkingToProceduralEvolver`, `EpisodicToSemanticEvolver`, `EpisodicToZettelEvolver`, `EpisodicDecayEvolver`, `SemanticDecayEvolver`, `ZettelPruneEvolver`, `ExponentialDecayEvolver`, `InterferenceBasedConsolidationEvolver`, `RetrievalBasedStrengtheningEvolver`
- **1 Grounder**: `WikipediaGrounder`

**Specialist plugins** (used by specific pipeline stages or opt-in features):
- **Extractors**: `GroqExtractor`, `DecisionExtractor`, `ReasoningExtractor`
- **Enrichers**: `LinkExpansionEnricher`
- **Evolvers**: `DecisionConfidenceEvolver`, `OpinionSynthesisEvolver`, `ObservationSynthesisEvolver`, `OpinionReinforcementEvolver`, `StaleMemoryEvolver`, `HebbianCoRetrievalEvolver`, `EnhancedWorkingToEpisodicEvolver`
- **Grounders**: `PublicKnowledgeGrounder` (Wikidata QIDs)

### Creating Custom Plugins

Create your own plugins by extending the base classes:

```python
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata

class MyCustomEnricher(EnricherPlugin):
    @classmethod
    def metadata(cls):
        return PluginMetadata(
            name="my_enricher",
            version="1.0.0",
            author="Your Name",
            description="My custom enricher",
            plugin_type="enricher",
            dependencies=["some-lib>=1.0.0"],
            security_profile="standard",
            requires_network=False,
            requires_llm=False
        )
    
    def enrich(self, item, node_ids=None):
        # Your enrichment logic
        item.metadata["custom_field"] = "value"
        return item.metadata
```

See the [examples](examples/) directory for complete plugin examples.

### Publishing Plugins

Publish your plugin as a Python package with entry points:

```toml
# pyproject.toml
[project.entry-points."smartmemory.plugins.enrichers"]
my_enricher = "my_package:MyCustomEnricher"
```

Install and use:
```bash
pip install my-smartmemory-plugin
# Automatically discovered and loaded!
```

### Plugin Types

- **ExtractorPlugin**: Extract entities and relationships from text
- **EnricherPlugin**: Add metadata and context to memories
- **GrounderPlugin**: Link memories to external knowledge sources
- **EvolverPlugin**: Transform memories based on conditions

All plugins are automatically discovered and registered at startup.

### Plugin Security

SmartMemory includes a comprehensive security system for plugins:

- **4 Security Profiles**: `trusted`, `standard` (default), `restricted`, `untrusted`
- **Permission System**: Control memory, network, file, and LLM access
- **Resource Limits**: Automatic timeout (30s), memory limits, network request limits
- **Sandboxing**: Isolated execution with security enforcement
- **Static Validation**: Detects security issues before execution

```python
# Plugins are secure by default
PluginMetadata(
    security_profile="standard",  # Balanced security
    requires_network=True,        # Explicitly declare requirements
    requires_llm=False
)
```

See the [full documentation](https://docs.smartmemory.ai/smartmemory/intro) for complete security documentation.

### Examples

See the `examples/` directory for complete plugin examples:
- `custom_enricher_example.py` - Sentiment analysis and keyword extraction
- `custom_evolver_example.py` - Memory promotion and archival
- `custom_extractor_example.py` - Regex and domain-specific NER
- `custom_grounder_example.py` - DBpedia and custom API grounding

## Testing

Run the test suite:

```bash
# Run all tests
PYTHONPATH=. pytest -v tests/

# Run specific test categories
PYTHONPATH=. pytest tests/unit/
PYTHONPATH=. pytest tests/integration/
PYTHONPATH=. pytest tests/e2e/

# Run examples
PYTHONPATH=. python examples/memory_system_usage_example.py
PYTHONPATH=. python examples/conversational_assistant_example.py
```

## API Reference

### SmartMemory Class

Main interface for memory operations:

```python
class SmartMemory:
    def __init__(
        self,
        scope_provider: Optional[ScopeProvider] = None,
        vector_backend=None,          # Any VectorStoreBackend; None uses default (FalkorDB)
        cache=None,                   # Any cache-compatible object; e.g. NoOpCache()
        observability: bool = True,   # False disables Redis Streams emission and metrics
        pipeline_profile=None,        # PipelineConfig instance; PipelineConfig.lite() for zero-infra
        entity_ruler_patterns=None,   # Any object with get_patterns() -> dict[str, str]
    )

    # Primary API
    def ingest(self, item, sync=True, **kwargs) -> str  # Full pipeline
    def add(self, item, **kwargs) -> str                # Simple storage
    def get(self, item_id: str) -> Optional[MemoryItem]
    def search(self, query: str, top_k: int = 5, memory_type: str = None) -> List[MemoryItem]
    def delete(self, item_id: str) -> bool

    # Graph Integrity (v0.3.8+)
    def delete_run(self, run_id: str) -> int            # Delete entities by pipeline run
    def rename_entity_type(self, old: str, new: str) -> int  # Ontology evolution
    def merge_entity_types(self, sources: List[str], target: str) -> int

    # Advanced
    def run_clustering(self) -> dict
    def run_evolution_cycle(self) -> None
    def personalize(self, traits: dict = None, preferences: dict = None) -> None
    def get_all_items_debug(self) -> Dict[str, Any]

    # Lifecycle
    def close(self) -> None                              # Clean shutdown (optional)
```

**API Design:**
- `ingest()` - Full agentic pipeline: extract → store → link → enrich → evolve. Use for user-facing ingestion.
- `add()` - Simple storage: normalize → store → embed. Use for internal operations or when pipeline is not needed.

**Scoping:**
- OSS mode: No scoping needed, all data accessible
- For multi-tenant applications, pass a `ScopeProvider` to enable automatic filtering
- See the [documentation](https://docs.smartmemory.ai/smartmemory/intro) for complete details

### MemoryItem Class

Core data structure for memory storage:

```python
@dataclass
class MemoryItem:
    content: str
    memory_type: str = 'semantic'
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    valid_start_time: Optional[datetime] = None
    valid_end_time: Optional[datetime] = None
    transaction_time: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    entities: Optional[list] = None
    relations: Optional[list] = None
    metadata: dict = field(default_factory=dict)
```

**Security Metadata:**
- For OSS usage, security metadata fields are not needed
- For multi-tenant applications, use a `ScopeProvider` for automatic metadata injection
- See the [documentation](https://docs.smartmemory.ai/smartmemory/intro) for details

## Dependencies

### Core Dependencies

SmartMemory requires the following key dependencies:

- `spacy`: Natural language processing and entity extraction
- `dspy`: LLM programming framework for extraction and classification
- `litellm`: LLM integration layer
- `openai`: OpenAI API client (for embeddings)
- `scikit-learn`: Machine learning utilities
- `pydantic`: Data validation
- `python-dateutil`: Date/time handling
- `vaderSentiment`: Sentiment analysis
- `jinja2`: Template rendering

**Lite mode** adds: `usearch` (vector search), uses Python's built-in `sqlite3`.

**Server mode** adds: `falkordb` (graph + vector storage), `redis` (caching).

### Optional Dependencies

Install additional features as needed:

```bash
# Modes
pip install smartmemory-core[lite]      # Zero-infra local mode (SQLite + usearch, no Docker)
pip install smartmemory-core[server]    # Server mode (FalkorDB + Redis)

# Tools
pip install smartmemory-core[cli]       # Command-line interface (add, search, rebuild)
pip install smartmemory-core[watch]     # Vault watcher for auto-ingesting Markdown files

# Integrations
pip install smartmemory-core[slack]     # Slack integration
pip install smartmemory-core[aws]       # AWS integration
pip install smartmemory-core[wikipedia] # Wikipedia enrichment

# Everything
pip install smartmemory-core[all]       # All optional features
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

## 📄 License

SmartMemory is dual-licensed to provide flexibility for both open-source and commercial use. See [LICENSE](LICENSE) for details.

## Security

SmartMemory takes plugin security seriously. All plugins run in a sandboxed environment with:

- ✅ **Permission checks** - Plugins must declare what they access
- ✅ **Resource limits** - Automatic timeouts and memory limits
- ✅ **Execution isolation** - Sandboxed plugin execution
- ✅ **Static analysis** - Security validation before execution

External plugins use the `standard` security profile by default.

## 🔗 Links

- **📦 PyPI Package**: https://pypi.org/project/smartmemory-core/
- **📚 Documentation**: https://docs.smartmemory.ai
- **🐙 GitHub Repository**: https://github.com/smart-memory/smart-memory-core
- **🐛 Issue Tracker**: https://github.com/smart-memory/smart-memory-core/issues
- **🔒 Security & Auth**: See [documentation](https://docs.smartmemory.ai/smartmemory/intro)
- **🔐 Plugin Security**: See [documentation](https://docs.smartmemory.ai/smartmemory/intro)

---

**Get started with SmartMemory today!**

```bash
pip install smartmemory[local]
```

Explore the [examples](examples/) directory for complete demonstrations and use cases.

---

## ✅ Recently Completed

### Incremental Evolution (v0.5.5)
- ✅ **Event-driven evolution in lite mode**: Memories evolve incrementally in the background as they're added, not just at pipeline end
- ✅ **Backend API unification**: `HebbianCoRetrievalEvolver` rewritten from Cypher to backend API — works on both SQLite and FalkorDB
- ✅ **`SmartMemory.close()`**: Optional clean shutdown for long-running scripts

### Code Indexer (v0.5.x)
- ✅ **AST-based Python parser**: Extracts modules, classes, functions, FastAPI routes, pytest tests
- ✅ **TypeScript/JavaScript parser**: tree-sitter-based with React component/hook detection
- ✅ **Cross-file call resolution**: Symbol table resolves import aliases to entity `item_id`s
- ✅ **Semantic code search**: Vector embeddings on code entities for intent-based queries
- ✅ **Git-anchored staleness**: `commit_hash` on entities + query-time drift detection
- ✅ **Memory↔code bridging**: `REFERENCES_CODE` edges link semantic memories to code entities
- ✅ **EntityRuler seeding**: Code class/function names auto-added to pattern dictionary

### Query Decomposition (v0.5.x)
- ✅ **Compound query splitting**: "auth flow and caching strategy" → 2 sub-queries
- ✅ **Cross-query RRF merge**: Independent search per sub-query, reciprocal rank fusion
- ✅ **`decompose_query=True`** on `SmartMemory.search()`, REST, and MCP surfaces

### Zero-Infra Lite Mode (v0.4.x)
- ✅ **`smartmemory-core[lite]`**: SQLite + usearch backend — no Docker, no FalkorDB, no Redis required
- ✅ **`create_lite_memory()`**: Factory function from `smartmemory.tools.factory` for zero-config setup
- ✅ **`PipelineConfig.lite(llm_enabled=None)`**: LLM extraction auto-detected from `OPENAI_API_KEY`/`GROQ_API_KEY`
- ✅ **Graceful degradation**: VersionTracker, TemporalQueries, Grounding all work on SQLite backend
- ✅ **Constructor injection**: `vector_backend`, `cache`, `observability`, `pipeline_profile`, `entity_ruler_patterns`

### Unified Pipeline v2 (v0.3.x)
- ✅ **11-stage composable pipeline**: classify → coreference → simplify → entity_ruler → llm_extract → ontology_constrain → store → link → enrich → ground → evolve
- ✅ **Self-learning ontology**: OntologyGraph with three-tier status (seed → provisional → confirmed), six-gate promotion, hot-reloadable patterns
- ✅ **Breakpoint execution**: `run_to()`, `run_from()`, `undo_to()` for debugging and resumption
- ✅ **Async mode**: `ingest(sync=False)` with Redis Streams transport

### Memory Types & Reasoning (v0.2.x–v0.3.x)
- ✅ **Decision memory**: Confidence tracking, provenance chains, causal chain traversal
- ✅ **Opinion/Observation synthesis**: Beliefs with confidence scores, entity summaries from scattered facts
- ✅ **Reasoning traces**: Chain-of-thought capture with auto-detection via classify stage
- ✅ **Graph validation**: Schema enforcement, health metrics, inference engine, symbolic reasoning

**See `CHANGELOG.md` for complete version history.**

Check the [GitHub repository](https://github.com/smart-memory/smart-memory-core) for the latest updates.
