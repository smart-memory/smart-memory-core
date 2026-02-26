# SmartMemory - Multi-Layered AI Memory System

[![Docs](https://img.shields.io/badge/docs-smartmemory.ai-blue)](https://docs.smartmemory.ai/smartmemory/intro)
[![PyPI version](https://badge.fury.io/py/smartmemory.svg)](https://pypi.org/project/smartmemory/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Read the docs](https://docs.smartmemory.ai/smartmemory/intro)** | **[Maya sample app](https://docs.smartmemory.ai/maya)**

SmartMemory is a comprehensive AI memory system that provides persistent, multi-layered memory storage and retrieval for AI applications. It combines graph databases, vector stores, and intelligent processing pipelines to create a unified memory architecture.

## 🚀 Quick Install

```bash
pip install smartmemory             # Full platform (requires Docker — FalkorDB + Redis)
pip install smartmemory[lite]       # Zero-infra local mode (SQLite + usearch, no Docker)
pip install smartmemory[lite,watch] # + vault watcher for auto-ingesting markdown files
```

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

# Offline / no-API-key mode: opt into restricted pipeline explicitly
from smartmemory.pipeline.config import PipelineConfig
memory = create_lite_memory(pipeline_profile=PipelineConfig.lite())
```

Or via CLI:
```bash
smartmemory add "Alice leads Project Atlas"
smartmemory search "who leads Atlas"
smartmemory rebuild       # Reindex vector store from graph data
smartmemory watch /path/to/vault  # Auto-ingest new/changed .md files
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

### Storage Backend

- **FalkorDB**: Graph database for relationships and vector storage
- **Redis**: Caching layer for performance optimization

### Processing Pipeline

The memory ingestion flow processes data through several stages:

1. **Input Adaptation**: Convert input data to MemoryItem format
2. **Classification**: Determine appropriate memory type
3. **Extraction**: Extract entities and relationships
4. **Storage**: Persist to appropriate memory stores
5. **Linking**: Create connections between related memories
6. **Enrichment**: Enhance memories with additional context
7. **Evolution**: Transform memories based on configured rules

## Key Features

- **Multi-Type Memory System**: Working, Semantic, Episodic, and Procedural memory types
- **Graph-Based Storage**: FalkorDB backend for complex relationship modeling
- **Vector Similarity**: FalkorDB vector storage for semantic search
- **Extensible Pipeline**: Modular processing stages for ingestion and evolution
- **Plugin Architecture**: 30+ built-in plugins with external plugin support
- **Plugin Security**: Sandboxing, permissions, and resource limits for safe plugin execution
- **Flexible Scoping**: Optional `ScopeProvider` for multi-tenancy or unrestricted OSS usage
- **Zero Configuration**: Works out-of-the-box for single-user applications
- **Caching Layer**: Redis-based performance optimization
- **Configuration Management**: Flexible configuration with environment variable support

## 📦 Installation

### From PyPI (Recommended)

```bash
# Core package (zero infra — SQLite + usearch, no Docker required)
pip install smartmemory

# With FalkorDB + Redis server backends
pip install smartmemory[server]        # FalkorDB graph DB + Redis cache

# With optional features
pip install smartmemory[cli]           # CLI tools
pip install smartmemory[rebel]         # REBEL relation extractor
pip install smartmemory[relik]         # ReliK relation extractor
pip install smartmemory[wikipedia]     # Wikipedia enrichment
pip install smartmemory[all]           # All optional features
```

### From Source (Development)

```bash
git clone https://github.com/smart-memory/smart-memory.git
cd smart-memory
pip install -e ".[dev]"

# Install spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### Required Services

SmartMemory requires **FalkorDB** (graph + vector storage) and **Redis** (caching) to be running.

#### Option 1: Docker Compose (Recommended)

Use the provided `docker-compose.yml` in the repository root:

```bash
docker-compose up -d
```

This starts:
- **FalkorDB** on port `9010` - graph database with vector storage
- **Redis** on port `9012` - caching layer

#### Option 2: Manual Installation

**FalkorDB:**
```bash
# macOS
brew tap falkordb/tap
brew install falkordb
falkordb-server --port 9010

# Or run via Docker directly
docker run -d -p 9010:6379 falkordb/falkordb:latest
```

**Redis:**
```bash
# macOS
brew install redis
redis-server --port 9012

# Or run via Docker directly
docker run -d -p 9012:6379 redis:7-alpine
```

#### Verify Services

```bash
# Test FalkorDB connection
redis-cli -p 9010 PING

# Test Redis connection
redis-cli -p 9012 PING
```

Both should return `PONG`.

## 🎯 Quick Start

### Basic Usage (OSS - Single User)

```python
from smartmemory import SmartMemory, MemoryItem

# Initialize SmartMemory (no configuration needed for OSS usage)
memory = SmartMemory()

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
pip install smartmemory[cli]

# Add a memory
smartmemory add "Python is great for AI" --memory-type semantic

# Search memories
smartmemory search "Python programming" --top-k 5

# Rebuild the vector index from graph data
smartmemory rebuild

# Auto-ingest new/changed Markdown files from a vault directory
pip install smartmemory[watch]
smartmemory watch /path/to/vault
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

The `examples/` directory contains several demonstration scripts:

- `memory_system_usage_example.py`: Basic memory operations (add, search, delete)
- `zettelkasten_example.py`: Complete Zettelkasten system demonstration
- `conversational_assistant_example.py`: Conversational AI with memory
- `advanced_programming_tutor.py`: Educational application example
- `working_holistic_example.py`: Comprehensive multi-session demo
- `background_processing_demo.py`: Asynchronous processing example

## Configuration

SmartMemory uses environment variables for configuration:

### Environment Variables

Key environment variables:
- `FALKORDB_HOST`: FalkorDB server host (default: localhost)
- `FALKORDB_PORT`: FalkorDB server port (default: 9010)
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 9012)
- `OPENAI_API_KEY`: OpenAI API key for embeddings

```bash
# Example .env file
export FALKORDB_HOST=localhost
export FALKORDB_PORT=9010
export REDIS_HOST=localhost
export REDIS_PORT=9012
export OPENAI_API_KEY=your-api-key-here
```

## Memory Evolution

SmartMemory includes built-in evolvers that automatically transform memories:

### Available Evolvers

- **WorkingToEpisodicEvolver**: Converts working memory to episodic when buffer is full
- **WorkingToProceduralEvolver**: Extracts repeated patterns as procedures
- **EpisodicToSemanticEvolver**: Promotes stable facts to semantic memory
- **EpisodicToZettelEvolver**: Converts episodic events to Zettelkasten notes
- **EpisodicDecayEvolver**: Archives old episodic memories
- **SemanticDecayEvolver**: Prunes low-relevance semantic facts
- **ZettelPruneEvolver**: Merges duplicate or low-quality notes
- **DecisionConfidenceEvolver**: Decays confidence on stale decisions, auto-retracts below threshold

Evolvers run automatically as part of the memory lifecycle. See the [examples](examples/) directory for evolution demonstrations.

## Plugin System

SmartMemory features a **unified, extensible plugin architecture** that allows you to customize and extend functionality. All plugins follow a consistent class-based pattern.

### Built-in Plugins

SmartMemory includes **30+ built-in plugins** across 4 types:

- **9 Extractors**: Extract entities and relationships
  - `LLMExtractor`, `LLMSingleExtractor`, `ConversationAwareLLMExtractor`, `SpacyExtractor`, `RelikExtractor`, `GLiNER2Extractor`, `HybridExtractor`, `DecisionExtractor`, `ReasoningExtractor`
- **7 Enrichers**: Add context and metadata to memories
  - `BasicEnricher`, `SentimentEnricher`, `TemporalEnricher`, `TopicEnricher`, `SkillsToolsEnricher`, `WikipediaEnricher`, `LinkExpansionEnricher`
- **1 Grounder**: Connect to external knowledge
  - `WikipediaGrounder`
- **13 Evolvers**: Transform memories based on rules
  - `WorkingToEpisodicEvolver`, `WorkingToProceduralEvolver`, `EpisodicToSemanticEvolver`, `EpisodicToZettelEvolver`, `EpisodicDecayEvolver`, `SemanticDecayEvolver`, `ZettelPruneEvolver`, `DecisionConfidenceEvolver`, `OpinionSynthesisEvolver`, `ObservationSynthesisEvolver`, `OpinionReinforcementEvolver`, etc.

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

See `docs/PLUGIN_SECURITY.md` for complete security documentation.

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
```

**API Design:**
- `ingest()` - Full agentic pipeline: extract → store → link → enrich → evolve. Use for user-facing ingestion.
- `add()` - Simple storage: normalize → store → embed. Use for internal operations or when pipeline is not needed.

**Scoping:**
- OSS mode: No scoping needed, all data accessible
- For multi-tenant applications, pass a `ScopeProvider` to enable automatic filtering
- See `docs/SECURITY_AND_AUTH.md` for complete details

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
- See `docs/SECURITY_AND_AUTH.md` for details

## Dependencies

### Core Dependencies

SmartMemory requires the following key dependencies:

- `falkordb`: Graph database and vector storage backend
- `spacy`: Natural language processing and entity extraction
- `dspy`: LLM programming framework for extraction and classification
- `litellm`: LLM integration layer
- `openai`: OpenAI API client (for embeddings)
- `redis`: Caching layer
- `scikit-learn`: Machine learning utilities
- `pydantic`: Data validation
- `python-dateutil`: Date/time handling
- `vaderSentiment`: Sentiment analysis
- `jinja2`: Template rendering

**Note:** SmartMemory uses FalkorDB for both graph and vector storage. While the codebase contains legacy ChromaDB integration code, FalkorDB is the primary and recommended backend.

### Optional Dependencies

Install additional features as needed:

```bash
# Specific extractors
pip install smartmemory[rebel]     # REBEL relation extraction
pip install smartmemory[relik]     # ReliK relation extraction

# Integrations
pip install smartmemory[slack]     # Slack integration
pip install smartmemory[aws]       # AWS integration
pip install smartmemory[wikipedia] # Wikipedia enrichment

# Tools
pip install smartmemory[cli]       # Command-line interface (add, search, rebuild)
pip install smartmemory[lite]      # Zero-infra local mode (SQLite + usearch, no Docker)
pip install smartmemory[watch]     # Vault watcher for auto-ingesting Markdown files

# Everything
pip install smartmemory[all]       # All optional features
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

External plugins use the `standard` security profile by default. See `docs/PLUGIN_SECURITY.md` for details.

## 🔗 Links

- **📦 PyPI Package**: https://pypi.org/project/smartmemory/
- **📚 Documentation**: https://docs.smartmemory.ai
- **🐙 GitHub Repository**: https://github.com/smart-memory/smart-memory
- **🐛 Issue Tracker**: https://github.com/smart-memory/smart-memory/issues
- **🔒 Security & Auth**: See `docs/SECURITY_AND_AUTH.md`
- **🔐 Plugin Security**: See `docs/PLUGIN_SECURITY.md`

---

**Get started with SmartMemory today!**

```bash
pip install smartmemory
```

Explore the [examples](examples/) directory for complete demonstrations and use cases.

---

## ✅ Recently Completed

### Zero-Infra Lite Mode (v0.3.9+)
- ✅ **`smartmemory[lite]`**: SQLite + usearch backend — no Docker, no FalkorDB, no Redis required
- ✅ **`create_lite_memory()`**: Factory function from `smartmemory.tools.factory` for zero-config setup
- ✅ **`lite_context()`**: Context manager that cleans up globals and closes SQLite on exit
- ✅ **`PipelineConfig.lite()`**: Named preset disabling coreference, LLM extraction, enrichers, and Wikidata grounding — opt-in, not bundled with Lite storage
- ✅ **`smartmemory[watch]`**: Vault watcher for auto-ingesting new/changed Markdown files
- ✅ **Constructor injection**: `vector_backend`, `cache`, `observability`, `pipeline_profile`, `entity_ruler_patterns` params added to `SmartMemory.__init__` — no monkey-patching required

### Unified Pipeline v2 (v0.3.5)
- ✅ **11-stage composable pipeline**: classify → coreference → simplify → entity_ruler → llm_extract → ontology_constrain → store → link → enrich → ground → evolve
- ✅ **Breakpoint execution**: `run_to()`, `run_from()`, `undo_to()` for debugging and resumption
- ✅ **Per-stage retry**: Configurable retry policies with exponential backoff
- ✅ **Async mode**: `ingest(sync=False)` with Redis Streams transport
- ✅ **Pipeline metrics**: Fire-and-forget metrics emission via Redis Streams

### Self-Learning Ontology (v0.3.4)
- ✅ **OntologyGraph**: Dedicated FalkorDB graph for entity types with three-tier status (seed → provisional → confirmed)
- ✅ **Promotion pipeline**: Six-gate evaluation (name length, blocklist, confidence, frequency, consistency, LLM validation)
- ✅ **Pattern manager**: Hot-reloadable learned entity patterns with Redis pub/sub
- ✅ **Layered ontology**: Base + overlay subscription system with hide/unhide, pin/unpin
- ✅ **Template catalog**: 3 built-in templates (General, Software Engineering, Business & Finance)

### Reasoning & Validation (v0.3.2)
- ✅ **Graph validation**: `MemoryValidator`, `EdgeValidator` for schema enforcement
- ✅ **Health metrics**: `GraphHealthChecker` with orphan ratio, provenance coverage
- ✅ **Inference engine**: Pattern-matching rules for automatic edge creation
- ✅ **Symbolic reasoning**: Residuation, query routing, proof trees, fuzzy confidence

### Decision Memory (v0.3.0)
- ✅ **First-class decisions**: Confidence tracking, provenance chains, lifecycle management
- ✅ **DecisionConfidenceEvolver**: Evidence-based reinforcement/contradiction with decay
- ✅ **Conflict detection**: Semantic search + content overlap heuristic
- ✅ **Causal chains**: Recursive traversal of DERIVED_FROM, CAUSED_BY, INFLUENCES edges


### Synthesis Memory (v0.2.6+)
- ✅ **Opinion memory**: Beliefs with confidence scores, reinforced/contradicted over time
- ✅ **Observation memory**: Synthesized entity summaries from scattered facts
- ✅ **Reasoning memory**: Chain-of-thought traces capturing "why" decisions were made

**See `CHANGELOG.md` for complete version history.**

Check the [GitHub repository](https://github.com/smart-memory/smart-memory) for the latest updates.
