# SmartMemory Examples

This directory contains example scripts demonstrating SmartMemory's features and capabilities.

## Requirements

### Core Dependencies
```bash
pip install smartmemory-core
```

### Optional Dependencies

| Feature | Package | Install |
|---------|---------|---------|
| HuggingFace Embeddings | `transformers`, `sentence-transformers` | `pip install transformers torch sentence-transformers` |
| Sentiment Analysis | `textblob` | `pip install textblob` |
| SPARQL/DBpedia | `SPARQLWrapper` | `pip install SPARQLWrapper` |
| Background Processing | `celery`, `redis` | `pip install celery redis` |

### Environment Setup

Most examples require:
- **FalkorDB** running locally (for graph storage)
- **OpenAI API key** or alternative LLM provider configured

```bash
# Set your API key
export OPENAI_API_KEY=your_key_here

# Or use HuggingFace
export EMBEDDING_PROVIDER=huggingface
export HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Running Examples

```bash
cd /path/to/smart-memory
PYTHONPATH=. python examples/<example_name>.py
```

---

## Core Examples

### Memory System Basics

| Example | Description |
|---------|-------------|
| [memory_system_usage_example.py](memory_system_usage_example.py) | Basic usage of `SmartMemory` - `ingest()`, `search()`, `delete()` |
| [factory_usage_example.py](factory_usage_example.py) | Using `MemoryFactory` and `StoreFactory` to create memory instances |
| [working_holistic_example.py](working_holistic_example.py) | Comprehensive demo across semantic, episodic, procedural, and working memory |

### Specialized Memory Types

| Example | Description |
|---------|-------------|
| [zettelkasten_example.py](zettelkasten_example.py) | Zettelkasten note system with bidirectional linking and discovery |
| [conversational_assistant_example.py](conversational_assistant_example.py) | Memory-enhanced conversational AI with context continuity |
| [decision_memory_example.py](decision_memory_example.py) | **NEW** - Decision Memory with confidence tracking and lifecycle |
| [reasoning_trace_example.py](reasoning_trace_example.py) | **NEW** - System 2 Memory for chain-of-thought reasoning traces |

---

## Pipeline & Processing

| Example | Description |
|---------|-------------|
| [pipeline_v2_example.py](pipeline_v2_example.py) | **NEW** - Pipeline v2 with breakpoints, resumption, and rollback |
| [background_processing_demo.py](background_processing_demo.py) | Async/background processing with Celery |
| [conversation_aware_extraction_demo.py](conversation_aware_extraction_demo.py) | Context-aware entity extraction from conversations |

---

## Ontology & Schema

| Example | Description |
|---------|-------------|
| [self_learning_ontology_example.py](self_learning_ontology_example.py) | **NEW** - Self-learning ontology with promotion and governance |
| [ontology_demo.py](ontology_demo.py) | Basic ontology management |
| [ontology_inference_e2e.py](ontology_inference_e2e.py) | End-to-end ontology inference |
| [demo_hitl_governance.py](demo_hitl_governance.py) | Human-in-the-loop governance for ontology changes |

---

## Temporal Features

| Example | Description |
|---------|-------------|
| [temporal_queries_basic.py](temporal_queries_basic.py) | Basic temporal queries (time-based filtering) |
| [temporal_audit_trail.py](temporal_audit_trail.py) | Audit trail and history tracking |
| [temporal_debugging.py](temporal_debugging.py) | Debugging with temporal queries |

---

## Custom Plugins

| Example | Description |
|---------|-------------|
| [custom_extractor_example.py](custom_extractor_example.py) | Creating custom extractor plugins |
| [custom_enricher_example.py](custom_enricher_example.py) | Creating custom enricher plugins |
| [custom_evolver_example.py](custom_evolver_example.py) | Creating custom evolver plugins |
| [custom_grounder_example.py](custom_grounder_example.py) | Creating custom grounder plugins (DBpedia, APIs) |

---

## Embeddings & Search

| Example | Description |
|---------|-------------|
| [huggingface_embeddings_example.py](huggingface_embeddings_example.py) | Using HuggingFace models for embeddings |
| [wikilink_demo.py](wikilink_demo.py) | Wikilink-style linking and navigation |

---

## Advanced Examples

| Example | Description |
|---------|-------------|
| [advanced_programming_tutor.py](advanced_programming_tutor.py) | AI programming tutor with memory |
| [environment_check.py](environment_check.py) | Verify environment setup |

---

## Quick Start

### 1. Basic Memory Usage
```python
from smartmemory.smart_memory import SmartMemory

memory = SmartMemory()

# Ingest with full pipeline (extraction, linking, enrichment)
memory.ingest("John Smith is a software engineer at Google.")

# Search
results = memory.search("software engineers")
for r in results:
    print(r.content)

# Simple storage (no pipeline)
memory.add("Quick note without extraction")
```

### 2. Decision Memory
```python
from smartmemory.decisions import DecisionManager

manager = DecisionManager(memory)

# Create a decision
decision = manager.create(
    content="Use PostgreSQL for the database",
    decision_type="choice",
    confidence=0.85,
)

# Reinforce with evidence
manager.reinforce(decision.decision_id, "benchmark_results_123")
```

### 3. Reasoning Traces
```python
from smartmemory.models.reasoning import ReasoningStep, ReasoningTrace

trace = ReasoningTrace(
    trace_id="trace_001",
    steps=[
        ReasoningStep(type="thought", content="Analyzing the problem..."),
        ReasoningStep(type="decision", content="The best approach is X"),
    ],
)
```

### 4. Pipeline with Breakpoints
```python
runner = memory.create_pipeline_runner()
config = PipelineConfig.preview()

# Run to extraction, pause
state = runner.run_to(text, config, stop_after="llm_extract")
print(state.entities)

# Resume from checkpoint
state = runner.run_from(state, config)
```

---

## Notes

- Examples marked **NEW** were added to demonstrate recent API features
- Some examples require external services (FalkorDB, LLM provider)
- Use `environment_check.py` to verify your setup
