#!/usr/bin/env python3
"""Migrate Claude Code MEMORY.md content into SmartMemory lite daemon.

One-time migration script. POSTs each memory to the lite daemon's
/memory/ingest endpoint (default localhost:9014).

Usage:
    smartmemory start                              # Ensure daemon is running
    python scripts/migrate_claude_memory.py        # Run migration
    python scripts/migrate_claude_memory.py --dry-run  # Preview only
"""

import os
import sys

import httpx

DAEMON_URL = os.environ.get("SMARTMEMORY_DAEMON_URL", "http://localhost:9014")

MEMORIES = [
    {
        "content": (
            "SmartMemory Extraction Architecture Decision: Two-Tier Async + Self-Learning EntityRuler\n\n"
            "DESIGN TARGET — roadmap item CORE-EXT-1. Two-tier exists, self-learning loop not yet.\n\n"
            "Fast tier (real-time, 4ms, $0): spaCy sm + EntityRuler → 96.9% E-F1, 65.1% R-F1\n"
            "Enrichment tier (async, 740ms): Groq Llama-3.3-70b → 97.7% E-F1, 85-88% R-F1\n"
            "Self-learning: LLM-discovered entities feed back into EntityRuler → ruler grows over time\n"
            "Zero-cost variant: Gemma-3-27b-it — after prompt optimization: 100% E-F1, 89.3% R-F1 at $0\n\n"
            "Key benchmark results (53 configs tested):\n"
            "- spaCy sm + EntityRuler: 96.9% E-F1 at 4ms — within 0.8% of Groq on entities\n"
            "- Gemma-3-27b-it: 95.3% E-F1, 86.7% R-F1 at $0 — best local model\n"
            "- EntityRuler adds +10.9 E-F1 points and +16.4 R-F1 points at zero latency cost\n"
            "- Progressive prompting FAILS — anchoring bias\n"
            "- Bigger ≠ better for local: Gemma-27B > QWQ-56B > Hermes-70B\n\n"
            "EntityRuler Self-Learning Loop:\n"
            "1. Text → spaCy sm + EntityRuler → instant results (4ms)\n"
            "2. Background async worker: LLM extracts full entities\n"
            "3. Diff: LLM entities - ruler entities = net-new patterns\n"
            "4. Quality gate (confidence > 0.8, length > 3, frequency > 1) → add to ruler\n"
            "5. Persist patterns to disk, load on startup\n"
            "6. Convergence: after ~10k memories, ruler covers 99%+ of domain vocabulary\n\n"
            "Benchmark doc: docs/plans/2026-02-05-extraction-benchmark-findings.md"
        ),
        "memory_type": "semantic",
        "topic": "extraction-architecture",
    },
    {
        "content": (
            "SmartMemory Extraction Strategy (validated 2026-02-05):\n"
            "- EntityRuler is the key innovation — pattern-matching NER before statistical model\n"
            "- spaCy sm + EntityRuler nearly matches LLM quality on entities at 200x speed\n"
            "- Progressive prompting always worse than LLM from scratch\n"
            "- Local specialized models (REBEL, GLiNER, NuExtract, NuNER) all inferior to spaCy+ruler\n"
            "- Relation extraction ceiling for local: ~65% R-F1 (dep-parse SVO limitation)\n"
            "- Only LLMs can do good RE (85%+ R-F1)\n\n"
            "Prompt Optimization (Gemma-3-27b-it): 95.3% → 100% E-F1, 86.7% → 89.3% R-F1\n"
            "Changes applied to SINGLE_CALL_PROMPT in llm_single.py\n\n"
            "Default extraction model: Groq Llama-3.3-70b-versatile\n"
            "- 100% E-F1, 89.3% R-F1 on original 16, 638ms-2017ms per doc, zero errors\n"
            "- GPT-5.4-mini has JSON parse failures. Sonnet via Agent SDK is 10x slower."
        ),
        "memory_type": "semantic",
        "topic": "extraction-strategy",
    },
    {
        "content": (
            "SmartMemory Architecture Patterns:\n"
            "- Memory types: string-based (not enum), add to MEMORY_TYPES set\n"
            "- Extended types (reasoning, opinion, observation, decision) NOT in MemoryNodeType enum\n"
            "- Edge schemas: EdgeSchema dataclass + register_edge_schema() in schema_validator.py\n"
            "- Multi-tenant: SecureSmartMemory wrapper, never instantiate SmartMemory directly\n"
            "- Core library is synchronous — only service layer is async\n"
            "- Extractors: ExtractorPlugin.extract(text) -> dict with PluginMetadata\n"
            "- Evolvers: EvolverPlugin.evolve(memory, log=None)\n"
            "- Service routes: auto-discovered in memory_service/api/routes/\n"
            "- Maya commands: CommandDefinition with handler, aliases, CommandCategory.MEMORY\n\n"
            "Existing capabilities (don't reinvent):\n"
            "- Hybrid search: SSG + BM25/embedding RRF fusion\n"
            "- Query decomposition: cross-query RRF for compound queries (CORE-SEARCH-1)\n"
            "- Reasoning traces: ReasoningTrace model with steps, evaluation, quality gating\n"
            "- Contradiction detection: AssertionChallenger with cascades\n"
            "- Temporal system: TemporalQueries + VersionTracker bi-temporal\n"
            "- Confidence: OpinionMetadata.reinforce()/contradict() with diminishing returns\n"
            "- Evolvers: episodic decay, semantic decay, opinion reinforcement/synthesis"
        ),
        "memory_type": "semantic",
        "topic": "architecture-patterns",
    },
    {
        "content": (
            "Blast Radius Analysis for SmartMemory refactoring:\n\n"
            "When mapping deletion/replacement scope, trace at four layers:\n"
            "1. Route URLs — grep for /auth/login etc.\n"
            "2. SDK config objects — endpoints: { login: '...' }, mode: 'custom' vs 'sso'\n"
            "3. SDK method implementations — AuthCore.login(), AuthAPI.signup() + test fixtures\n"
            "4. UI component methods — login(), signup() callbacks in AuthContext/AuthProvider\n\n"
            "A route-string grep catches layer 1 but misses layers 2-4."
        ),
        "memory_type": "procedural",
        "topic": "blast-radius-analysis",
    },
    {
        "content": (
            "Query Decomposition (CORE-SEARCH-1, implemented 2026-03-05):\n"
            "- decompose_query=True on SmartMemory.search() splits compound queries\n"
            "- Splits on and/or/,/; into 1-4 sub-queries\n"
            "- Each sub-query runs independently, results merged via cross-query RRF (rrf_k=60)\n"
            "- Working memory exemption: memory_type='working' ignores decomposition\n"
            "- API surface: decompose: bool = False on service MCP, REST, standalone MCP\n"
            "- Key files: smartmemory/search/query_decomposer.py, rrf_merge.py, smart_memory.py\n"
            "- Contract: docs/features/CORE-SEARCH-1/search-contract.json"
        ),
        "memory_type": "semantic",
        "topic": "query-decomposition",
    },
    {
        "content": (
            "Reasoning Detection (CORE-SYS2-1c, implemented 2026-02-26):\n"
            "- Opt-in ReasoningDetectStage in v2 pipeline after LLMExtractStage\n"
            "- extract_reasoning=True on ingest()\n"
            "- Sync-only: extract_decisions (1b) and extract_reasoning (1c) only work in sync path\n"
            "- Flag stamps must happen AFTER _apply_pipeline_profile() to survive lite override\n"
            "- PipelineConfig.with_reasoning() NOT usable as pipeline_profile=\n"
            "- Report: docs/features/CORE-SYS2-1c/report.md\n\n"
            "Decision Memory (implemented 2026-02-04):\n"
            "- Phases 1-7 complete. Decision model, manager, API (11 endpoints)\n"
            "- Maya /decide /beliefs /why commands\n"
            "- Spec: docs/plans/2026-02-04-decision-memory-causal-tracking-design.md"
        ),
        "memory_type": "semantic",
        "topic": "reasoning-decisions",
    },
    {
        "content": (
            "Ontology-Grounded Extraction — all 15 design questions (Q0-Q14) decided:\n"
            "- Separate FalkorDB graphs for ontology vs data\n"
            "- Three-tier type status: seed → provisional → confirmed\n"
            "- PromotionConfig, Redis Streams metrics, ExtractionConfig composite stage config\n"
            "- No RelationRuler, 8-phase implementation plan\n"
            "- Strategic plan: docs/plans/2026-02-05-ontology-strategic-implementation-plan.md (v5)\n"
            "- Design spec: docs/plans/2026-02-05-ontology-grounded-extraction-design.md\n"
            "- Use cases: docs/plans/2026-02-05-use-cases-reference.md (25 cases)\n\n"
            "Pending designs (ready for implementation):\n"
            "- Graph Validation & Health: docs/plans/2026-02-04-graph-validation-health-inference-design.md\n"
            "- Symbolic Reasoning: docs/plans/2026-02-04-symbolic-reasoning-layer-design.md\n"
            "- Unified Roadmap: docs/plans/2026-02-04-reasoning-system-roadmap.md"
        ),
        "memory_type": "semantic",
        "topic": "ontology-design",
    },
    {
        "content": (
            "file: Symlink Integration Patterns (validated VIS-GRAPH-4 + VIS-GRAPH-5):\n\n"
            "Mandatory steps when wiring a file: dependency into a consuming app:\n"
            "1. Tailwind content scanning: Add '../<package>/src/**/*.{js,jsx}' to config\n"
            "2. CSS @import ordering: @import MUST precede @tailwind — PostCSS spec\n"
            "3. Cytoscape direct dep: Rollup can't resolve transitive deps through file: symlinks\n"
            "4. Vite dedupe: Add dedupe: ['cytoscape', 'react', 'react-dom'] to vite.config.js\n"
            "5. SDK dist rebuild: After changes, run npm run build — file: deps resolve main: dist/\n"
            "6. Vite cache clear: Delete node_modules/.vite after rebuilding file: deps"
        ),
        "memory_type": "procedural",
        "topic": "symlink-patterns",
    },
    {
        "content": (
            "SmartMemory Contracts Policy (updated 2026-03-05):\n"
            "- New contracts go in smart-memory-docs/docs/features/<FEATURE-ID>/<name>-contract.json (git-tracked)\n"
            "- Legacy contracts remain at contracts/ (monorepo root, NOT git-tracked)\n"
            "- Check feature folder first, then legacy contracts/"
        ),
        "memory_type": "semantic",
        "topic": "contracts-policy",
    },
]


def migrate(dry_run: bool = False) -> None:
    for i, mem in enumerate(MEMORIES):
        topic = mem["topic"]
        body = {
            "content": mem["content"],
            "memory_type": mem["memory_type"],
        }

        if dry_run:
            print(f"[{i+1}/{len(MEMORIES)}] {topic} ({mem['memory_type']}, {len(mem['content'])} chars)")
            continue

        try:
            r = httpx.post(
                f"{DAEMON_URL}/memory/ingest",
                json=body,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            item_id = data.get("item_id", data.get("id", "?"))
            print(f"[{i+1}/{len(MEMORIES)}] {topic} → {item_id}")
        except httpx.ConnectError:
            print(f"[{i+1}/{len(MEMORIES)}] {topic} FAILED: daemon unreachable at {DAEMON_URL}")
            print("  Run 'smartmemory start' first.")
            sys.exit(1)
        except Exception as e:
            print(f"[{i+1}/{len(MEMORIES)}] {topic} FAILED: {e}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("DRY RUN — no API calls\n")
    else:
        # Health check
        try:
            r = httpx.get(f"{DAEMON_URL}/health", timeout=3)
            r.raise_for_status()
            print(f"Daemon healthy at {DAEMON_URL}")
        except Exception:
            print(f"Daemon unreachable at {DAEMON_URL}. Run 'smartmemory start' first.")
            sys.exit(1)
        print(f"Migrating {len(MEMORIES)} memories\n")
    migrate(dry_run=dry)
    print(f"\nDone. {'Would migrate' if dry else 'Migrated'} {len(MEMORIES)} memories.")
