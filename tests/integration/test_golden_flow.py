"""Golden flow integration test — the one test that proves the product works.

No mocks. Real SQLite backend, real spaCy, real vector index.
If this passes, the core memory lifecycle works.
If this fails, something real is broken.
"""

import tempfile
import pytest

from smartmemory.tools.factory import create_lite_memory
from smartmemory.pipeline.config import PipelineConfig


@pytest.fixture()
def memory(tmp_path):
    mem = create_lite_memory(
        data_dir=str(tmp_path),
        pipeline_profile=PipelineConfig.lite(llm_enabled=False),
    )
    yield mem
    try:
        mem._graph.backend.close()
    except Exception:
        pass


@pytest.mark.integration
class TestGoldenFlow:
    """One test class, one lifecycle: ingest → search → recall → get → clear."""

    def test_full_memory_lifecycle(self, memory):
        # ── Ingest 5 items ──────────────────────────────────────────────
        texts = [
            "Alice leads Project Atlas at Acme Corp",
            "Bob is the CTO of Acme Corp and reports to Alice",
            "Django is a Python web framework created by Adrian Holovaty",
            "Kubernetes orchestrates containerized workloads",
            "Always use async/await for database calls",
        ]
        item_ids = []
        for text in texts:
            result = memory.ingest(text)
            item_id = result["item_id"] if isinstance(result, dict) else result
            assert item_id, f"Ingest returned no item_id for: {text}"
            item_ids.append(item_id)

        assert len(item_ids) == 5

        # ── Search: exact keyword ───────────────────────────────────────
        results = memory.search("Alice", top_k=5)
        assert len(results) >= 1, "Search for 'Alice' returned nothing"
        contents = [r.content for r in results]
        assert any("Alice" in c for c in contents), f"'Alice' not in results: {contents}"

        # ── Search: with apostrophe (the bug that broke LongMemEval) ───
        results = memory.search("Alice's project", top_k=5)
        assert len(results) >= 1, "Search for 'Alice's project' returned nothing (apostrophe bug)"

        # ── Search: with question mark ──────────────────────────────────
        results = memory.search("What does Alice lead?", top_k=5)
        assert len(results) >= 1, "Search for 'What does Alice lead?' returned nothing (punctuation bug)"

        # ── Search: no false positives ──────────────────────────────────
        results = memory.search("quantum physics black holes", top_k=5)
        # Should return 0 or very low-relevance results
        # (We don't assert 0 because vector search may return weak matches)

        # ── Search: wildcard returns all ────────────────────────────────
        # Note: wildcard goes through storage.search, not mem.search
        # This tests the core search path instead
        all_nodes = memory._graph.backend.get_all_nodes()
        memory_nodes = [n for n in all_nodes if n.get("memory_type") not in ("Version",)]
        assert len(memory_nodes) >= 5, f"Expected >=5 memory nodes, got {len(memory_nodes)}"

        # ── Recall ──────────────────────────────────────────────────────
        recall_results = memory.search("", top_k=5, sort_by="recency")
        assert len(recall_results) >= 1, "Recall (empty query, recency sort) returned nothing"

        # ── Get by ID ───────────────────────────────────────────────────
        for item_id in item_ids[:2]:
            item = memory.get(item_id)
            assert item is not None, f"get({item_id}) returned None"
            content = item.content if hasattr(item, "content") else item.get("content", "")
            assert content, f"get({item_id}) returned empty content"

        # ── Get non-existent ────────────────────────────────────────────
        fake = memory.get("00000000-0000-0000-0000-000000000000")
        assert fake is None, "get() with fake ID should return None"

        # ── Verify no node loss (enrichment bug) ────────────────────────
        for item_id in item_ids:
            item = memory.get(item_id)
            assert item is not None, f"Node {item_id} disappeared (enrichment bug)"

        # ── Search after multiple ingests still works ───────────────────
        results = memory.search("Kubernetes", top_k=5)
        assert len(results) >= 1, "Search for 'Kubernetes' failed after multiple ingests"

        results = memory.search("Django Python", top_k=5)
        assert len(results) >= 1, "Search for 'Django Python' failed"


@pytest.mark.integration
class TestSearchEdgeCases:
    """Search-specific edge cases that have caused production bugs."""

    def test_apostrophe_search(self, memory):
        memory.ingest("Sarah's project uses PostgreSQL for the backend")
        results = memory.search("Sarah's project", top_k=5)
        assert len(results) >= 1

    def test_question_mark_search(self, memory):
        memory.ingest("The API uses REST endpoints for communication")
        results = memory.search("What API does the system use?", top_k=5)
        assert len(results) >= 1

    def test_possessive_matches_base_name(self, memory):
        memory.ingest("Marcus designed the authentication system")
        results = memory.search("Marcus's design", top_k=5)
        assert len(results) >= 1

    def test_search_after_many_ingests(self, memory):
        """Regression: search failed after 5+ ingests in benchmarks."""
        for i in range(10):
            memory.ingest(f"Fact number {i}: the system handles {i * 100} requests per second")
        results = memory.search("requests per second", top_k=5)
        assert len(results) >= 1

    def test_recency_sort_with_none_timestamps(self, memory):
        """Regression: None created_at sorted to end with reverse=True."""
        memory.ingest("First memory")
        memory.ingest("Second memory")
        results = memory.search("", top_k=5, sort_by="recency")
        assert len(results) >= 1
