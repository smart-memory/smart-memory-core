"""Golden flow integration test — the one test that proves the product works.

No mocks. Real SQLite backend, real spaCy, real vector index.
If this passes, the core memory lifecycle works.
If this fails, something real is broken.
"""

import json
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


@pytest.fixture()
def data_dir(tmp_path):
    """Return a fresh temp dir path for tests that need to manage their own memory instance."""
    return str(tmp_path / "sm-data")


@pytest.mark.integration
class TestGoldenFlow:
    """One test, one lifecycle: ingest → search → recall → get → clear → re-add → types → export/import."""

    def test_full_memory_lifecycle(self, data_dir):
        memory = create_lite_memory(
            data_dir=data_dir,
            pipeline_profile=PipelineConfig.lite(llm_enabled=False),
        )

        try:
            # ── 1. Ingest 5 items with different types ──────────────────
            items = [
                ("Alice leads Project Atlas at Acme Corp", "episodic"),
                ("Bob is the CTO of Acme Corp and reports to Alice", "episodic"),
                ("Django is a Python web framework created by Adrian Holovaty", "semantic"),
                ("Kubernetes orchestrates containerized workloads", "semantic"),
                ("Always use async/await for database calls", "procedural"),
            ]
            item_ids = []
            for text, mem_type in items:
                result = memory.ingest(text, context={"memory_type": mem_type})
                item_id = result["item_id"] if isinstance(result, dict) else result
                assert item_id, f"Ingest returned no item_id for: {text}"
                item_ids.append(item_id)

            assert len(item_ids) == 5

            # ── 2. Search: exact keyword ────────────────────────────────
            results = memory.search("Alice", top_k=5)
            assert len(results) >= 1, "Search for 'Alice' returned nothing"
            contents = [r.content for r in results]
            assert any("Alice" in c for c in contents), f"'Alice' not in results: {contents}"

            # ── 3. Search: with apostrophe ──────────────────────────────
            results = memory.search("Alice's project", top_k=5)
            assert len(results) >= 1, "Apostrophe search failed"

            # ── 4. Search: with question mark ───────────────────────────
            results = memory.search("What does Alice lead?", top_k=5)
            assert len(results) >= 1, "Question mark search failed"

            # ── 5. Search: multi-word ───────────────────────────────────
            results = memory.search("Django Python", top_k=5)
            assert len(results) >= 1, "Multi-word search failed"

            # ── 6. Verify all nodes exist (no enrichment loss) ──────────
            all_nodes = memory._graph.backend.get_all_nodes()
            memory_nodes = [n for n in all_nodes if n.get("memory_type") not in ("Version",)]
            assert len(memory_nodes) >= 5, f"Expected >=5 memory nodes, got {len(memory_nodes)}"

            # ── 7. Recall (empty query, recency sort) ───────────────────
            recall_results = memory.search("", top_k=5, sort_by="recency")
            assert len(recall_results) >= 1, "Recall returned nothing"

            # ── 8. Get by ID ────────────────────────────────────────────
            for item_id in item_ids:
                item = memory.get(item_id)
                assert item is not None, f"get({item_id}) returned None"
                content = item.content if hasattr(item, "content") else item.get("content", "")
                assert content, f"get({item_id}) returned empty content"

            # ── 9. Get non-existent returns None ────────────────────────
            fake = memory.get("00000000-0000-0000-0000-000000000000")
            assert fake is None, "get() with fake ID should return None"

            # ── 10. Memory types are preserved ──────────────────────────
            procedural_item = memory.get(item_ids[4])
            mt = procedural_item.memory_type if hasattr(procedural_item, "memory_type") else procedural_item.get("memory_type")
            assert mt == "procedural", f"Expected procedural, got {mt}"

            # ── 11. Export ──────────────────────────────────────────────
            export_path = data_dir + "/export.jsonl"
            from smartmemory.corpus.exporter import CorpusExporter
            exporter = CorpusExporter(smart_memory=memory)
            export_count = exporter.run(export_path, source="test", domain="test")
            assert export_count >= 5, f"Exported {export_count}, expected >=5"

            # Verify export file exists and has content
            with open(export_path) as f:
                lines = f.readlines()
            assert len(lines) >= 2, "Export file too short (need header + records)"

            # ── 12. Clear all ───────────────────────────────────────────
            backend = memory._graph.backend
            # Close and recreate to simulate clear
            backend.close()

        finally:
            try:
                memory._graph.backend.close()
            except Exception:
                pass

        # ── 13. Fresh instance on same data dir after clear ─────────
        import shutil
        import os
        # Delete data files to simulate clear
        for f in os.listdir(data_dir):
            fpath = os.path.join(data_dir, f)
            if f != "export.jsonl":
                try:
                    os.unlink(fpath)
                except Exception:
                    pass

        memory2 = create_lite_memory(
            data_dir=data_dir,
            pipeline_profile=PipelineConfig.lite(llm_enabled=False),
        )
        try:
            # ── 14. Verify empty after clear ────────────────────────────
            all_nodes = memory2._graph.backend.get_all_nodes()
            memory_nodes = [n for n in all_nodes if n.get("memory_type") not in ("Version",)]
            assert len(memory_nodes) == 0, f"Expected 0 nodes after clear, got {len(memory_nodes)}"

            # ── 15. Re-add works after clear ────────────────────────────
            result = memory2.ingest("Fresh memory after clear")
            fresh_id = result["item_id"] if isinstance(result, dict) else result
            assert fresh_id, "Ingest after clear returned no item_id"

            results = memory2.search("Fresh memory", top_k=5)
            assert len(results) >= 1, "Search after clear+re-add returned nothing"

            # ── 16. Import from export ──────────────────────────────────
            from smartmemory.corpus.reader import CorpusReader
            from smartmemory.corpus.importer import CorpusImporter

            importer = CorpusImporter(smart_memory=memory2, mode="direct")
            stats = importer.run(export_path, resume=False, dry_run=False)
            assert stats.imported >= 5, f"Imported {stats.imported}, expected >=5"

            # ── 17. Search works after import ───────────────────────────
            results = memory2.search("Alice", top_k=5)
            assert len(results) >= 1, "Search after import returned nothing"

            results = memory2.search("Kubernetes", top_k=5)
            assert len(results) >= 1, "Search for imported 'Kubernetes' failed"

        finally:
            try:
                memory2._graph.backend.close()
            except Exception:
                pass


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

    def test_case_insensitive_search(self, memory):
        memory.ingest("PostgreSQL supports JSONB columns")
        results = memory.search("postgresql", top_k=5)
        assert len(results) >= 1

    def test_search_with_special_characters(self, memory):
        memory.ingest("The config uses key=value pairs")
        results = memory.search("key=value", top_k=5)
        # Should not crash, may or may not find results
        assert isinstance(results, list)
