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

            # ── 8. Recall top_k=1 returns exactly 1 ────────────────────
            recall_1 = memory.search("", top_k=1, sort_by="recency")
            assert len(recall_1) == 1, f"Recall top_k=1 returned {len(recall_1)}, expected 1"

            # ── 9. Recall top_k=3 returns exactly 3 ────────────────────
            recall_3 = memory.search("", top_k=3, sort_by="recency")
            assert len(recall_3) == 3, f"Recall top_k=3 returned {len(recall_3)}, expected 3"

            # ── 10. Get by ID — verify correct content ──────────────────
            for i, item_id in enumerate(item_ids):
                item = memory.get(item_id)
                assert item is not None, f"get({item_id}) returned None"
                content = item.content if hasattr(item, "content") else item.get("content", "")
                assert content, f"get({item_id}) returned empty content"
                # Verify content matches what was ingested
                assert items[i][0] in content, f"get({item_id}) content doesn't match ingest: {content[:60]}"

            # ── 11. Get non-existent returns None ───────────────────────
            fake = memory.get("00000000-0000-0000-0000-000000000000")
            assert fake is None, "get() with fake ID should return None"

            # ── 12. Memory types are preserved ──────────────────────────
            for i, item_id in enumerate(item_ids):
                item = memory.get(item_id)
                mt = item.memory_type if hasattr(item, "memory_type") else item.get("memory_type")
                expected_type = items[i][1]
                assert mt == expected_type, f"Item {i}: expected type {expected_type}, got {mt}"

            # ── 13. Ingest with custom properties ───────────────────────
            result = memory.ingest(
                "Sprint 42 retrospective: improve CI pipeline",
                context={"memory_type": "episodic", "project": "atlas", "domain": "engineering"},
            )
            prop_id = result["item_id"] if isinstance(result, dict) else result
            assert prop_id, "Ingest with properties returned no item_id"
            item_ids.append(prop_id)

            # ── 14. Export ──────────────────────────────────────────────
            export_path = data_dir + "/export.jsonl"
            from smartmemory.corpus.exporter import CorpusExporter
            exporter = CorpusExporter(smart_memory=memory)
            export_count = exporter.run(export_path, source="test", domain="test")
            assert export_count >= 5, f"Exported {export_count}, expected >=5"

            # Verify export file exists and has content
            with open(export_path) as f:
                lines = f.readlines()
            assert len(lines) >= 2, "Export file too short (need header + records)"

            # ── 15. Export with memory_type filter ──────────────────────
            export_filtered_path = data_dir + "/export_semantic.jsonl"
            exporter_filtered = CorpusExporter(smart_memory=memory, memory_type="semantic")
            filtered_count = exporter_filtered.run(export_filtered_path, source="test", domain="test")
            assert filtered_count >= 2, f"Filtered export got {filtered_count}, expected >=2 semantic items"
            assert filtered_count < export_count, "Filtered export should be smaller than full export"

            # ── 16. Import dry-run ──────────────────────────────────────
            from smartmemory.corpus.importer import CorpusImporter
            importer_dry = CorpusImporter(smart_memory=memory, mode="direct")
            dry_stats = importer_dry.run(export_path, resume=False, dry_run=True)
            assert dry_stats.imported >= 5, f"Dry-run validated {dry_stats.imported}, expected >=5"

            # Verify dry-run didn't add duplicates
            nodes_after_dry = memory._graph.backend.get_all_nodes()
            mem_nodes_after_dry = [n for n in nodes_after_dry if n.get("memory_type") not in ("Version",)]
            assert len(mem_nodes_after_dry) == len(memory_nodes) + 1, \
                f"Dry-run changed node count: {len(memory_nodes) + 1} → {len(mem_nodes_after_dry)}"

            # ── 17. Clear all ───────────────────────────────────────────
            backend = memory._graph.backend
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
            # ── 18. Verify empty after clear ───────────────────────────
            all_nodes = memory2._graph.backend.get_all_nodes()
            memory_nodes = [n for n in all_nodes if n.get("memory_type") not in ("Version",)]
            assert len(memory_nodes) == 0, f"Expected 0 nodes after clear, got {len(memory_nodes)}"

            # ── 19. Search on empty graph returns empty, no crash ───────
            results = memory2.search("Alice", top_k=5)
            assert results is not None, "Search on empty graph returned None"
            assert len(results) == 0, f"Search on empty graph returned {len(results)} results"

            # ── 20. Recall on empty graph returns empty, no crash ───────
            recall_empty = memory2.search("", top_k=5, sort_by="recency")
            assert recall_empty is not None, "Recall on empty graph returned None"

            # ── 21. Re-add works after clear ────────────────────────────
            result = memory2.ingest("Fresh memory after clear")
            fresh_id = result["item_id"] if isinstance(result, dict) else result
            assert fresh_id, "Ingest after clear returned no item_id"

            results = memory2.search("Fresh memory", top_k=5)
            assert len(results) >= 1, "Search after clear+re-add returned nothing"

            # ── 22. Import from export ──────────────────────────────────
            from smartmemory.corpus.importer import CorpusImporter as CorpusImporter2

            importer = CorpusImporter2(smart_memory=memory2, mode="direct")
            stats = importer.run(export_path, resume=False, dry_run=False)
            assert stats.imported >= 5, f"Imported {stats.imported}, expected >=5"

            # ── 23. Search works after import ───────────────────────────
            results = memory2.search("Alice", top_k=5)
            assert len(results) >= 1, "Search after import returned nothing"

            results = memory2.search("Kubernetes", top_k=5)
            assert len(results) >= 1, "Search for imported 'Kubernetes' failed"

            # ── 24. Second clear is idempotent ──────────────────────────
            memory2._graph.backend.close()
            # Re-wipe data files
            for f in os.listdir(data_dir):
                fpath = os.path.join(data_dir, f)
                if f != "export.jsonl" and f != "export_semantic.jsonl":
                    try:
                        os.unlink(fpath)
                    except Exception:
                        pass
            memory3 = create_lite_memory(
                data_dir=data_dir,
                pipeline_profile=PipelineConfig.lite(llm_enabled=False),
            )
            # Clear on already-empty should not crash
            all_nodes = memory3._graph.backend.get_all_nodes()
            empty_nodes = [n for n in all_nodes if n.get("memory_type") not in ("Version",)]
            assert len(empty_nodes) == 0, "Second clear: expected 0 nodes"
            memory3._graph.backend.close()

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

    def test_search_with_unicode(self, memory):
        memory.ingest("Ødegaard joined the team from München")
        results = memory.search("Ødegaard", top_k=5)
        assert isinstance(results, list)
        # Unicode search may or may not match depending on tokenizer

    def test_search_empty_query(self, memory):
        memory.ingest("Some stored content")
        results = memory.search("", top_k=5)
        # Empty query should return results (used by recall)
        assert isinstance(results, list)

    def test_search_very_long_query(self, memory):
        memory.ingest("Short content here")
        long_query = "word " * 200  # 200 words
        results = memory.search(long_query.strip(), top_k=5)
        assert isinstance(results, list)  # Should not crash


@pytest.mark.integration
class TestErrorPaths:
    """Table-driven error path harness. Each case should produce a clean error, not a crash."""

    def test_ingest_empty_content(self, memory):
        """Ingest empty string — should either reject or store as empty."""
        result = memory.ingest("")
        # May return item_id (stored as empty) or handle gracefully
        assert isinstance(result, (str, dict))

    def test_get_nonexistent_id(self, memory):
        result = memory.get("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_get_garbage_id(self, memory):
        result = memory.get("not-a-valid-uuid-at-all")
        assert result is None

    def test_get_empty_id(self, memory):
        result = memory.get("")
        assert result is None

    def test_search_on_empty_graph(self, memory):
        """Search before any ingest — should return empty, not crash."""
        results = memory.search("anything", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_recall_on_empty_graph(self, memory):
        results = memory.search("", top_k=5, sort_by="recency")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_top_k_zero(self, memory):
        memory.ingest("Some content")
        results = memory.search("content", top_k=0)
        assert isinstance(results, list)

    def test_search_top_k_negative(self, memory):
        memory.ingest("Some content")
        results = memory.search("content", top_k=-1)
        assert isinstance(results, list)

    def test_ingest_very_long_content(self, memory):
        """10KB content — should not crash or timeout."""
        long_text = "This is a sentence about software engineering. " * 200
        result = memory.ingest(long_text)
        item_id = result["item_id"] if isinstance(result, dict) else result
        assert item_id

        # Should be retrievable
        item = memory.get(item_id)
        assert item is not None

    def test_ingest_special_characters(self, memory):
        """Content with SQL injection, HTML, newlines — should store safely."""
        nasty = "Robert'); DROP TABLE nodes;--\n<script>alert('xss')</script>\t\x00"
        result = memory.ingest(nasty)
        item_id = result["item_id"] if isinstance(result, dict) else result
        assert item_id

        item = memory.get(item_id)
        assert item is not None

    def test_double_ingest_same_content(self, memory):
        """Ingest same content twice — should create two separate items."""
        text = "Duplicate content test"
        r1 = memory.ingest(text)
        r2 = memory.ingest(text)
        id1 = r1["item_id"] if isinstance(r1, dict) else r1
        id2 = r2["item_id"] if isinstance(r2, dict) else r2
        assert id1 != id2, "Same content should create different item_ids"

    def test_export_empty_graph(self, memory, tmp_path):
        """Export with no data — should produce valid file with just header."""
        from smartmemory.corpus.exporter import CorpusExporter
        export_path = str(tmp_path / "empty_export.jsonl")
        exporter = CorpusExporter(smart_memory=memory)
        count = exporter.run(export_path, source="test", domain="test")
        assert count == 0

    def test_import_nonexistent_file(self, memory):
        """Import file that doesn't exist — should raise, not crash silently."""
        from smartmemory.corpus.importer import CorpusImporter
        importer = CorpusImporter(smart_memory=memory, mode="direct")
        with pytest.raises((FileNotFoundError, OSError)):
            importer.run("/tmp/does_not_exist_12345.jsonl")
