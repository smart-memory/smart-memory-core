"""Lite golden flow — validates ingest -> get -> search with zero external services.

Requires only local dependencies (spaCy, SQLite, usearch). No Docker, no API keys.

The integration/ conftest auto-marks this as pytest.mark.integration.
"""
import pytest


class TestLiteGoldenFlow:
    """End-to-end lite mode: ingest, get, search in a single scenario."""

    @pytest.fixture
    def lite_memory(self, tmp_path):
        """Create a fully local SmartMemory instance backed by SQLite + usearch."""
        from smartmemory.pipeline.config import PipelineConfig
        from smartmemory.tools.factory import lite_context

        profile = PipelineConfig.lite(llm_enabled=False)
        with lite_context(str(tmp_path), pipeline_profile=profile) as memory:
            yield memory

    def test_ingest_get_search(self, lite_memory):
        """Golden flow: ingest content, retrieve by ID, search by query."""
        # 1. Ingest
        result = lite_memory.ingest("Alice leads Project Atlas at Acme Corp in Austin, Texas")
        # ingest() returns a str item_id on the sync path; handle dict defensively
        if isinstance(result, dict):
            item_id = result.get("item_id") or result.get("memory_node_id")
        else:
            item_id = result
        assert item_id is not None, "ingest() must return a non-None item_id"

        # 2. Get by ID
        item = lite_memory.get(item_id)
        assert item is not None, f"get({item_id!r}) returned None — item was not stored"
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        assert content is not None, "Retrieved item must have a content field"
        assert len(content) > 0, "Retrieved item content must not be empty"

        # 3. Search — the ingested item must appear in results
        results = lite_memory.search("Alice Project Atlas", top_k=5)
        result_list = results if isinstance(results, list) else results.get("results", [])
        assert len(result_list) >= 1, "search() returned no results after ingest"

        found_ids = []
        for r in result_list:
            r_id = r.get("item_id") if isinstance(r, dict) else getattr(r, "item_id", None)
            found_ids.append(r_id)

        assert item_id in found_ids, (
            f"Ingested item {item_id!r} not found in search results. "
            f"Found IDs: {found_ids}"
        )
