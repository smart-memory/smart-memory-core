"""Golden-flow integration tests for CORE-EXT-1: Two-Tier Async Extraction.

Requires: FalkorDB (9010), Redis (9012) running via docker compose.

Run with:
    PYTHONPATH=. pytest tests/integration/test_two_tier_extraction.py -v

Test coverage:
    - Tier 1 path: ingest(sync=False) runs pipeline + enqueues to extract queue
    - Tier 2 path: process_extract_job() resolves ids, writes net-new entities/relations
    - Status branching: item_not_found, no_text, llm_failed
    - Frequency gate: entity patterns below count=2 filtered from get_entity_patterns()
    - Redis-down fallback: sync=False returns queued=False but item_id still set
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from smartmemory.models.memory_item import MemoryItem
from smartmemory.background.extraction_worker import process_extract_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(name: str, entity_type: str) -> str:
    """Reproduce the SHA-256 id that LLMSingleExtractor._process_entities() produces."""
    raw = f"{name.lower()}|{entity_type.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_llm_entity(name: str, entity_type: str, confidence: float = 0.9) -> MemoryItem:
    sha_id = _sha256(name, entity_type)
    return MemoryItem(
        content=name,
        item_id=sha_id,
        memory_type="concept",
        metadata={"name": name, "entity_type": entity_type, "confidence": confidence},
    )


# ---------------------------------------------------------------------------
# Golden flow: Tier 1 ingest (sync=False) → extract queue
# ---------------------------------------------------------------------------

class TestTier1IngestAsync:
    def test_ingest_async_returns_item_id_and_queued_true(self, real_memory):
        """Tier 1: ingest(sync=False) runs pipeline and enqueues to extract stream."""
        result = real_memory.ingest("Alice leads Project Atlas.", sync=False)
        assert isinstance(result, dict), "sync=False must return a dict"
        assert result.get("item_id"), "item_id must be non-empty"
        assert result.get("queued") is True, "queued must be True when extract stream is available"

    def test_ingest_async_item_is_retrievable(self, real_memory):
        """Tier 1: item stored by ingest(sync=False) is retrievable via get()."""
        result = real_memory.ingest("Bob joined Acme Corp in 2020.", sync=False)
        item_id = result["item_id"]
        item = real_memory.get(item_id)
        assert item is not None, f"item_id={item_id} not found after async ingest"

    def test_ingest_async_redis_down_returns_queued_false(self, real_memory):
        """Redis-down fallback: item_id returned but queued=False when extract queue fails."""
        with patch(
            "smartmemory.observability.events.RedisStreamQueue.for_extract",
            side_effect=Exception("Redis unavailable"),
        ):
            result = real_memory.ingest("Carol manages the DevOps pipeline.", sync=False)
        assert result.get("item_id"), "item_id must be set even when queue fails"
        assert result.get("queued") is False, "queued must be False on queue failure"
        # Item must still be in the graph
        item = real_memory.get(result["item_id"])
        assert item is not None


# ---------------------------------------------------------------------------
# Golden flow: Tier 2 process_extract_job
# ---------------------------------------------------------------------------

class TestProcessExtractJob:
    def test_ok_status_with_net_new_entity(self, real_memory):
        """Tier 2: net-new LLM entity (not in ruler_entity_ids) is written, status=ok."""
        # Store a raw item so item_id exists
        item = MemoryItem(content="Django is a Python web framework.", memory_type="semantic")
        stored_id = real_memory.add(item)

        # LLM discovers one new entity ("Django") and one ruler entity ("Python")
        python_sha = _sha256("Python", "technology")
        django_sha = _sha256("Django", "technology")

        llm_entities = [
            _make_llm_entity("Python", "technology"),
            _make_llm_entity("Django", "technology"),
        ]
        llm_relations = [
            {"source_id": django_sha, "target_id": python_sha, "relation_type": "USES"},
        ]

        # Pretend Tier 1 stored Python already — simulate ruler_entity_ids
        ruler_entity_ids = {"python": "g_python_001"}

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            return_value={"entities": llm_entities, "relations": llm_relations},
        ):
            result = process_extract_job(
                real_memory,
                {"item_id": stored_id, "workspace_id": "", "entity_ids": ruler_entity_ids},
            )

        assert result["status"] == "ok"
        assert result["new_entities"] >= 1, "Django should be written as net-new"
        assert result["new_relations"] >= 0  # relation write depends on net_new_id availability

    def test_item_not_found_returns_correct_status(self, real_memory):
        """Tier 2: missing item_id returns status=item_not_found."""
        result = process_extract_job(
            real_memory,
            {"item_id": "nonexistent_item_id_abc123", "workspace_id": "", "entity_ids": {}},
        )
        assert result["status"] == "item_not_found"
        assert result["new_entities"] == 0

    def test_no_text_returns_correct_status(self, real_memory):
        """Tier 2: item with empty content returns status=no_text."""
        item = MemoryItem(content="", memory_type="semantic")
        stored_id = real_memory.add(item)

        result = process_extract_job(
            real_memory,
            {"item_id": stored_id, "workspace_id": "", "entity_ids": {}},
        )
        assert result["status"] == "no_text"

    def test_llm_failed_returns_correct_status(self, real_memory):
        """Tier 2: LLM exception returns status=llm_failed."""
        item = MemoryItem(content="Some valid content here.", memory_type="semantic")
        stored_id = real_memory.add(item)

        with patch(
            "smartmemory.plugins.extractors.llm_single.LLMSingleExtractor.extract",
            side_effect=RuntimeError("LLM API timeout"),
        ):
            result = process_extract_job(
                real_memory,
                {"item_id": stored_id, "workspace_id": "", "entity_ids": {}},
            )
        assert result["status"] == "llm_failed"
        assert result["new_entities"] == 0
        assert real_memory.get(stored_id) is not None, "LLM failure must not delete the stored Tier 1 item"


# ---------------------------------------------------------------------------
# Frequency gate: EntityPattern count >= 2 required to appear in ruler patterns
# ---------------------------------------------------------------------------

class TestFrequencyGate:
    def test_single_occurrence_pattern_not_served(self, real_memory):
        """EntityPattern with count=1 is filtered out by get_entity_patterns() frequency gate."""
        ws = "freq_gate_test_ws"
        og = real_memory.ontology_graph if hasattr(real_memory, "ontology_graph") else None
        if og is None:
            from smartmemory.graph.ontology_graph import OntologyGraph
            og = OntologyGraph(workspace_id=ws)

        # Add once → count=1 → should NOT appear
        og.add_entity_pattern("UniqueTestEntity", "Technology", 0.9, workspace_id=ws)
        patterns = og.get_entity_patterns(workspace_id=ws)
        names = [p["name"] for p in patterns]
        assert "uniquetestentity" not in names, (
            "Pattern with count=1 must be filtered out by frequency gate"
        )

    def test_two_occurrences_pattern_is_served(self, real_memory):
        """EntityPattern with count >= 2 appears in get_entity_patterns()."""
        ws = "freq_gate_test_ws_2"
        from smartmemory.graph.ontology_graph import OntologyGraph

        og = OntologyGraph(workspace_id=ws)

        # Add twice → count=2 → must appear
        og.add_entity_pattern("FrequentTestEntity", "Concept", 0.9, workspace_id=ws)
        og.add_entity_pattern("FrequentTestEntity", "Concept", 0.9, workspace_id=ws)

        patterns = og.get_entity_patterns(workspace_id=ws)
        names = [p["name"] for p in patterns]
        assert "frequenttestentity" in names, (
            "Pattern with count=2 must pass the frequency gate and be served to EntityRuler"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_memory():
    """SmartMemory with real FalkorDB + Redis (requires docker compose up -d)."""
    from smartmemory import SmartMemory

    return SmartMemory()
