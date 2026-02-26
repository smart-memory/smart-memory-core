"""Integration tests for CORE-SYS2-1b: auto-extract decisions via LLM schema extension.

Golden flow: ingest decision-language content with extract_decisions=True, verify that
at least one decision MemoryItem was created and carries the 'auto_extracted' tag.

Requires: FalkorDB + MongoDB running (docker compose up -d from SmartMemory root).
Run:
    PYTHONPATH=. pytest tests/integration/test_ingest_decision_extraction.py -v -s
"""

import pytest
from smartmemory import SmartMemory
from smartmemory.decisions.manager import DecisionManager


@pytest.fixture
def memory():
    """SmartMemory against real infrastructure."""
    sm = SmartMemory()
    yield sm


class TestDecisionExtractionGoldenFlow:
    """Golden flow: ingest → LLM extracts decisions → DecisionManager stores them."""

    def test_decision_created_from_decision_language(self, memory):
        """Ingest content rich with decision language; assert at least one Decision is created."""
        # Ingest with the flag on
        item_id = memory.ingest(
            item=(
                "After evaluating several options we decided to adopt Python for all "
                "backend services because of its mature ML ecosystem. We chose FalkorDB "
                "as our graph database due to its Redis-native performance."
            ),
            extract_decisions=True,
        )
        # Normalise return value (str or dict)
        if isinstance(item_id, dict):
            item_id = item_id.get("item_id") or item_id.get("memory_node_id")
        assert item_id is not None, "ingest() returned no item_id"

        # Query decisions via DecisionManager
        dm = DecisionManager(memory=memory)
        from smartmemory.decisions.queries import DecisionQueries
        dq = DecisionQueries(memory=memory)

        decisions = dq.get_active_decisions(min_confidence=0.0, limit=20)

        # At least one decision should carry the auto_extracted tag from this run
        auto_extracted = [
            d for d in decisions
            if "auto_extracted" in (d.tags or [])
        ]
        assert len(auto_extracted) >= 1, (
            f"Expected at least 1 auto-extracted decision, found 0. "
            f"All decisions: {[d.content for d in decisions]}"
        )

    def test_no_decision_from_neutral_content(self, memory):
        """Neutral factual content should not generate auto-extracted decisions."""
        # Record baseline count before ingest
        from smartmemory.decisions.queries import DecisionQueries
        dq = DecisionQueries(memory=memory)
        before = dq.get_active_decisions(min_confidence=0.0, limit=50)
        before_auto = {d.decision_id for d in before if "auto_extracted" in (d.tags or [])}

        memory.ingest(
            item="The sky is blue. Water boils at 100 degrees Celsius.",
            extract_decisions=True,
        )

        after = dq.get_active_decisions(min_confidence=0.0, limit=50)
        after_auto = {d.decision_id for d in after if "auto_extracted" in (d.tags or [])}

        new_decisions = after_auto - before_auto
        # LLMs are non-deterministic; neutral content SHOULD produce 0 but may occasionally
        # yield a spurious extraction.  Allow at most 1 to avoid flakiness while still
        # catching systematic false-positive regressions (e.g. broken confidence gate).
        assert len(new_decisions) <= 1, (
            f"Neutral content produced {len(new_decisions)} auto-extracted decisions — "
            f"expected 0 (at most 1 from LLM non-determinism). IDs: {new_decisions}"
        )

    def test_ingest_succeeds_without_flag(self, memory):
        """extract_decisions=False (default) must not affect normal ingest behaviour."""
        item_id = memory.ingest(
            item="We decided to migrate all data to FalkorDB for better graph performance.",
            extract_decisions=False,
        )
        if isinstance(item_id, dict):
            item_id = item_id.get("item_id") or item_id.get("memory_node_id")
        assert item_id is not None, "ingest() with extract_decisions=False returned None"

        item = memory.get(item_id)
        assert item is not None, "Stored item is not retrievable after ingest without flag"

    def test_ingest_nonfatal_on_bad_payload(self, memory):
        """Ingest must complete even if the LLM returns garbage in the decisions field.

        We can't force a bad payload from outside, but we can verify that with valid
        content ingest never raises regardless of what the LLM includes.
        """
        # Any content with the flag on must not raise at the ingest() boundary
        try:
            item_id = memory.ingest(
                item="Maybe we should consider Python. Or perhaps not.",
                extract_decisions=True,
            )
        except Exception as exc:
            pytest.fail(
                f"ingest() raised unexpectedly with extract_decisions=True: {exc!r}"
            )
        # If we reach here, the pipeline was non-fatal as designed
