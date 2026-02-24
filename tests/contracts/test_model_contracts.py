"""
Serialization contract tests for core SmartMemory models.

Verifies that models survive to_dict → from_dict round-trips without
data loss. Catches field renames, type changes, and missing defaults.

No external services needed — pure logic tests.
"""

from datetime import datetime, timezone

import pytest

pytestmark = [pytest.mark.contract]


class TestMemoryItemContract:
    """MemoryItem round-trip serialization."""

    def test_semantic_item_round_trip(self):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="Test content for contract",
            memory_type="semantic",
            metadata={"source": "test", "priority": 1},
        )
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)

        assert restored.content == item.content
        assert restored.memory_type == item.memory_type
        assert restored.item_id == item.item_id
        assert restored.metadata == item.metadata

    def test_zettel_item_round_trip(self):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="Zettelkasten note",
            memory_type="zettel",
            metadata={"tags": ["test", "contract"]},
        )
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)

        assert restored.memory_type == "zettel"
        assert restored.metadata["tags"] == ["test", "contract"]

    def test_item_with_embedding_round_trip(self):
        from smartmemory.models.memory_item import MemoryItem

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        item = MemoryItem(
            content="Embedded content",
            memory_type="semantic",
            embedding=embedding,
        )
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)

        assert restored.embedding == embedding

    def test_item_preserves_timestamps(self):
        from smartmemory.models.memory_item import MemoryItem

        now = datetime.now(timezone.utc)
        item = MemoryItem(
            content="Timestamped content",
            memory_type="semantic",
        )
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)

        assert restored.transaction_time is not None


class TestDecisionContract:
    """Decision round-trip serialization."""

    def test_basic_decision_round_trip(self):
        from smartmemory.models.decision import Decision

        decision = Decision(
            content="User prefers TypeScript over JavaScript",
            decision_type="preference",
            confidence=0.85,
            source_type="explicit",
            domain="programming",
        )
        d = decision.to_dict()
        restored = Decision.from_dict(d)

        assert restored.content == decision.content
        assert restored.decision_type == "preference"
        assert restored.confidence == 0.85
        assert restored.source_type == "explicit"
        assert restored.domain == "programming"
        assert restored.status == "active"

    def test_decision_with_evidence_round_trip(self):
        from smartmemory.models.decision import Decision

        decision = Decision(
            content="Test decision with evidence",
            decision_type="inference",
            confidence=0.7,
            source_type="reasoning",
            evidence_ids=["ev1", "ev2", "ev3"],
        )
        d = decision.to_dict()
        restored = Decision.from_dict(d)

        assert restored.evidence_ids == ["ev1", "ev2", "ev3"]

    def test_superseded_decision_round_trip(self):
        from smartmemory.models.decision import Decision

        decision = Decision(
            content="Superseded decision",
            decision_type="classification",
            confidence=0.6,
            source_type="inferred",
            status="superseded",
            superseded_by="new_decision_id",
        )
        d = decision.to_dict()
        restored = Decision.from_dict(d)

        assert restored.status == "superseded"
        assert restored.superseded_by == "new_decision_id"


class TestReasoningTraceContract:
    """ReasoningTrace round-trip serialization."""

    def test_basic_trace_round_trip(self):
        from smartmemory.models.reasoning import ReasoningTrace, ReasoningStep, TaskContext

        trace = ReasoningTrace(
            trace_id="test-trace-123",
            task_context=TaskContext(
                goal="Analyze user preferences",
                domain="testing",
            ),
            steps=[
                ReasoningStep(type="thought", content="Let me consider the evidence"),
                ReasoningStep(type="observation", content="User mentioned TypeScript 5 times"),
                ReasoningStep(type="conclusion", content="User prefers TypeScript"),
            ],
        )
        d = trace.to_dict()
        restored = ReasoningTrace.from_dict(d)

        assert restored.task_context is not None
        assert restored.task_context.goal == "Analyze user preferences"
        assert len(restored.steps) == 3
        assert restored.steps[0].type == "thought"
        assert restored.steps[2].content == "User prefers TypeScript"


class TestReasoningStepContract:
    """ReasoningStep round-trip serialization."""

    def test_all_step_types(self):
        from smartmemory.models.reasoning import ReasoningStep

        step_types = ["thought", "action", "observation", "decision", "conclusion", "reflection"]
        for step_type in step_types:
            step = ReasoningStep(type=step_type, content=f"Test {step_type}")
            d = step.to_dict()
            restored = ReasoningStep.from_dict(d)
            assert restored.type == step_type
            assert restored.content == f"Test {step_type}"


class TestDataclassModelMixinContract:
    """Verify the base DataclassModelMixin to_dict/from_dict contract."""

    def test_to_dict_returns_dict(self):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content="test", memory_type="semantic")
        d = item.to_dict()
        assert isinstance(d, dict)

    def test_from_dict_accepts_dict(self):
        from smartmemory.models.memory_item import MemoryItem

        d = {"content": "test", "memory_type": "semantic", "item_id": "test_id"}
        item = MemoryItem.from_dict(d)
        assert item.content == "test"
        assert item.item_id == "test_id"

    def test_from_dict_handles_extra_fields(self):
        from smartmemory.models.memory_item import MemoryItem

        d = {"content": "test", "memory_type": "semantic", "unknown_field": "ignored"}
        item = MemoryItem.from_dict(d)
        assert item.content == "test"

    def test_from_dict_handles_missing_optional_fields(self):
        from smartmemory.models.memory_item import MemoryItem

        d = {"content": "test"}
        item = MemoryItem.from_dict(d)
        assert item.content == "test"
        assert item.memory_type == "semantic"  # default
