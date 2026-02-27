"""Unit tests for CORE-SYS2-1c: post-pipeline reasoning dispatch in ingest()."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock


@dataclass
class MockEvaluation:
    quality_score: float = 0.85
    should_store: bool = True


@dataclass
class MockTrace:
    trace_id: str = "trace_abc"
    has_explicit_markup: bool = False
    evaluation: MockEvaluation = field(default_factory=MockEvaluation)

    @property
    def step_count(self) -> int:
        return 3

    @property
    def content(self) -> str:
        return "Goal: test\nThought: reasoning step"

    def to_dict(self):
        return {"trace_id": self.trace_id, "steps": []}


def _run_ingest_with_trace(trace=None, reasoning_enabled=True, graph=None):
    """Run SmartMemory.ingest() with a mocked pipeline that returns a state with reasoning_trace.

    Returns (smart_memory_instance, add_mock, graph_mock).
    """
    from smartmemory.pipeline.state import PipelineState

    # Build a realistic post-pipeline state.
    # memory_type="semantic" reflects a typical classify-stage output — required by
    # the CORE-SYS2-1d PRODUCED gate (target must be in schema_validator allowed set).
    state = PipelineState(
        text="test content",
        item_id="item_123",
        memory_type="semantic",
        reasoning_trace=trace,
    )

    sm = MagicMock()
    sm._pipeline_profile = None
    sm.scope_provider.get_scope.side_effect = Exception("no scope")
    sm._pipeline_runner.run.return_value = state
    sm._graph = graph

    # Make add() return a predictable ID
    sm.add.return_value = "reasoning_item_456"

    # Import the real _build_pipeline_config so flags propagate correctly
    from smartmemory.smart_memory import SmartMemory

    sm._build_pipeline_config = lambda **kw: SmartMemory._build_pipeline_config(sm, **kw)

    # Call the real ingest() as an unbound method
    SmartMemory.ingest(sm, "test content", extract_reasoning=reasoning_enabled)

    return sm, sm.add, sm._graph


class TestReasoningDispatch:
    def test_stores_reasoning_item_via_add(self):
        """When reasoning_detect is enabled and trace exists, dispatch stores via add()."""
        trace = MockTrace()
        sm, add_mock, _ = _run_ingest_with_trace(trace=trace, graph=None)

        # add() should be called with a MemoryItem
        assert add_mock.called
        item_arg = add_mock.call_args[0][0]
        assert item_arg.memory_type == "reasoning"
        assert item_arg.content == "Goal: test\nThought: reasoning step"
        assert item_arg.metadata["trace_id"] == "trace_abc"
        assert item_arg.metadata["auto_extracted"] is True
        assert item_arg.metadata["quality_score"] == 0.85
        assert item_arg.metadata["step_count"] == 3
        assert item_arg.metadata["has_explicit_markup"] is False

    def test_creates_produced_edge_when_graph_available(self):
        """PRODUCED edge links reasoning item → source item."""
        trace = MockTrace()
        graph_mock = MagicMock()
        sm, _, _ = _run_ingest_with_trace(trace=trace, graph=graph_mock)

        graph_mock.add_edge.assert_called_once()
        kwargs = graph_mock.add_edge.call_args[1]
        assert kwargs["source_id"] == "reasoning_item_456"
        assert kwargs["target_id"] == "item_123"
        assert kwargs["edge_type"] == "PRODUCED"
        assert kwargs["properties"]["confidence"] == 0.85

    def test_no_edge_when_no_graph(self):
        """When graph is None, dispatch stores item but skips edge creation."""
        trace = MockTrace()
        sm, add_mock, graph_mock = _run_ingest_with_trace(trace=trace, graph=None)

        assert add_mock.called
        # graph is None, so no add_edge call

    def test_no_dispatch_when_trace_is_none(self):
        """When pipeline produces no trace, dispatch is skipped entirely."""
        sm, add_mock, _ = _run_ingest_with_trace(trace=None, reasoning_enabled=True)

        # add() should NOT be called for reasoning (may be called by pipeline internals)
        for c in add_mock.call_args_list:
            if len(c[0]) > 0 and hasattr(c[0][0], "memory_type"):
                assert c[0][0].memory_type != "reasoning"

    def test_no_dispatch_when_flag_disabled(self):
        """When extract_reasoning=False, no reasoning dispatch even if trace somehow exists."""
        sm, add_mock, _ = _run_ingest_with_trace(trace=MockTrace(), reasoning_enabled=False)

        for c in add_mock.call_args_list:
            if len(c[0]) > 0 and hasattr(c[0][0], "memory_type"):
                assert c[0][0].memory_type != "reasoning"

    def test_dispatch_is_non_fatal(self):
        """Exceptions in dispatch must not propagate — ingest() still returns item_id."""
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.smart_memory import SmartMemory

        trace = MockTrace()
        state = PipelineState(text="test", item_id="item_123", reasoning_trace=trace)

        sm = MagicMock()
        sm._pipeline_profile = None
        sm.scope_provider.get_scope.side_effect = Exception("no scope")
        sm._pipeline_runner.run.return_value = state
        sm._graph = None

        # Make add() raise an exception
        sm.add.side_effect = RuntimeError("storage broken")

        sm._build_pipeline_config = lambda **kw: SmartMemory._build_pipeline_config(sm, **kw)

        # Must NOT raise
        SmartMemory.ingest(sm, "test content", extract_reasoning=True)

    def test_quality_score_none_when_no_evaluation(self):
        """When trace has no evaluation, quality_score is None and edge confidence is 0.5."""
        trace = MockTrace(evaluation=None)
        graph_mock = MagicMock()
        sm, add_mock, _ = _run_ingest_with_trace(trace=trace, graph=graph_mock)

        item_arg = add_mock.call_args[0][0]
        assert item_arg.metadata["quality_score"] is None

        edge_kwargs = graph_mock.add_edge.call_args[1]
        assert edge_kwargs["properties"]["confidence"] == 0.5
