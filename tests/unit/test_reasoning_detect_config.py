"""Unit tests for CORE-SYS2-1c: ReasoningDetectConfig, ExtractionConfig, PipelineState serialization."""

import json
from dataclasses import dataclass


class TestReasoningDetectConfig:
    def test_defaults(self):
        from smartmemory.pipeline.config import ReasoningDetectConfig

        cfg = ReasoningDetectConfig()
        assert cfg.enabled is False
        assert cfg.min_quality_score == 0.4
        assert cfg.use_llm_detection is True

    def test_extraction_config_has_reasoning_detect(self):
        from smartmemory.pipeline.config import ExtractionConfig, ReasoningDetectConfig

        ec = ExtractionConfig()
        assert isinstance(ec.reasoning_detect, ReasoningDetectConfig)
        assert ec.reasoning_detect.enabled is False

    def test_with_reasoning_factory(self):
        from smartmemory.pipeline.config import PipelineConfig

        cfg = PipelineConfig.with_reasoning()
        assert cfg.extraction.reasoning_detect.enabled is True
        # Other defaults unchanged
        assert cfg.extraction.llm_extract.enabled is True
        assert cfg.extraction.llm_extract.extract_decisions is False


class TestPipelineStateReasoningTrace:
    def test_to_dict_with_trace_is_json_serializable(self):
        """reasoning_trace with to_dict() must produce a JSON-serializable dict."""
        from smartmemory.pipeline.state import PipelineState

        @dataclass
        class MockTrace:
            trace_id: str = "t1"

            def to_dict(self):
                return {"trace_id": self.trace_id, "steps": []}

        state = PipelineState(reasoning_trace=MockTrace())
        d = state.to_dict()
        # Must not raise TypeError
        serialized = json.dumps(d)
        assert '"trace_id": "t1"' in serialized

    def test_to_dict_with_none_trace(self):
        from smartmemory.pipeline.state import PipelineState

        state = PipelineState()
        d = state.to_dict()
        assert d["reasoning_trace"] is None
        # Must be JSON-serializable
        json.dumps(d)

    def test_to_dict_with_trace_missing_to_dict(self):
        """Object without to_dict() gets serialized as None."""
        from smartmemory.pipeline.state import PipelineState

        state = PipelineState(reasoning_trace="not a trace object")
        d = state.to_dict()
        # String has no to_dict() — but it's also not None, so the handler fires
        # and hasattr(str, "to_dict") is False → val becomes None
        # Wait — actually strings don't have to_dict, so val = None
        # But actually "not a trace object" is not None, so the elif fires
        # Let me re-check: the handler is `elif f.name == "reasoning_trace" and val is not None`
        # val = "not a trace object", hasattr returns False, so val = None
        assert d["reasoning_trace"] is None

    def test_from_dict_drops_reasoning_trace(self):
        from smartmemory.pipeline.state import PipelineState

        state = PipelineState.from_dict({
            "text": "hello",
            "reasoning_trace": {"trace_id": "t1", "steps": []},
        })
        assert state.reasoning_trace is None
        assert state.text == "hello"

    def test_default_reasoning_trace_is_none(self):
        from smartmemory.pipeline.state import PipelineState

        state = PipelineState()
        assert state.reasoning_trace is None
