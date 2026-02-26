"""Unit tests for CORE-SYS2-1c: ReasoningDetectStage."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from smartmemory.pipeline.config import PipelineConfig, ReasoningDetectConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.reasoning_detect import ReasoningDetectStage


def _make_config(enabled=True, min_quality=0.4, use_llm=True):
    cfg = PipelineConfig()
    cfg.extraction.reasoning_detect = ReasoningDetectConfig(
        enabled=enabled,
        min_quality_score=min_quality,
        use_llm_detection=use_llm,
    )
    return cfg


@dataclass
class MockTrace:
    trace_id: str = "trace_abc"
    step_count: int = 3
    has_explicit_markup: bool = False
    evaluation: None = None

    @property
    def content(self):
        return "Goal: test\nThought: reasoning step"

    def to_dict(self):
        return {"trace_id": self.trace_id}


LONG_TEXT = "Let me think about this problem step by step. First, I need to analyze the issue. " * 5


class TestReasoningDetectStage:
    def test_name(self):
        assert ReasoningDetectStage().name == "reasoning_detect"

    def test_returns_unchanged_when_disabled(self):
        stage = ReasoningDetectStage()
        state = PipelineState(text=LONG_TEXT)
        config = _make_config(enabled=False)
        result = stage.execute(state, config)
        assert result is state  # exact same object — no replace()

    def test_returns_unchanged_when_text_too_short(self):
        stage = ReasoningDetectStage()
        state = PipelineState(text="short")
        config = _make_config(enabled=True)
        result = stage.execute(state, config)
        assert result.reasoning_trace is None

    def test_returns_unchanged_when_no_reasoning_detected(self):
        stage = ReasoningDetectStage()
        state = PipelineState(text=LONG_TEXT)
        config = _make_config(enabled=True)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": [],
            "relations": [],
            "reasoning_trace": None,
        }

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ):
            result = stage.execute(state, config)
        assert result.reasoning_trace is None

    def test_writes_trace_to_state(self):
        stage = ReasoningDetectStage()
        state = PipelineState(text=LONG_TEXT)
        config = _make_config(enabled=True)

        trace = MockTrace()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": [],
            "relations": [],
            "reasoning_trace": trace,
        }

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ):
            result = stage.execute(state, config)
        assert result.reasoning_trace is trace

    def test_prefers_simplified_sentences(self):
        stage = ReasoningDetectStage()
        state = PipelineState(
            text="original text that is long enough to pass the length check easily",
            resolved_text="resolved text that is long enough to pass the length check easily",
            simplified_sentences=[
                "Simplified sentence one about reasoning.",
                "Simplified sentence two about more reasoning steps.",
                "Simplified sentence three with additional details.",
            ],
        )
        config = _make_config(enabled=True)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": [],
            "relations": [],
            "reasoning_trace": None,
        }

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ):
            stage.execute(state, config)

        call_text = mock_extractor.extract.call_args[0][0]
        assert "Simplified sentence one" in call_text
        assert "original text" not in call_text

    def test_falls_back_to_resolved_text(self):
        stage = ReasoningDetectStage()
        state = PipelineState(
            text="original text that is long enough to pass the length check easily",
            resolved_text="resolved text that is long enough to pass the length check easily",
            simplified_sentences=[],
        )
        config = _make_config(enabled=True)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": [],
            "relations": [],
            "reasoning_trace": None,
        }

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ):
            stage.execute(state, config)

        call_text = mock_extractor.extract.call_args[0][0]
        assert "resolved text" in call_text

    def test_falls_back_to_raw_text(self):
        stage = ReasoningDetectStage()
        state = PipelineState(
            text="original text that is long enough to pass the length check for extraction",
        )
        config = _make_config(enabled=True)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": [],
            "relations": [],
            "reasoning_trace": None,
        }

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ):
            stage.execute(state, config)

        call_text = mock_extractor.extract.call_args[0][0]
        assert "original text" in call_text

    def test_nonfatal_on_extractor_exception(self):
        stage = ReasoningDetectStage()
        state = PipelineState(text=LONG_TEXT)
        config = _make_config(enabled=True)

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            side_effect=RuntimeError("extractor broken"),
        ):
            result = stage.execute(state, config)
        # Must not raise, must return state without trace
        assert result.reasoning_trace is None

    def test_nonfatal_on_extract_call_exception(self):
        """extract() raising must not propagate — returns state without trace."""
        stage = ReasoningDetectStage()
        state = PipelineState(text=LONG_TEXT)
        config = _make_config(enabled=True)

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("extract() broken")

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ):
            result = stage.execute(state, config)
        assert result.reasoning_trace is None

    def test_undo_clears_trace(self):
        stage = ReasoningDetectStage()
        state = PipelineState(reasoning_trace=MockTrace())
        result = stage.undo(state)
        assert result.reasoning_trace is None

    def test_forwards_config_to_extractor(self):
        stage = ReasoningDetectStage()
        state = PipelineState(text=LONG_TEXT)
        config = _make_config(enabled=True, min_quality=0.7, use_llm=False)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "entities": [],
            "relations": [],
            "reasoning_trace": None,
        }

        with patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractor",
            return_value=mock_extractor,
        ), patch(
            "smartmemory.plugins.extractors.reasoning.ReasoningExtractorConfig",
        ) as mock_cfg_cls:
            mock_cfg_cls.return_value = MagicMock()
            stage.execute(state, config)
            mock_cfg_cls.assert_called_once_with(
                min_quality_score=0.7,
                use_llm_detection=False,
            )
