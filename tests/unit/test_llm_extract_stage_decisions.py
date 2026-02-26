"""Unit tests for CORE-SYS2-1b: LLMExtractStage decisions pass-through and Groq injection."""

from unittest.mock import MagicMock, patch

import pytest

from smartmemory.pipeline.state import PipelineState


def _make_config(extract_decisions: bool = False, enabled: bool = True):
    from smartmemory.pipeline.config import LLMExtractConfig, PipelineConfig

    config = PipelineConfig()
    config.extraction.llm_extract.enabled = enabled
    config.extraction.llm_extract.extract_decisions = extract_decisions
    return config


class TestLLMExtractStageDecisions:
    def _run_stage(self, extractor_result: dict, extract_decisions: bool) -> PipelineState:
        from smartmemory.pipeline.stages.llm_extract import LLMExtractStage

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = extractor_result
        stage = LLMExtractStage(extractor=mock_extractor)
        state = PipelineState(text="We decided to use Python.")
        config = _make_config(extract_decisions=extract_decisions)
        return stage.execute(state, config)

    def test_llm_decisions_written_when_enabled(self):
        result = self._run_stage(
            extractor_result={
                "entities": [],
                "relations": [],
                "decisions": [{"content": "chose Python", "decision_type": "choice", "confidence": 0.9}],
            },
            extract_decisions=True,
        )
        assert len(result.llm_decisions) == 1
        assert result.llm_decisions[0]["content"] == "chose Python"

    def test_llm_decisions_empty_when_disabled(self):
        result = self._run_stage(
            extractor_result={
                "entities": [],
                "relations": [],
                "decisions": [{"content": "chose Python", "decision_type": "choice", "confidence": 0.9}],
            },
            extract_decisions=False,
        )
        assert result.llm_decisions == []

    def test_llm_decisions_empty_when_extractor_returns_no_key(self):
        result = self._run_stage(
            extractor_result={"entities": [], "relations": []},
            extract_decisions=True,
        )
        assert result.llm_decisions == []

    def test_undo_clears_decisions(self):
        from dataclasses import replace
        from smartmemory.pipeline.stages.llm_extract import LLMExtractStage

        stage = LLMExtractStage()
        state = PipelineState(
            llm_decisions=[{"content": "X", "decision_type": "choice", "confidence": 0.9}]
        )
        undone = stage.undo(state)
        assert undone.llm_decisions == []


class TestGroqCfgInjection:
    def test_extract_decisions_injected_into_groq_cfg(self, monkeypatch):
        """After _create_extractor(), GroqExtractor.cfg.extract_decisions matches llm_cfg.

        mock_groq.cfg must be a real LLMSingleExtractorConfig dataclass — not a MagicMock —
        so that dataclasses.replace() inside _create_extractor() can operate on it without
        raising TypeError, and the injection actually runs.
        """
        from smartmemory.pipeline.stages.llm_extract import LLMExtractStage
        from smartmemory.pipeline.config import LLMExtractConfig
        from smartmemory.plugins.extractors.llm_single import LLMSingleExtractorConfig

        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        # Use a real dataclass for cfg so dataclasses.replace() works inside _create_extractor
        mock_groq = MagicMock()
        mock_groq.cfg = LLMSingleExtractorConfig()

        with patch(
            "smartmemory.plugins.extractors.llm_single.GroqExtractor",
            return_value=mock_groq,
        ):
            stage = LLMExtractStage()
            llm_cfg = LLMExtractConfig(extract_decisions=True)
            stage._create_extractor(llm_cfg)

        # replace() assigns a new cfg back onto mock_groq; assert the flag was injected
        assert mock_groq.cfg.extract_decisions is True
