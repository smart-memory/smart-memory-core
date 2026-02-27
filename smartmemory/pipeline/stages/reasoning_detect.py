"""ReasoningDetect stage — detect reasoning traces via ReasoningExtractor (CORE-SYS2-1c)."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class ReasoningDetectStage:
    """Detect reasoning traces in content using ReasoningExtractor.

    Opt-in stage (config.extraction.reasoning_detect.enabled).
    Pure detection — no storage side effects. The detected trace is
    written to state.reasoning_trace for post-pipeline dispatch.
    """

    @property
    def name(self) -> str:
        return "reasoning_detect"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        rd_cfg = config.extraction.reasoning_detect
        if not rd_cfg.enabled and state.memory_type != "reasoning":
            return state

        # Build text from best available source
        text = "\n".join(state.simplified_sentences) if state.simplified_sentences else (
            state.resolved_text or state.text
        )
        if not text or len(text.strip()) < 50:
            return state

        try:
            from smartmemory.plugins.extractors.reasoning import (
                ReasoningExtractor,
                ReasoningExtractorConfig,
            )

            extractor_cfg = ReasoningExtractorConfig(
                min_quality_score=rd_cfg.min_quality_score,
                use_llm_detection=rd_cfg.use_llm_detection,
            )
            extractor = ReasoningExtractor(config=extractor_cfg)
            result = extractor.extract(text)

            trace = result.get("reasoning_trace")
            if trace is not None:
                return replace(state, reasoning_trace=trace)
        except Exception as e:
            logger.warning("ReasoningDetectStage failed (non-fatal): %s", e)

        return state

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, reasoning_trace=None)
