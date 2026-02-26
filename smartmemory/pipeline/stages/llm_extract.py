"""LLMExtract stage — extract entities and relations via LLMSingleExtractor."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor

logger = logging.getLogger(__name__)


class LLMExtractStage:
    """Extract entities and relations using an LLM in a single call."""

    def __init__(self, extractor: LLMSingleExtractor | None = None):
        """Args: extractor — optional pre-built LLMSingleExtractor (for testing/DI)."""
        self._extractor = extractor

    @property
    def name(self) -> str:
        return "llm_extract"

    # Average tokens per extraction call — used to estimate avoided tokens on cache hit.
    _AVG_EXTRACTION_TOKENS: int = 800

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        llm_cfg = config.extraction.llm_extract
        if not llm_cfg.enabled:
            if state.token_tracker:
                state.token_tracker.record_avoided(
                    "llm_extract",
                    self._AVG_EXTRACTION_TOKENS,
                    model=llm_cfg.model or "gpt-4o-mini",
                    reason="stage_disabled",
                )
            return replace(state, extraction_status="ruler_only")

        # Build input text from simplified sentences or resolved/raw text
        if state.simplified_sentences:
            text = " ".join(state.simplified_sentences)
        else:
            text = state.resolved_text or state.text

        if not text or not text.strip():
            return replace(state, extraction_status="ruler_only")

        try:
            extractor = self._extractor or self._create_extractor(llm_cfg)
            result = extractor.extract(text)

            # Track token usage (CFS-1)
            self._track_tokens(state, llm_cfg)

            entities = result.get("entities", [])
            relations = result.get("relations", [])

            # Truncate to configured limits
            entities = entities[: llm_cfg.max_entities]
            relations = relations[: llm_cfg.max_relations]

            # CORE-SYS2-1b: pass raw decision dicts through to SmartMemory.ingest() dispatch
            llm_decisions = result.get("decisions", []) if llm_cfg.extract_decisions else []

            return replace(
                state,
                llm_entities=entities,
                llm_relations=relations,
                llm_decisions=llm_decisions,
                extraction_status="llm_enriched",
            )
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return replace(state, extraction_status="llm_failed")

    def _track_tokens(self, state: PipelineState, llm_cfg) -> None:
        """Record spent or avoided tokens from the extraction call."""
        tracker = state.token_tracker
        if not tracker:
            return

        model = llm_cfg.model or "gpt-4o-mini"

        # Check the positive cache-hit flag from the extractor first.
        # This avoids false positives from missing DSPy usage data.
        try:
            from smartmemory.plugins.extractors.llm_single import was_last_extract_cached

            cache_hit = was_last_extract_cached()
        except ImportError:
            cache_hit = False

        if cache_hit:
            tracker.record_avoided("llm_extract", self._AVG_EXTRACTION_TOKENS, model=model, reason="cache_hit")
            return

        # Not a cache hit — try to capture actual token usage from DSPy.
        try:
            from smartmemory.utils.llm_client.dspy import get_last_usage

            usage = get_last_usage()
        except ImportError:
            usage = None

        if usage:
            tracker.record_spent(
                "llm_extract",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                model=model,
            )
        # If usage is None and it was NOT a cache hit, we have no data —
        # don't record anything rather than producing a false "cache_hit".

    def _create_extractor(self, llm_cfg):
        """Build an LLMSingleExtractor from config.

        When no model is explicitly configured and ``GROQ_API_KEY`` is set,
        automatically uses the higher-quality GroqExtractor (Llama-3.3-70b,
        97.7% E-F1 at 878ms) instead of defaulting to gpt-4o-mini.
        """
        import os

        from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor, LLMSingleExtractorConfig

        # Auto-detect Groq when no model explicitly configured
        if not llm_cfg.model and os.getenv("GROQ_API_KEY"):
            try:
                from smartmemory.plugins.extractors.llm_single import GroqExtractor

                extractor = GroqExtractor()
                if llm_cfg.temperature is not None:
                    extractor.cfg.temperature = llm_cfg.temperature
                if llm_cfg.max_tokens is not None:
                    extractor.cfg.max_tokens = llm_cfg.max_tokens
                # CORE-SYS2-1b: GroqExtractor is a zero-arg subclass of LLMSingleExtractor;
                # inject extract_decisions after construction via replace() (already imported).
                # Do NOT call dataclasses.replace() — only 'replace' is in scope here.
                extractor.cfg = replace(extractor.cfg, extract_decisions=llm_cfg.extract_decisions)
                return extractor
            except Exception:
                logger.debug("GroqExtractor init failed, falling back to default")

        ext_config = LLMSingleExtractorConfig()
        if llm_cfg.model:
            ext_config.model_name = llm_cfg.model
        if llm_cfg.temperature is not None:
            ext_config.temperature = llm_cfg.temperature
        if llm_cfg.max_tokens is not None:
            ext_config.max_tokens = llm_cfg.max_tokens
        # CORE-SYS2-1b: forward the flag so non-Groq providers (OpenAI, etc.) also extract decisions
        ext_config.extract_decisions = llm_cfg.extract_decisions

        return LLMSingleExtractor(config=ext_config)

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, llm_entities=[], llm_relations=[], llm_decisions=[])
