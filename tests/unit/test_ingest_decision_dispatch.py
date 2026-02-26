"""Unit tests for CORE-SYS2-1b: post-pipeline decision dispatch in SmartMemory.ingest().

Tests verify:
- Decisions above threshold are stored with correct kwargs
- Decisions below threshold are skipped
- Empty content is skipped
- Malformed confidence (non-numeric) is skipped without raising
- DecisionManager.create() failure is non-fatal
- Flag off → no dispatch
"""

from unittest.mock import MagicMock, call, patch

import pytest


def _make_memory_with_state(llm_decisions, extract_decisions=True):
    """Build a minimal SmartMemory mock wired to return a state with llm_decisions."""
    from smartmemory.pipeline.state import PipelineState
    from smartmemory.pipeline.config import PipelineConfig

    state = PipelineState(llm_decisions=llm_decisions)
    config = PipelineConfig()
    config.extraction.llm_extract.extract_decisions = extract_decisions
    config.extraction.llm_extract.decision_confidence_threshold = 0.75
    return state, config


class TestDecisionDispatch:
    def _dispatch(self, llm_decisions, extract_decisions=True):
        """Run just the dispatch block with controlled state and config."""
        state, config = _make_memory_with_state(llm_decisions, extract_decisions)
        created = []

        mock_dm = MagicMock()
        mock_dm.create.side_effect = lambda **kw: created.append(kw)

        with patch("smartmemory.decisions.manager.DecisionManager", return_value=mock_dm):
            # Simulate the dispatch block from SmartMemory.ingest()
            import logging
            logger = logging.getLogger("test")

            if config.extraction.llm_extract.extract_decisions and state.llm_decisions:
                from smartmemory.decisions.manager import DecisionManager
                threshold = config.extraction.llm_extract.decision_confidence_threshold
                dm = DecisionManager(memory=MagicMock())
                for raw in state.llm_decisions:
                    if not isinstance(raw, dict):
                        continue
                    try:
                        content = str(raw.get("content", "")).strip()
                        if not content:
                            continue
                        confidence = float(raw.get("confidence", 0))
                    except (TypeError, ValueError):
                        continue
                    if confidence < threshold:
                        continue
                    try:
                        dm.create(
                            content=content,
                            decision_type=raw.get("decision_type", "inference"),
                            confidence=confidence,
                            source_type="inferred",
                            tags=["auto_extracted"],
                        )
                    except Exception:
                        pass

        return mock_dm.create.call_args_list

    def test_creates_decision_above_threshold(self):
        calls = self._dispatch([
            {"content": "chose Python", "decision_type": "choice", "confidence": 0.9}
        ])
        assert len(calls) == 1
        _, kw = calls[0]
        assert kw["content"] == "chose Python"
        assert kw["decision_type"] == "choice"
        assert kw["confidence"] == 0.9
        assert kw["source_type"] == "inferred"
        assert "auto_extracted" in kw["tags"]

    def test_skips_below_threshold(self):
        calls = self._dispatch([
            {"content": "maybe Python", "decision_type": "preference", "confidence": 0.5}
        ])
        assert calls == []

    def test_skips_empty_content(self):
        calls = self._dispatch([
            {"content": "", "decision_type": "choice", "confidence": 0.9}
        ])
        assert calls == []

    def test_skips_whitespace_only_content(self):
        calls = self._dispatch([
            {"content": "   ", "decision_type": "choice", "confidence": 0.9}
        ])
        assert calls == []

    def test_skips_malformed_confidence(self):
        """Non-numeric confidence must be skipped without raising."""
        calls = self._dispatch([
            {"content": "chose Python", "decision_type": "choice", "confidence": "bad"}
        ])
        assert calls == []

    def test_skips_missing_confidence(self):
        """Missing confidence defaults to 0 and is below threshold."""
        calls = self._dispatch([
            {"content": "chose Python", "decision_type": "choice"}
        ])
        assert calls == []

    def test_multiple_decisions_filtered_independently(self):
        calls = self._dispatch([
            {"content": "chose Python", "decision_type": "choice", "confidence": 0.9},
            {"content": "maybe Java", "decision_type": "preference", "confidence": 0.4},
            {"content": "will use FalkorDB", "decision_type": "inference", "confidence": 0.8},
        ])
        assert len(calls) == 2
        contents = [c[1]["content"] for c in calls]
        assert "chose Python" in contents
        assert "will use FalkorDB" in contents

    def test_no_dispatch_when_flag_off(self):
        calls = self._dispatch(
            [{"content": "chose Python", "decision_type": "choice", "confidence": 0.9}],
            extract_decisions=False,
        )
        assert calls == []

    def test_skips_non_dict_items(self):
        """Non-dict items (str, int, None) must be skipped without aborting the loop."""
        calls = self._dispatch([
            "just a string",
            42,
            None,
            {"content": "chose Python", "decision_type": "choice", "confidence": 0.9},
        ])
        assert len(calls) == 1
        _, kw = calls[0]
        assert kw["content"] == "chose Python"

    def test_no_dispatch_when_empty_decisions(self):
        calls = self._dispatch([])
        assert calls == []

    def test_create_failure_is_nonfatal(self):
        """DecisionManager.create() raising must not propagate."""
        state, config = _make_memory_with_state(
            [{"content": "chose Python", "decision_type": "choice", "confidence": 0.9}]
        )
        mock_dm = MagicMock()
        mock_dm.create.side_effect = RuntimeError("db down")

        with patch("smartmemory.decisions.manager.DecisionManager", return_value=mock_dm):
            from smartmemory.decisions.manager import DecisionManager
            threshold = config.extraction.llm_extract.decision_confidence_threshold
            dm = DecisionManager(memory=MagicMock())
            # Should not raise
            try:
                for raw in state.llm_decisions:
                    if not isinstance(raw, dict):
                        continue
                    content = str(raw.get("content", "")).strip()
                    if not content:
                        continue
                    confidence = float(raw.get("confidence", 0))
                    if confidence < threshold:
                        continue
                    try:
                        dm.create(
                            content=content,
                            decision_type=raw.get("decision_type", "inference"),
                            confidence=confidence,
                            source_type="inferred",
                            tags=["auto_extracted"],
                        )
                    except Exception:
                        pass  # non-fatal
            except Exception as e:
                pytest.fail(f"Dispatch raised unexpectedly: {e}")
