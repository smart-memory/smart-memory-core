"""Unit tests for LLMExtractStage."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig, LLMExtractConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.llm_extract import LLMExtractStage


def _mock_extractor(entities=None, relations=None):
    """Build a mock LLMSingleExtractor."""
    ext = MagicMock()
    ext.extract.return_value = {
        "entities": entities or [],
        "relations": relations or [],
    }
    return ext


class TestLLMExtractStage:
    """Tests for the LLM extract pipeline stage."""

    def test_disabled_mode_returns_unchanged(self):
        """When disabled, state is returned unchanged."""
        stage = LLMExtractStage()
        state = PipelineState(text="Claude is an AI.")
        config = PipelineConfig()
        config.extraction.llm_extract = LLMExtractConfig(enabled=False)

        result = stage.execute(state, config)

        assert result.llm_entities == []
        assert result.llm_relations == []

    def test_extractor_called_with_text(self):
        """Extractor is called with the input text."""
        ext = _mock_extractor()
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Claude is an AI assistant.")
        config = PipelineConfig()

        stage.execute(state, config)

        ext.extract.assert_called_once_with("Claude is an AI assistant.")

    def test_entities_and_relations_populated(self):
        """Extraction result populates llm_entities and llm_relations."""
        entities = [MagicMock(content="Claude")]
        relations = [{"source_id": "a", "target_id": "b", "relation_type": "is_a"}]
        ext = _mock_extractor(entities=entities, relations=relations)
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Claude is an AI.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.llm_entities == entities
        assert result.llm_relations == relations

    def test_uses_simplified_sentences(self):
        """When simplified_sentences set, joins them for extraction."""
        ext = _mock_extractor()
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(
            text="Original.",
            simplified_sentences=["Claude is great.", "Claude codes."],
        )
        config = PipelineConfig()

        stage.execute(state, config)

        ext.extract.assert_called_once_with("Claude is great. Claude codes.")

    def test_truncation_entities(self):
        """Entities are truncated to max_entities."""
        entities = [MagicMock() for _ in range(15)]
        ext = _mock_extractor(entities=entities)
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Many entities here in the text for extraction.")
        config = PipelineConfig()
        config.extraction.llm_extract = LLMExtractConfig(max_entities=5)

        result = stage.execute(state, config)

        assert len(result.llm_entities) == 5

    def test_truncation_relations(self):
        """Relations are truncated to max_relations."""
        relations = [{"source_id": f"s{i}", "target_id": f"t{i}", "relation_type": "r"} for i in range(40)]
        ext = _mock_extractor(relations=relations)
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Many relations here in the text for extraction.")
        config = PipelineConfig()
        config.extraction.llm_extract = LLMExtractConfig(max_relations=10)

        result = stage.execute(state, config)

        assert len(result.llm_relations) == 10

    def test_empty_result(self):
        """Empty extraction result leaves state unchanged."""
        ext = _mock_extractor()
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Nothing to extract here.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.llm_entities == []
        assert result.llm_relations == []

    def test_disabled_sets_ruler_only(self):
        """When disabled, extraction_status is set to ruler_only."""
        stage = LLMExtractStage()
        state = PipelineState(text="Claude is an AI.")
        config = PipelineConfig()
        config.extraction.llm_extract = LLMExtractConfig(enabled=False)

        result = stage.execute(state, config)

        assert result.extraction_status == "ruler_only"

    def test_success_sets_llm_enriched(self):
        """When extraction succeeds, extraction_status is llm_enriched."""
        ext = _mock_extractor(entities=[MagicMock()])
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Claude is an AI assistant.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.extraction_status == "llm_enriched"

    def test_failure_sets_llm_failed(self):
        """When extraction throws, extraction_status is llm_failed."""
        ext = MagicMock()
        ext.extract.side_effect = RuntimeError("LLM unavailable")
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="Claude is an AI assistant.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.extraction_status == "llm_failed"

    def test_empty_text_sets_ruler_only(self):
        """When text is empty, extraction_status is ruler_only."""
        ext = _mock_extractor()
        stage = LLMExtractStage(extractor=ext)
        state = PipelineState(text="")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert result.extraction_status == "ruler_only"

    def test_undo_clears_llm_fields(self):
        """Undo resets llm_entities and llm_relations."""
        stage = LLMExtractStage()
        state = PipelineState(
            text="Some text.",
            llm_entities=[MagicMock()],
            llm_relations=[{"source_id": "a", "target_id": "b"}],
        )

        result = stage.undo(state)

        assert result.llm_entities == []
        assert result.llm_relations == []
