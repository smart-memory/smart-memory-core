"""Unit tests for EntityRulerStage."""

import pytest

pytestmark = pytest.mark.unit


from unittest.mock import MagicMock

from smartmemory.pipeline.config import PipelineConfig, EntityRulerConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage, _map_label


def _make_mock_nlp(entities=None):
    """Build a mock spaCy Language that returns configurable entities."""
    nlp = MagicMock()

    def process(text):
        doc = MagicMock()
        doc.text = text
        ents = []
        for e in (entities or []):
            ent = MagicMock()
            ent.text = e["text"]
            ent.label_ = e["label"]
            ent.start_char = e.get("start", 0)
            ent.end_char = e.get("end", len(e["text"]))
            ents.append(ent)
        doc.ents = ents
        return doc

    nlp.side_effect = process
    return nlp


class TestEntityRulerStage:
    """Tests for the entity ruler pipeline stage."""

    def test_disabled_mode_returns_unchanged(self):
        """When disabled, state is returned unchanged."""
        stage = EntityRulerStage()
        state = PipelineState(text="John works at Google.")
        config = PipelineConfig()
        config.extraction.entity_ruler = EntityRulerConfig(enabled=False)

        result = stage.execute(state, config)

        assert result.ruler_entities == []

    def test_entity_detection(self):
        """Entities detected by spaCy NER are captured."""
        nlp = _make_mock_nlp([
            {"text": "John", "label": "PERSON", "start": 0, "end": 4},
            {"text": "Google", "label": "ORG", "start": 14, "end": 20},
        ])
        stage = EntityRulerStage(nlp=nlp)
        state = PipelineState(text="John works at Google.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.ruler_entities) == 2
        assert result.ruler_entities[0]["name"] == "John"
        assert result.ruler_entities[0]["entity_type"] == "person"
        assert result.ruler_entities[1]["name"] == "Google"
        assert result.ruler_entities[1]["entity_type"] == "organization"

    def test_deduplication(self):
        """Duplicate entities (same name+type) are deduplicated."""
        nlp = _make_mock_nlp([
            {"text": "John", "label": "PERSON"},
            {"text": "John", "label": "PERSON"},
        ])
        stage = EntityRulerStage(nlp=nlp)
        state = PipelineState(text="John met John at the park.")
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.ruler_entities) == 1

    def test_label_mapping(self):
        """spaCy labels are mapped to SmartMemory entity types."""
        assert _map_label("PERSON") == "person"
        assert _map_label("ORG") == "organization"
        assert _map_label("GPE") == "location"
        assert _map_label("LOC") == "location"
        assert _map_label("DATE") == "temporal"
        assert _map_label("EVENT") == "event"
        assert _map_label("UNKNOWN_LABEL") == "concept"

    def test_uses_simplified_sentences(self):
        """When simplified_sentences is set, input is joined from them."""
        nlp = _make_mock_nlp([{"text": "Claude", "label": "PERSON"}])
        stage = EntityRulerStage(nlp=nlp)
        state = PipelineState(
            text="Original text.",
            simplified_sentences=["Claude is great.", "Claude codes."],
        )
        config = PipelineConfig()

        stage.execute(state, config)

        nlp.assert_called_once_with("Claude is great. Claude codes.")

    def test_min_confidence_filtering(self):
        """Entities below min_confidence are filtered out."""
        # spaCy NER assigns fixed 0.9 confidence — set threshold above that
        nlp = _make_mock_nlp([{"text": "John", "label": "PERSON"}])
        stage = EntityRulerStage(nlp=nlp)
        state = PipelineState(text="John works here.")
        config = PipelineConfig()
        config.extraction.entity_ruler = EntityRulerConfig(min_confidence=0.95)

        result = stage.execute(state, config)

        assert len(result.ruler_entities) == 0

    def test_undo_clears_ruler_entities(self):
        """Undo resets ruler_entities to empty."""
        stage = EntityRulerStage()
        state = PipelineState(
            text="Some text.",
            ruler_entities=[{"name": "John", "entity_type": "person"}],
        )

        result = stage.undo(state)

        assert result.ruler_entities == []
