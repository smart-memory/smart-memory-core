"""Tests for TemporalEnricher — focused on malformed LLM response handling (CORE-14)."""

import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from smartmemory.plugins.enrichers.temporal import TemporalEnricher, TemporalEnricherConfig


@dataclass
class FakeItem:
    content: str = "Alice joined Acme Corp in January 2025"
    item_id: str = "test-item-1"


def _make_response(content: Optional[str]) -> MagicMock:
    """Build a fake OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


@pytest.fixture
def enricher():
    """TemporalEnricher with mocked openai."""
    e = TemporalEnricher(TemporalEnricherConfig())
    e._openai = MagicMock()
    return e


class TestTemporalEnricherMalformedResponses:
    """CORE-14: TemporalEnricher must degrade gracefully on bad LLM output."""

    def test_valid_json(self, enricher):
        payload = {"temporal_references": [{"event": "joined", "date": "2025-01"}]}
        enricher._openai.chat.completions.create.return_value = _make_response(json.dumps(payload))
        result = enricher.enrich(FakeItem())
        assert result["temporal"] == payload

    def test_empty_content(self, enricher):
        enricher._openai.chat.completions.create.return_value = _make_response("")
        result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}

    def test_none_content(self, enricher):
        enricher._openai.chat.completions.create.return_value = _make_response(None)
        result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}

    def test_truncated_json(self, enricher):
        enricher._openai.chat.completions.create.return_value = _make_response('{"temporal_ref')
        result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}

    def test_markdown_wrapped_json(self, enricher):
        payload = {"event": "joined", "date": "2025-01"}
        wrapped = f"```json\n{json.dumps(payload)}\n```"
        enricher._openai.chat.completions.create.return_value = _make_response(wrapped)
        result = enricher.enrich(FakeItem())
        assert result["temporal"] == payload

    def test_json_with_preamble(self, enricher):
        payload = {"event": "joined"}
        text = f"Here is the analysis:\n{json.dumps(payload)}\nDone."
        enricher._openai.chat.completions.create.return_value = _make_response(text)
        result = enricher.enrich(FakeItem())
        assert result["temporal"] == payload

    def test_plain_text_no_json(self, enricher):
        enricher._openai.chat.completions.create.return_value = _make_response(
            "I cannot determine temporal information from this text."
        )
        result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}

    def test_api_exception(self, enricher):
        enricher._openai.chat.completions.create.side_effect = RuntimeError("rate limited")
        result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}

    def test_array_response_degrades(self, enricher):
        """parse_json_response returns None for arrays; enricher should return empty dict."""
        enricher._openai.chat.completions.create.return_value = _make_response('[1, 2, 3]')
        result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}

    def test_no_traceback_on_malformed(self, enricher, caplog):
        """CORE-14: malformed JSON must NOT produce ERROR-level traceback."""
        enricher._openai.chat.completions.create.return_value = _make_response('{"broken')
        with caplog.at_level("WARNING"):
            result = enricher.enrich(FakeItem())
        assert result == {"temporal": {}}
        # Should not have ERROR-level log entries from the enricher
        error_records = [r for r in caplog.records if r.levelname == "ERROR" and "TemporalEnricher" in r.getMessage()]
        assert len(error_records) == 0
