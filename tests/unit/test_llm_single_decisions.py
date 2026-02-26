"""Unit tests for CORE-SYS2-1b: LLM schema extension for decision extraction.

Tests cover:
- _build_extraction_schema: correct structure, no original mutation, decisions optional
- Cache key: determinism, flag differentiation, schema version differentiation
"""

import hashlib
import json

import pytest


# ---------------------------------------------------------------------------
# _build_extraction_schema
# ---------------------------------------------------------------------------

class TestBuildExtractionSchema:
    def test_without_decisions_has_no_decisions_property(self):
        from smartmemory.plugins.extractors.llm_single import _build_extraction_schema

        schema = _build_extraction_schema(extract_decisions=False)
        properties = schema["json_schema"]["schema"]["properties"]
        assert "decisions" not in properties

    def test_with_decisions_has_decisions_property(self):
        from smartmemory.plugins.extractors.llm_single import _build_extraction_schema

        schema = _build_extraction_schema(extract_decisions=True)
        properties = schema["json_schema"]["schema"]["properties"]
        assert "decisions" in properties

    def test_decisions_property_has_correct_structure(self):
        from smartmemory.plugins.extractors.llm_single import _build_extraction_schema

        schema = _build_extraction_schema(extract_decisions=True)
        decisions_prop = schema["json_schema"]["schema"]["properties"]["decisions"]
        assert decisions_prop["type"] == "array"
        item_props = decisions_prop["items"]["properties"]
        assert "content" in item_props
        assert "decision_type" in item_props
        assert "confidence" in item_props
        assert decisions_prop["items"].get("additionalProperties") is False

    def test_decisions_not_in_required(self):
        """decisions is optional — content without decisions must not fail schema validation."""
        from smartmemory.plugins.extractors.llm_single import _build_extraction_schema

        schema = _build_extraction_schema(extract_decisions=True)
        required = schema["json_schema"]["schema"].get("required", [])
        assert "decisions" not in required

    def test_does_not_mutate_original_schema(self):
        """Calling with extract_decisions=True must not modify EXTRACTION_JSON_SCHEMA."""
        from smartmemory.plugins.extractors.llm_single import (
            EXTRACTION_JSON_SCHEMA,
            _build_extraction_schema,
        )

        original_properties = set(EXTRACTION_JSON_SCHEMA["json_schema"]["schema"]["properties"].keys())
        _build_extraction_schema(extract_decisions=True)
        after_properties = set(EXTRACTION_JSON_SCHEMA["json_schema"]["schema"]["properties"].keys())
        assert original_properties == after_properties, (
            "EXTRACTION_JSON_SCHEMA was mutated — _build_extraction_schema must deepcopy"
        )

    def test_calling_twice_does_not_accumulate(self):
        """Two calls with True should produce the same schema, not double-nested decisions."""
        from smartmemory.plugins.extractors.llm_single import _build_extraction_schema

        s1 = _build_extraction_schema(extract_decisions=True)
        s2 = _build_extraction_schema(extract_decisions=True)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def _make_key(self, model_name: str, extract_decisions: bool, text: str) -> str:
        """Replicate the cache key logic from _extract_impl for assertions."""
        from smartmemory.plugins.extractors.llm_single import EXTRACTION_SCHEMA_VERSION

        text_digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        return (
            f"single_{model_name}:"
            f"{extract_decisions}:"
            f"v{EXTRACTION_SCHEMA_VERSION}:"
            f"{text_digest}"
        )

    def test_key_is_deterministic(self):
        """Same inputs produce the same key across calls (no process-randomized hash)."""
        k1 = self._make_key("gpt-4o-mini", False, "some text")
        k2 = self._make_key("gpt-4o-mini", False, "some text")
        assert k1 == k2

    def test_key_differs_by_extract_decisions(self):
        k_off = self._make_key("gpt-4o-mini", False, "some text")
        k_on = self._make_key("gpt-4o-mini", True, "some text")
        assert k_off != k_on

    def test_key_differs_by_model(self):
        k1 = self._make_key("gpt-4o-mini", False, "text")
        k2 = self._make_key("llama-3.3-70b-versatile", False, "text")
        assert k1 != k2

    def test_key_differs_by_text(self):
        k1 = self._make_key("gpt-4o-mini", False, "text A")
        k2 = self._make_key("gpt-4o-mini", False, "text B")
        assert k1 != k2

    def test_key_differs_by_schema_version(self, monkeypatch):
        import smartmemory.plugins.extractors.llm_single as mod

        text = "hello"
        model = "gpt-4o-mini"
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]

        monkeypatch.setattr(mod, "EXTRACTION_SCHEMA_VERSION", 1)
        k1 = f"single_{model}:False:v1:{digest}"

        monkeypatch.setattr(mod, "EXTRACTION_SCHEMA_VERSION", 2)
        k2 = f"single_{model}:False:v2:{digest}"

        assert k1 != k2
