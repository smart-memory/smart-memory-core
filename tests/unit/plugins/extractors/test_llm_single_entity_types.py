"""
Unit tests for EXTRACTION_JSON_SCHEMA entity_type enum constraint (CROSS-API-1).

Validates that:
- The JSON schema forces structured-output-capable LLMs to a known type set
- The enum exactly matches the types documented in SINGLE_CALL_PROMPT
- The schema version has been incremented to bust cached results (CORE-SYS2-1b)
"""

import re

import pytest

pytestmark = pytest.mark.unit

from smartmemory.plugins.extractors.llm_single import (
    EXTRACTION_JSON_SCHEMA,
    EXTRACTION_SCHEMA_VERSION,
    SINGLE_CALL_PROMPT,
    VALID_ENTITY_TYPES,
    LLMSingleExtractor,
)

# The canonical list of valid entity types, shared by SINGLE_CALL_PROMPT and the JSON schema.
EXPECTED_ENTITY_TYPES = frozenset(
    [
        "person",
        "organization",
        "location",
        "event",
        "product",
        "work_of_art",
        "temporal",
        "concept",
        "technology",
        "award",
    ]
)


class TestExtractionJsonSchemaEntityTypeEnum:
    """The entity_type field in EXTRACTION_JSON_SCHEMA must carry an enum constraint."""

    def _entity_type_schema(self) -> dict:
        """Navigate to the entity_type property dict inside the schema."""
        return (
            EXTRACTION_JSON_SCHEMA["json_schema"]["schema"]["properties"]["entities"][
                "items"
            ]["properties"]["entity_type"]
        )

    def test_entity_type_has_enum_key(self):
        """entity_type schema must include an 'enum' key."""
        prop = self._entity_type_schema()
        assert "enum" in prop, (
            "entity_type in EXTRACTION_JSON_SCHEMA is missing 'enum'. "
            "Structured-output LLMs will accept any string without it."
        )

    def test_entity_type_enum_contains_exactly_10_types(self):
        """The enum must list exactly the 10 canonical entity types."""
        prop = self._entity_type_schema()
        enum_values = prop["enum"]
        assert len(enum_values) == 10, (
            f"Expected 10 entity types in enum, got {len(enum_values)}: {enum_values}"
        )

    def test_entity_type_enum_values_match_expected_set(self):
        """Every enum value must be in the expected canonical set, and none may be missing."""
        prop = self._entity_type_schema()
        enum_set = set(prop["enum"])
        assert enum_set == EXPECTED_ENTITY_TYPES, (
            f"Enum mismatch.\n"
            f"  Extra (in schema, not expected): {enum_set - EXPECTED_ENTITY_TYPES}\n"
            f"  Missing (expected, not in schema): {EXPECTED_ENTITY_TYPES - enum_set}"
        )

    def test_entity_type_enum_values_are_lowercase_strings(self):
        """All enum values must be lowercase strings (no casing drift)."""
        prop = self._entity_type_schema()
        for value in prop["enum"]:
            assert isinstance(value, str), f"Enum value {value!r} is not a string"
            assert value == value.lower(), (
                f"Enum value {value!r} is not lowercase — entity processing lowercases "
                "incoming types, so the enum must match."
            )

    def test_entity_type_field_is_still_type_string(self):
        """The type field must remain 'string' alongside the enum."""
        prop = self._entity_type_schema()
        assert prop.get("type") == "string", (
            "entity_type schema 'type' must remain 'string' (enum alone is not sufficient "
            "for all validators)."
        )


class TestExtractionSchemaVersionIncremented:
    """EXTRACTION_SCHEMA_VERSION must be >= 2 after the enum constraint was added."""

    def test_schema_version_is_at_least_2(self):
        """Schema version must have been incremented to bust cached results (CORE-SYS2-1b)."""
        assert EXTRACTION_SCHEMA_VERSION >= 2, (
            f"EXTRACTION_SCHEMA_VERSION is {EXTRACTION_SCHEMA_VERSION}. "
            "It must be incremented when EXTRACTION_JSON_SCHEMA changes shape "
            "so that cached extraction results are invalidated (CORE-SYS2-1b)."
        )


class TestEnumMatchesSingleCallPrompt:
    """The enum in the JSON schema must match the types listed in SINGLE_CALL_PROMPT."""

    def _extract_types_from_prompt(self) -> frozenset:
        """
        Parse the ENTITY TYPES bullet list from SINGLE_CALL_PROMPT.

        Each bullet has the form:  - <type>: <description>
        """
        types = set()
        for line in SINGLE_CALL_PROMPT.splitlines():
            match = re.match(r"^\s*-\s+([a-z_]+):", line)
            if match:
                types.add(match.group(1))
        return frozenset(types)

    def test_prompt_types_match_schema_enum(self):
        """Types declared in SINGLE_CALL_PROMPT and EXTRACTION_JSON_SCHEMA enum must be identical."""
        prompt_types = self._extract_types_from_prompt()
        schema_prop = (
            EXTRACTION_JSON_SCHEMA["json_schema"]["schema"]["properties"]["entities"][
                "items"
            ]["properties"]["entity_type"]
        )
        schema_enum = frozenset(schema_prop["enum"])

        assert prompt_types == schema_enum, (
            f"Mismatch between prompt and schema enum.\n"
            f"  In prompt only: {prompt_types - schema_enum}\n"
            f"  In schema only: {schema_enum - prompt_types}\n"
            "The prompt and the structured-output schema must declare the same type list."
        )

    def test_prompt_contains_all_expected_types(self):
        """SINGLE_CALL_PROMPT must document every type in EXPECTED_ENTITY_TYPES."""
        prompt_types = self._extract_types_from_prompt()
        assert prompt_types == EXPECTED_ENTITY_TYPES, (
            f"Prompt type list does not match EXPECTED_ENTITY_TYPES.\n"
            f"  Extra in prompt: {prompt_types - EXPECTED_ENTITY_TYPES}\n"
            f"  Missing from prompt: {EXPECTED_ENTITY_TYPES - prompt_types}"
        )


class TestEntityTypeFallbackNormalization:
    """CROSS-API-2: _process_entities() normalizes out-of-enum entity_type values.

    LLM providers (Groq, Ollama, non-strict JSON mode) may ignore the schema
    enum constraint and return arbitrary type strings. The normalization gate
    in _process_entities() must remap these to "concept" before they enter the graph.
    """

    @pytest.fixture
    def extractor(self):
        """Extractor instance with a dummy API key (no LLM calls made in these tests)."""
        return LLMSingleExtractor(api_key="test-key-unused")

    def test_valid_entity_types_constant_matches_schema_enum(self):
        """VALID_ENTITY_TYPES must equal the enum from EXTRACTION_JSON_SCHEMA."""
        schema_enum = frozenset(
            EXTRACTION_JSON_SCHEMA["json_schema"]["schema"]["properties"]["entities"][
                "items"
            ]["properties"]["entity_type"]["enum"]
        )
        assert VALID_ENTITY_TYPES == schema_enum, (
            "VALID_ENTITY_TYPES has drifted from EXTRACTION_JSON_SCHEMA enum.\n"
            f"  Extra in constant: {VALID_ENTITY_TYPES - schema_enum}\n"
            f"  Missing from constant: {schema_enum - VALID_ENTITY_TYPES}"
        )

    def test_unknown_type_normalized_to_concept(self, extractor):
        """An out-of-enum type returned by the LLM is remapped to 'concept'."""
        raw = [{"name": "Pizza", "entity_type": "food", "confidence": 0.9}]
        result = extractor._process_entities(raw)

        assert len(result) == 1
        assert result[0].metadata["entity_type"] == "concept", (
            "entity_type 'food' is not in the enum and must be normalized to 'concept'"
        )

    def test_multiple_unknown_types_all_normalized(self, extractor):
        """Each out-of-enum type in a batch is individually remapped."""
        raw = [
            {"name": "Pizza", "entity_type": "food", "confidence": 0.9},
            {"name": "Dog", "entity_type": "animal", "confidence": 0.85},
            {"name": "Happiness", "entity_type": "emotion", "confidence": 0.7},
        ]
        result = extractor._process_entities(raw)

        assert len(result) == 3
        for item in result:
            assert item.metadata["entity_type"] == "concept", (
                f"Expected 'concept' for {item.content!r}, got {item.metadata['entity_type']!r}"
            )

    def test_valid_types_pass_through_unchanged(self, extractor):
        """Known enum types are not remapped."""
        for valid_type in ["person", "organization", "location", "technology", "award"]:
            raw = [{"name": "Test", "entity_type": valid_type, "confidence": 0.9}]
            result = extractor._process_entities(raw)
            assert result[0].metadata["entity_type"] == valid_type, (
                f"Valid type {valid_type!r} should not be remapped"
            )

    def test_uppercase_type_lowercased_then_validated(self, extractor):
        """Types are lowercased before validation — 'PERSON' normalizes to valid 'person'."""
        raw = [{"name": "Alice", "entity_type": "PERSON", "confidence": 0.95}]
        result = extractor._process_entities(raw)
        assert result[0].metadata["entity_type"] == "person"

    def test_mixed_valid_and_invalid_in_same_batch(self, extractor):
        """Valid and invalid types in the same LLM response are handled correctly."""
        raw = [
            {"name": "Alice", "entity_type": "person", "confidence": 0.95},
            {"name": "Mars", "entity_type": "planet", "confidence": 0.8},   # out-of-enum
            {"name": "Python", "entity_type": "technology", "confidence": 0.9},
        ]
        result = extractor._process_entities(raw)
        by_name = {item.content: item.metadata["entity_type"] for item in result}

        assert by_name["Alice"] == "person"
        assert by_name["Mars"] == "concept"   # "planet" → fallback
        assert by_name["Python"] == "technology"
