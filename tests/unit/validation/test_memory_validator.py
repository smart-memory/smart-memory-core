"""Unit tests for MemoryValidator and ValidationResult."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.models.memory_item import MemoryItem
from smartmemory.validation.memory_validator import (
    MemoryValidator,
    ValidationIssue,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------
class TestValidationIssue:
    def test_defaults(self):
        issue = ValidationIssue()
        assert issue.severity == "error"
        assert issue.field == ""
        assert issue.message == ""

    def test_to_dict(self):
        issue = ValidationIssue(severity="warning", field="content", message="too short")
        d = issue.to_dict()
        assert d == {"severity": "warning", "field": "content", "message": "too short"}

    def test_from_dict(self):
        d = {"severity": "info", "field": "x", "message": "note"}
        issue = ValidationIssue.from_dict(d)
        assert issue.severity == "info"
        assert issue.field == "x"

    def test_from_dict_defaults(self):
        issue = ValidationIssue.from_dict({})
        assert issue.severity == "error"


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------
class TestValidationResult:
    def test_empty_is_valid(self):
        r = ValidationResult()
        assert r.is_valid is True
        assert r.errors == []
        assert r.warnings == []

    def test_with_warning_still_valid(self):
        r = ValidationResult(issues=[ValidationIssue(severity="warning", message="w")])
        assert r.is_valid is True
        assert len(r.warnings) == 1

    def test_with_error_not_valid(self):
        r = ValidationResult(issues=[ValidationIssue(severity="error", message="e")])
        assert r.is_valid is False
        assert len(r.errors) == 1

    def test_to_dict(self):
        r = ValidationResult(issues=[ValidationIssue(severity="error", field="f", message="m")])
        d = r.to_dict()
        assert d["is_valid"] is False
        assert len(d["issues"]) == 1

    def test_from_dict(self):
        d = {"issues": [{"severity": "warning", "field": "x", "message": "y"}]}
        r = ValidationResult.from_dict(d)
        assert r.is_valid is True
        assert len(r.warnings) == 1


# ---------------------------------------------------------------------------
# MemoryValidator
# ---------------------------------------------------------------------------
class TestMemoryValidator:
    @pytest.fixture
    def validator(self):
        memory = MagicMock()
        memory._graph = MagicMock()
        return MemoryValidator(memory)

    def test_valid_semantic_item(self, validator):
        item = MemoryItem(content="Python is a programming language", memory_type="semantic")
        result = validator.validate_item(item)
        assert result.is_valid is True

    def test_empty_content_is_error(self, validator):
        item = MemoryItem(content="", memory_type="semantic")
        result = validator.validate_item(item)
        assert result.is_valid is False
        assert any(i.field == "content" for i in result.errors)

    def test_whitespace_only_content_is_error(self, validator):
        item = MemoryItem(content="   ", memory_type="semantic")
        result = validator.validate_item(item)
        assert result.is_valid is False

    def test_unknown_memory_type_is_warning(self, validator):
        item = MemoryItem(content="test", memory_type="unknown_type_xyz")
        result = validator.validate_item(item)
        # Unknown type is a warning, not error
        assert result.is_valid is True
        assert any(i.field == "memory_type" for i in result.warnings)

    def test_confidence_out_of_range_is_error(self, validator):
        item = MemoryItem(content="test", memory_type="semantic", metadata={"confidence": 1.5})
        result = validator.validate_item(item)
        assert result.is_valid is False
        assert any(i.field == "confidence" for i in result.errors)

    def test_confidence_negative_is_error(self, validator):
        item = MemoryItem(content="test", memory_type="semantic", metadata={"confidence": -0.1})
        result = validator.validate_item(item)
        assert result.is_valid is False

    def test_confidence_valid_range(self, validator):
        for c in [0.0, 0.5, 1.0]:
            item = MemoryItem(content="test", memory_type="semantic", metadata={"confidence": c})
            result = validator.validate_item(item)
            assert result.is_valid is True, f"confidence={c} should be valid"

    def test_decision_missing_decision_id_is_warning(self, validator):
        item = MemoryItem(content="User prefers Python", memory_type="decision")
        result = validator.validate_item(item)
        assert any(i.field == "decision_id" and i.severity == "warning" for i in result.issues)

    def test_decision_with_decision_id_ok(self, validator):
        item = MemoryItem(
            content="User prefers Python",
            memory_type="decision",
            metadata={"decision_id": "dec_123"},
        )
        result = validator.validate_item(item)
        assert not any(i.field == "decision_id" for i in result.issues)

    def test_reasoning_missing_trace_id_is_warning(self, validator):
        item = MemoryItem(content="Because X implies Y", memory_type="reasoning")
        result = validator.validate_item(item)
        assert any(i.field == "trace_id" and i.severity == "warning" for i in result.issues)

    def test_reasoning_with_trace_id_ok(self, validator):
        item = MemoryItem(
            content="Because X implies Y",
            memory_type="reasoning",
            metadata={"trace_id": "trace_abc"},
        )
        result = validator.validate_item(item)
        assert not any(i.field == "trace_id" for i in result.issues)
