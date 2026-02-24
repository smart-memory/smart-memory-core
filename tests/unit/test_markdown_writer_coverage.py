"""Additional coverage for smartmemory.tools.markdown_writer.

Focuses on:
- write_markdown raises ValueError when item not found
- _render with entities that have .label and .type attributes
- _render with relations that have .source, .relation, .target attributes
- _render with objects missing optional attrs (no entities/relations)
- _render with entities where iteration raises (exception suppression)
- write_markdown creates intermediate directories
- _render output is valid YAML parseable frontmatter
"""
from unittest.mock import MagicMock
import pytest


def _make_item(item_id="test-id", content="Test content", memory_type="semantic"):
    item = MagicMock()
    item.item_id = item_id
    item.content = content
    item.memory_type = memory_type
    item.created_at = "2026-01-01T00:00:00"
    item.entities = []
    item.relations = []
    return item


# ── write_markdown raises ValueError for missing item ─────────────────────────

def test_write_markdown_raises_for_missing_item(tmp_path):
    """write_markdown() raises ValueError when memory.get() returns None."""
    from smartmemory.tools.markdown_writer import write_markdown
    memory = MagicMock()
    memory.get.return_value = None
    with pytest.raises(ValueError, match="not found"):
        write_markdown(memory, "nonexistent-id", str(tmp_path))


def test_write_markdown_raises_contains_item_id(tmp_path):
    """ValueError message includes the missing item_id."""
    from smartmemory.tools.markdown_writer import write_markdown
    memory = MagicMock()
    memory.get.return_value = None
    with pytest.raises(ValueError, match="my-special-id"):
        write_markdown(memory, "my-special-id", str(tmp_path))


# ── _render with structured entities ─────────────────────────────────────────

def test_render_with_structured_entities():
    """_render includes entity label and type in YAML frontmatter."""
    from smartmemory.tools.markdown_writer import _render
    entity = MagicMock()
    entity.label = "Alice"
    entity.type = "PERSON"
    item = _make_item()
    item.entities = [entity]
    rendered = _render(item)
    assert "Alice" in rendered
    assert "PERSON" in rendered


def test_render_with_structured_relations():
    """_render includes relation source, relation, and target in YAML frontmatter."""
    from smartmemory.tools.markdown_writer import _render
    relation = MagicMock()
    relation.source = "Alice"
    relation.relation = "leads"
    relation.target = "Project Atlas"
    item = _make_item()
    item.relations = [relation]
    rendered = _render(item)
    assert "Alice" in rendered
    assert "leads" in rendered
    assert "Project Atlas" in rendered


# ── _render with missing optional attrs ──────────────────────────────────────

def test_render_item_without_entities_attr():
    """_render on an object with no entities attribute doesn't crash."""
    from smartmemory.tools.markdown_writer import _render

    class MinimalItem:
        item_id = "min-1"
        memory_type = "semantic"
        created_at = "2026-01-01"
        content = "minimal content"

    item = MinimalItem()
    rendered = _render(item)
    assert "minimal content" in rendered
    assert "item_id:" in rendered


def test_render_item_without_relations_attr():
    """_render on an object with no relations attribute doesn't crash."""
    from smartmemory.tools.markdown_writer import _render

    class NoRelationsItem:
        item_id = "nr-1"
        memory_type = "episodic"
        created_at = "2026-02-01"
        content = "no relations here"
        entities = []

    item = NoRelationsItem()
    rendered = _render(item)
    assert "no relations here" in rendered


# ── _render exceptions are suppressed ────────────────────────────────────────

def test_render_entity_iteration_raises_produces_fallback():
    """_render suppresses exceptions when iterating entities and falls back to str()."""
    from smartmemory.tools.markdown_writer import _render
    item = _make_item()
    # entities attribute exists but raises on iteration
    bad_iterable = MagicMock()
    bad_iterable.__bool__ = MagicMock(return_value=True)
    bad_iterable.__iter__ = MagicMock(side_effect=RuntimeError("boom"))
    item.entities = bad_iterable
    # Must not raise
    rendered = _render(item)
    assert "---\n" in rendered
    assert item.content in rendered


# ── write_markdown creates intermediate directories ───────────────────────────

def test_write_markdown_creates_nested_directories(tmp_path):
    """write_markdown() creates intermediate directories if they don't exist."""
    from smartmemory.tools.markdown_writer import write_markdown
    item = _make_item(item_id="nested-item")
    memory = MagicMock()
    memory.get.return_value = item
    nested_output = str(tmp_path / "level1" / "level2")
    path = write_markdown(memory, "nested-item", nested_output)
    assert path.exists()
    assert path.name == "nested-item.md"


# ── _render output is valid YAML ─────────────────────────────────────────────

def test_render_frontmatter_is_valid_yaml():
    """_render() produces valid YAML frontmatter that can be parsed by yaml.safe_load."""
    import yaml
    from smartmemory.tools.markdown_writer import _render
    entity = MagicMock()
    entity.label = "Bob"
    entity.type = "PERSON"
    item = _make_item(content="Bob works at Acme")
    item.entities = [entity]
    rendered = _render(item)
    # Extract frontmatter between first and second ---
    parts = rendered.split("---\n")
    assert len(parts) >= 3, "Expected at least 2 --- delimiters"
    fm_str = parts[1]
    parsed = yaml.safe_load(fm_str)
    assert isinstance(parsed, dict)
    assert "item_id" in parsed
    assert "memory_type" in parsed
    assert "entities" in parsed
    assert "relations" in parsed


# ── write_markdown file content matches _render ───────────────────────────────

def test_write_markdown_content_matches_render(tmp_path):
    """File written by write_markdown contains exactly what _render would produce."""
    from smartmemory.tools.markdown_writer import write_markdown, _render
    item = _make_item(item_id="match-test", content="Exact match content")
    memory = MagicMock()
    memory.get.return_value = item
    path = write_markdown(memory, "match-test", str(tmp_path))
    file_content = path.read_text(encoding="utf-8")
    expected = _render(item)
    assert file_content == expected
