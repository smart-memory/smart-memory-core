"""Tests for smartmemory.tools.markdown_writer. No infra required."""
from unittest.mock import MagicMock


def _make_item(item_id="test-id", content="Alice leads Project Atlas", memory_type="semantic"):
    item = MagicMock()
    item.item_id = item_id
    item.content = content
    item.memory_type = memory_type
    item.created_at = "2026-01-01T00:00:00"
    item.entities = []
    item.relations = []
    return item


def test_render_produces_valid_yaml_frontmatter():
    """_render() output starts with YAML frontmatter block."""
    from smartmemory.tools.markdown_writer import _render
    item = _make_item()
    rendered = _render(item)
    assert rendered.startswith("---\n")
    assert "---\n\n" in rendered
    assert "item_id:" in rendered
    assert "memory_type:" in rendered
    assert "entities:" in rendered
    assert "relations:" in rendered


def test_render_body_contains_content():
    """_render() body (after frontmatter) contains the item's content verbatim."""
    from smartmemory.tools.markdown_writer import _render
    item = _make_item(content="Alice leads Project Atlas")
    rendered = _render(item)
    # Body is after the second ---
    body = rendered.split("---\n\n", 1)[-1]
    assert "Alice leads Project Atlas" in body


def test_write_markdown_creates_file(tmp_path):
    """write_markdown() creates a file at output_dir/<item_id>.md."""
    from smartmemory.tools.markdown_writer import write_markdown
    item = _make_item(item_id="abc-123")
    memory = MagicMock()
    memory.get.return_value = item
    path = write_markdown(memory, "abc-123", str(tmp_path))
    assert path == tmp_path / "abc-123.md"
    assert path.exists()


def test_write_markdown_frontmatter_has_entities_and_relations(tmp_path):
    """Written markdown file contains entities and relations keys."""
    from smartmemory.tools.markdown_writer import write_markdown
    item = _make_item()
    memory = MagicMock()
    memory.get.return_value = item
    path = write_markdown(memory, item.item_id, str(tmp_path))
    content = path.read_text(encoding="utf-8")
    assert "entities:" in content
    assert "relations:" in content
