"""smartmemory.tools.markdown_writer — write MemoryItems to markdown files."""
from pathlib import Path


def _render(item) -> str:
    """Render a MemoryItem to a markdown string with YAML frontmatter."""
    import yaml

    entities = []
    relations = []
    try:
        if hasattr(item, "entities") and item.entities:
            entities = [{"label": e.label, "type": e.type} if hasattr(e, "label") else str(e)
                        for e in item.entities]
    except Exception:
        pass
    try:
        if hasattr(item, "relations") and item.relations:
            relations = [{"source": r.source, "relation": r.relation, "target": r.target}
                         if hasattr(r, "source") else str(r) for r in item.relations]
    except Exception:
        pass

    frontmatter = {
        "item_id": str(item.item_id) if hasattr(item, "item_id") else "",
        "memory_type": str(item.memory_type) if hasattr(item, "memory_type") else "semantic",
        "created_at": str(item.created_at) if hasattr(item, "created_at") else "",
        "entities": entities,
        "relations": relations,
    }
    fm_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    content = item.content if hasattr(item, "content") else str(item)
    return f"---\n{fm_str}---\n\n{content}\n"


def write_markdown(memory, item_id: str, output_dir: str) -> Path:
    """Retrieve item_id from memory and write it as a markdown file in output_dir."""
    item = memory.get(item_id)
    if item is None:
        raise ValueError(f"Item {item_id!r} not found in memory")
    output_path = Path(output_dir) / f"{item_id}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(item), encoding="utf-8")
    return output_path
