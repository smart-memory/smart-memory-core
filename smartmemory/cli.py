"""smartmemory.cli — command-line interface for SmartMemory Lite."""
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from smartmemory.tools.factory import create_lite_memory
from smartmemory.tools.markdown_writer import write_markdown

console = Console()


@click.group()
@click.option("--data-dir", default=None, help="Data directory (default: ~/.smartmemory)")
@click.pass_context
def main(ctx, data_dir):
    """SmartMemory Lite — zero-infra persistent memory for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["memory"] = None  # lazy init


def _get_memory(ctx):
    if ctx.obj["memory"] is None:
        ctx.obj["memory"] = create_lite_memory(ctx.obj["data_dir"])
    return ctx.obj["memory"]


@main.command()
@click.argument("text")
@click.option("--no-markdown", is_flag=True, default=False, help="Skip markdown file creation")
@click.pass_context
def add(ctx, text, no_markdown):
    """Add a memory. Extracts entities and stores to local database."""
    memory = _get_memory(ctx)
    item_id = memory.ingest(text)
    console.print(f"[green]Stored:[/green] {item_id}")
    if not no_markdown:
        data_dir = ctx.obj["data_dir"] or str(Path.home() / ".smartmemory")
        notes_dir = Path(data_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        path = write_markdown(memory, item_id, str(notes_dir))
        console.print(f"[dim]Written:[/dim] {path}")


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
@click.pass_context
def search(ctx, query, top_k, as_json):
    """Search memories by semantic similarity."""
    memory = _get_memory(ctx)
    results = memory.search(query, top_k=top_k)
    if as_json:
        click.echo(json.dumps([
            {"item_id": r.item_id if hasattr(r, "item_id") else str(r),
             "content": r.content if hasattr(r, "content") else str(r)}
            for r in results
        ], indent=2))
        return
    table = Table(title=f"Results for: {query}")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Content")
    for r in results:
        item_id = r.item_id if hasattr(r, "item_id") else str(r)
        content = r.content if hasattr(r, "content") else str(r)
        table.add_row(item_id[:12], content[:120])
    console.print(table)


@main.command()
@click.pass_context
def rebuild(ctx):
    """Rebuild the vector index from existing graph data.

    This is a pure vector reindex — graph nodes are read but never written.
    It calls create_embeddings() + VectorStore().upsert() directly, the same
    two-step path that SmartMemory.add() uses internally after the graph write,
    but without the graph write itself. All node metadata (source, label, custom
    fields, etc.) is preserved from the serialized graph snapshot.
    """
    from smartmemory.plugins.embedding import create_embeddings
    from smartmemory.stores.vector.vector_store import VectorStore
    memory = _get_memory(ctx)
    data = memory._graph.backend.serialize()
    VectorStore().clear()
    count = 0
    skipped = 0
    for node in data.get("nodes", []):
        node_id = node.get("item_id") or node.get("id", "")
        # Properties dict holds all metadata stored alongside the node
        properties = node.get("properties") or {}
        content = properties.get("content") or node.get("content", "")
        if not (content and node_id):
            skipped += 1
            continue
        try:
            embedding = create_embeddings(content)
            if embedding is not None:
                VectorStore().upsert(
                    item_id=node_id,
                    embedding=embedding,
                    metadata=properties,
                    node_ids=[node_id],
                    is_global=False,
                )
                count += 1
            else:
                skipped += 1
        except Exception as exc:
            console.print(f"[yellow]Skipped {node_id}:[/yellow] {exc}")
            skipped += 1
    console.print(f"[green]Rebuilt:[/green] {count} items re-indexed", end="")
    if skipped:
        console.print(f" ([dim]{skipped} skipped[/dim])")
    else:
        console.print()


@main.command()
@click.argument("vault_path", type=click.Path(exists=True))
@click.pass_context
def watch(ctx, vault_path):
    """Watch a directory and auto-ingest new/changed markdown files."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        console.print("[red]watchdog not installed.[/red] Run: pip install smartmemory-core[watch]")
        raise SystemExit(1)

    memory = _get_memory(ctx)

    class MarkdownHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".md"):
                self._ingest(event.src_path)

        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(".md"):
                self._ingest(event.src_path)

        def _ingest(self, path):
            try:
                content = Path(path).read_text(encoding="utf-8")
                item_id = memory.ingest(content)
                console.print(f"[green]Ingested:[/green] {path} → {item_id}")
            except Exception as e:
                console.print(f"[red]Error ingesting {path}:[/red] {e}")

    observer = Observer()
    observer.schedule(MarkdownHandler(), vault_path, recursive=True)
    observer.start()
    console.print(f"[green]Watching:[/green] {vault_path} (Ctrl+C to stop)")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
