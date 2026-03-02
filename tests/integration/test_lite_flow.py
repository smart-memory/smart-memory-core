"""Integration tests for SmartMemory Lite — no Docker, no external services required.

Run with:
    pip install -e ../smartmemory-lite   # from smart-memory-core/ directory
    PYTHONPATH=. pytest tests/integration/test_lite_flow.py -v
"""
import socket

import pytest

from smartmemory_lite import create_lite_memory
from smartmemory_lite.markdown_writer import write_markdown


@pytest.mark.integration
def test_lite_golden_flow(tmp_path):
    """Golden flow: add → get → search → markdown file written and readable."""
    memory = create_lite_memory(str(tmp_path))

    item_id = memory.ingest("Alice leads Project Atlas")

    # get() returns the stored item
    item = memory.get(item_id)
    assert item is not None, "get() must return the ingested item"
    assert "Alice" in item.content

    # search() surfaces the ingested item
    results = memory.search("who leads Atlas", top_k=3)
    assert any(r.item_id == item_id for r in results), (
        "search() must return the ingested item among top-3 results"
    )

    # write_markdown() creates a file with the item content
    notes_dir = tmp_path / "notes"
    md_path = write_markdown(memory, item_id, str(notes_dir))
    assert md_path.exists(), f"Markdown file not found at {md_path}"
    content = md_path.read_text(encoding="utf-8")
    assert "Alice" in content


@pytest.mark.integration
def test_lite_no_network(tmp_path):
    """Lite ingest must make zero network connections.

    create_lite_memory() may trigger import-time side effects from third-party
    packages (e.g. wandb/litellm telemetry). We track only the ingest() call
    itself, which is the operation that must be network-free.
    """
    memory = create_lite_memory(str(tmp_path))

    original_connect = socket.socket.connect
    connections: list = []

    def track_connect(self, addr):
        connections.append(addr)
        return original_connect(self, addr)

    socket.socket.connect = track_connect
    try:
        memory.ingest("test content, no network")
    finally:
        socket.socket.connect = original_connect

    assert connections == [], (
        f"Lite ingest made unexpected network calls: {connections}"
    )


@pytest.mark.integration
def test_lite_mcp_no_network(tmp_path):
    """LiteMCPHandler.ingest must make zero network connections.

    Handler construction (which calls create_lite_memory) may trigger import-time
    side effects from third-party packages. We track only the ingest() call itself.
    """
    from smartmemory_lite.mcp_server import LiteMCPHandler

    handler = LiteMCPHandler(data_dir=str(tmp_path))

    original_connect = socket.socket.connect
    connections: list = []

    def track_connect(self, addr):
        connections.append(addr)
        return original_connect(self, addr)

    socket.socket.connect = track_connect
    try:
        result = handler.ingest("test content via MCP")
    finally:
        socket.socket.connect = original_connect

    assert "error" not in result, f"MCP ingest returned error: {result}"
    assert connections == [], (
        f"LiteMCPHandler.ingest made unexpected network calls: {connections}"
    )
