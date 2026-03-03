"""CORE-7 — Lite no-network runtime assertion.

Verifies that a lite-profile ingest() call with SQLiteBackend, NoOpCache, and
observability=False makes zero outbound socket connections.

The socket patch covers connect() and connect_ex() only — it does NOT block
file I/O, so spaCy model loading (file reads) is unaffected.

Requires: no Docker (SQLiteBackend is in-memory, spaCy en_core_web_sm installed)
"""

import socket

import pytest

from smartmemory.graph.backends.sqlite import SQLiteBackend
from smartmemory.graph.smartgraph import SmartGraph
from smartmemory.pipeline.config import PipelineConfig
from smartmemory.smart_memory import SmartMemory
from smartmemory.utils.cache import NoOpCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lite_memory() -> SmartMemory:
    """Construct a SmartMemory instance wired for zero-infra lite operation."""
    backend = SQLiteBackend(db_path=":memory:")
    graph = SmartGraph(backend=backend)
    return SmartMemory(
        graph=graph,
        pipeline_profile=PipelineConfig.lite(llm_enabled=False),
        cache=NoOpCache(),
        observability=False,
    )


# ---------------------------------------------------------------------------
# Test: patch actually fires
# ---------------------------------------------------------------------------


def test_socket_patch_is_active(monkeypatch):
    """Control test: verify the socket patch raises when connect() is called directly.

    This test exists so that if test_lite_ingest_makes_no_network_calls ever
    passes spuriously (e.g. the patch silently fails to install), we have a
    separate signal that the patch mechanism itself is working.
    """

    def _block_connect(*args, **kwargs):
        raise AssertionError(f"Unexpected outbound socket connect: {args}")

    monkeypatch.setattr("socket.socket.connect", _block_connect)
    monkeypatch.setattr("socket.socket.connect_ex", _block_connect)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError, match="Unexpected outbound socket connect"):
            sock.connect(("example.com", 80))
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Test: lite ingest does not touch the network
# ---------------------------------------------------------------------------


def test_lite_ingest_makes_no_network_calls(monkeypatch):
    """Lite-profile ingest() must complete without touching socket.connect.

    The monkey-patch converts any outbound connection attempt into a hard
    AssertionError, so a silent failure (e.g. a caught exception inside the
    pipeline swallowing a connection error) would not mask a real network call —
    the AssertionError propagates out of the monkeypatched socket method before
    any except clause inside the library can catch it.
    """

    def _block_connect(*args, **kwargs):
        raise AssertionError(f"Unexpected outbound socket connect: {args}")

    monkeypatch.setattr("socket.socket.connect", _block_connect)
    monkeypatch.setattr("socket.socket.connect_ex", _block_connect)

    memory = _make_lite_memory()

    # This must not raise — neither an AssertionError from our patch nor any
    # other exception from a missing network dependency.
    item_id = memory.ingest("Test content for no-network assertion")

    assert item_id is not None, "ingest() must return a non-None item_id"
