"""Tests for SmartGraph backend injection (P0-1 prerequisite for DIST-LITE-1)."""
from unittest.mock import MagicMock

from smartmemory.graph.smartgraph import SmartGraph


def test_backend_injection_bypasses_get_backend_class():
    """When backend= is provided, _get_backend_class() is never called."""
    mock_backend = MagicMock()
    graph = SmartGraph(backend=mock_backend)
    assert graph.backend is mock_backend


def test_backend_injection_skips_import():
    """Injected backend means no falkordb import attempt."""
    mock_backend = MagicMock()
    # Should not raise even though FalkorDB is not configured
    graph = SmartGraph(backend=mock_backend)
    assert graph.backend is mock_backend


def test_no_injection_uses_get_backend_class(monkeypatch):
    """When backend= is not provided, _get_backend_class() is called as before."""
    mock_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(SmartGraph, "_get_backend_class", staticmethod(lambda: mock_cls))
    graph = SmartGraph()
    assert graph.backend is mock_cls.return_value


def test_submodules_receive_injected_backend():
    """SmartGraphNodes/Edges/Search all receive the injected backend."""
    mock_backend = MagicMock()
    graph = SmartGraph(backend=mock_backend)
    assert graph.nodes.backend is mock_backend
    assert graph.edges.nodes.backend is mock_backend
    assert graph.search.backend is mock_backend
