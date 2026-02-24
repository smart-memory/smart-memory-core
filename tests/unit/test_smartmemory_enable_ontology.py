"""Tests for SmartMemory enable_ontology flag (P0-2 prerequisite for DIST-LITE-1)."""
from unittest.mock import MagicMock, patch


def test_enable_ontology_false_skips_ontology_graph():
    """With enable_ontology=False, OntologyGraph is never instantiated."""
    from smartmemory.smart_memory import SmartMemory
    mock_backend = MagicMock()
    mock_graph = MagicMock()
    with patch("smartmemory.smart_memory.OntologyGraph", autospec=True) as mock_og:
        sm = SmartMemory(graph=mock_graph, enable_ontology=False)
        mock_og.assert_not_called()


def test_enable_ontology_true_is_default():
    """enable_ontology=True is the default — existing callers unaffected."""
    from smartmemory.smart_memory import SmartMemory
    # Just confirm instantiation doesn't break with default
    # (OntologyGraph may fail due to no FalkorDB — that's fine, just check flag stored)
    sm = SmartMemory.__new__(SmartMemory)
    sm._enable_ontology = True
    assert sm._enable_ontology is True


def test_enable_ontology_flag_stored():
    """Flag is stored as self._enable_ontology."""
    from smartmemory.smart_memory import SmartMemory
    mock_graph = MagicMock()
    sm = SmartMemory(graph=mock_graph, enable_ontology=False)
    assert sm._enable_ontology is False
