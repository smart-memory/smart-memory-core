"""Unit tests for PatternManager and EntityRulerStage learned pattern scan."""

import pytest


pytestmark = pytest.mark.unit

from smartmemory.ontology.pattern_manager import PatternManager
from smartmemory.graph.ontology_graph import OntologyGraph
from smartmemory.pipeline.stages.entity_ruler import _ngram_scan

from tests.unit.pipeline_v2.test_ontology_graph_extended import ExtendedMockBackend


@pytest.fixture
def backend():
    return ExtendedMockBackend()


@pytest.fixture
def graph(backend):
    return OntologyGraph(workspace_id="test", backend=backend)


# ------------------------------------------------------------------ #
# PatternManager
# ------------------------------------------------------------------ #


def test_pattern_manager_loads_patterns(graph):
    graph.add_entity_pattern("python", "Technology", 0.9, workspace_id="test", initial_count=2)
    graph.add_entity_pattern("react", "Technology", 0.85, workspace_id="test", initial_count=2)

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "python" in patterns
    assert "react" in patterns
    assert patterns["python"] == "Technology"


def test_pattern_manager_filters_blocklist_words(graph):
    graph.add_entity_pattern("the", "Concept", 0.5, workspace_id="test", initial_count=2)
    graph.add_entity_pattern("python", "Technology", 0.9, workspace_id="test", initial_count=2)

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "the" not in patterns
    assert "python" in patterns


def test_pattern_manager_filters_short_names(graph):
    graph.add_entity_pattern("a", "Concept", 0.5, workspace_id="test", initial_count=2)
    graph.add_entity_pattern("qt", "Technology", 0.9, workspace_id="test", initial_count=2)

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "a" not in patterns
    assert "qt" in patterns  # length 2 passes


def test_pattern_manager_reload(graph):
    pm = PatternManager(graph, workspace_id="test")
    assert pm.pattern_count == 0

    graph.add_entity_pattern("docker", "Technology", 0.9, workspace_id="test", initial_count=2)
    pm.reload()

    assert pm.pattern_count == 1
    assert pm.version == 2  # initial load + reload


def test_pattern_manager_includes_global_patterns(graph):
    graph.add_entity_pattern("python", "Technology", 0.9, is_global=True, source="seed", initial_count=2)
    graph.add_entity_pattern("mylib", "Technology", 0.7, workspace_id="test", is_global=False, initial_count=2)

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "python" in patterns
    assert "mylib" in patterns


# ------------------------------------------------------------------ #
# add_patterns with source / initial_count (CODE-DEV-4)
# ------------------------------------------------------------------ #


def test_add_patterns_initial_count_2_survives_reload(graph):
    """Patterns added with initial_count=2 persist through a reload (count >= 2 threshold)."""
    pm = PatternManager(graph, workspace_id="test")
    pm.add_patterns({"fastapi": "Technology"}, source="code_index", initial_count=2)

    assert "fastapi" in pm.get_patterns()  # in-memory cache

    pm.reload()  # re-reads from backend via get_entity_patterns()
    assert "fastapi" in pm.get_patterns()  # survives reload


def test_add_patterns_default_count_lost_on_reload(graph):
    """Patterns added with default initial_count=1 are in cache but lost on reload."""
    pm = PatternManager(graph, workspace_id="test")
    pm.add_patterns({"flask": "Technology"})

    assert "flask" in pm.get_patterns()  # in-memory cache

    pm.reload()
    assert "flask" not in pm.get_patterns()  # count=1 < threshold, filtered out


def test_add_patterns_blocklist_rejected_regardless_of_initial_count(graph):
    """Blocklist words are rejected even with initial_count=2."""
    pm = PatternManager(graph, workspace_id="test")
    accepted = pm.add_patterns({"the": "Concept", "python": "Technology"}, initial_count=2)

    assert accepted == 1
    assert "the" not in pm.get_patterns()
    assert "python" in pm.get_patterns()


def test_add_patterns_custom_source(graph, backend):
    """Source parameter is passed through to ontology graph."""
    pm = PatternManager(graph, workspace_id="test")
    pm.add_patterns({"react": "Technology"}, source="code_index", initial_count=2)

    pattern = backend._patterns[("react", "Technology")]
    assert pattern["source"] == "code_index"


# ------------------------------------------------------------------ #
# _ngram_scan
# ------------------------------------------------------------------ #


def test_ngram_scan_finds_single_word():
    patterns = {"python": "Technology", "react": "Technology"}
    text = "I use Python for web development"
    matches = _ngram_scan(text, patterns)

    assert len(matches) == 1
    assert matches[0] == ("Python", "Technology")


def test_ngram_scan_finds_multi_word():
    patterns = {"machine learning": "Concept", "python": "Technology"}
    text = "We use machine learning with Python"
    matches = _ngram_scan(text, patterns)

    names = {m[0].lower() for m in matches}
    assert "machine learning" in names
    assert "python" in names


def test_ngram_scan_no_duplicate_subspan():
    """If 'machine learning' matches as 2-gram, 'machine' alone shouldn't match."""
    patterns = {"machine learning": "Concept", "machine": "Tool"}
    text = "We use machine learning every day"
    matches = _ngram_scan(text, patterns)

    # Should only get the 2-gram match, not the 1-gram
    assert len(matches) == 1
    assert matches[0][0].lower() == "machine learning"


def test_ngram_scan_empty_patterns():
    matches = _ngram_scan("some text here", {})
    assert matches == []


# ------------------------------------------------------------------ #
# EntityRulerStage with PatternManager
# ------------------------------------------------------------------ #


def test_entity_ruler_stage_uses_learned_patterns(graph):
    """EntityRulerStage should find entities from learned patterns."""
    from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage
    from smartmemory.pipeline.state import PipelineState
    from smartmemory.pipeline.config import PipelineConfig
    from unittest.mock import MagicMock

    graph.add_entity_pattern("fastapi", "Technology", 0.9, workspace_id="test", initial_count=2)
    pm = PatternManager(graph, workspace_id="test")

    # Mock spaCy to avoid requiring the model
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_doc.ents = []
    mock_nlp.return_value = mock_doc

    stage = EntityRulerStage(nlp=mock_nlp, pattern_manager=pm)
    state = PipelineState(text="We built our API with FastAPI and Python")
    config = PipelineConfig()

    result = stage.execute(state, config)

    learned_entities = [e for e in result.ruler_entities if e.get("source") == "entity_ruler_learned"]
    assert len(learned_entities) >= 1
    names = {e["name"].lower() for e in learned_entities}
    assert "fastapi" in names
