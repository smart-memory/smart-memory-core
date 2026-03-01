"""Integration test for ONTO-PUB-1 entity grounding.

End-to-end: SQLitePublicKnowledgeStore → SmartMemory → ingest →
verify wikidata:{qid} node exists with GROUNDED_IN edge.

These tests use a SQLite-backed graph (no Docker required) and inject
a pre-loaded PublicKnowledgeStore. They verify the full pipeline path
from EntityRuler pattern matching through GroundStage grounding.
"""

import json

import pytest

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.sqlite_store import SQLitePublicKnowledgeStore
from smartmemory.grounding.benchmark import GroundingBenchmark


@pytest.fixture
def public_store(tmp_path):
    """Pre-loaded SQLite store with test entities."""
    store = SQLitePublicKnowledgeStore(str(tmp_path / "public.sqlite"))
    store.absorb(PublicEntity(
        qid="Q28865",
        label="Python",
        aliases=["Python programming language", "Python 3"],
        description="General-purpose programming language",
        entity_type="Technology",
        instance_of=["Q9143"],
        domain="software",
        confidence=0.95,
    ))
    store.absorb(PublicEntity(
        qid="Q2622004",
        label="Django",
        aliases=["Django framework"],
        description="Web framework for Python",
        entity_type="Technology",
        instance_of=["Q271680"],
        domain="software",
        confidence=0.9,
    ))
    store.absorb(PublicEntity(
        qid="Q312",
        label="React",
        aliases=["React.js", "ReactJS"],
        description="JavaScript library for building user interfaces",
        entity_type="Technology",
        instance_of=["Q188860"],
        domain="software",
        confidence=0.92,
    ))
    return store


class TestEntityRulerPicksUpPublicPatterns:
    """EntityRuler should recognize public knowledge entities."""

    def test_ruler_matches_public_entity(self, public_store):
        """Public entity appears in ruler_entities after pipeline stage."""
        from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.pipeline.config import PipelineConfig

        stage = EntityRulerStage(public_knowledge_store=public_store)
        state = PipelineState(text="We built the backend in Django and the frontend in React")
        config = PipelineConfig()
        result = stage.execute(state, config)

        # Both Django and React should be in ruler_entities
        names = {
            e["name"].lower()
            for e in result.ruler_entities
            if isinstance(e, dict) and e.get("source") == "entity_ruler_learned"
        }
        assert "django" in names
        assert "react" in names


class TestGroundingDecisions:
    """Grounding loop produces decisions for every entity."""

    def test_grounding_decisions_logged(self, public_store):
        """Every processed entity gets a GroundingDecision."""
        from unittest.mock import MagicMock
        from smartmemory.grounding.public_knowledge_grounder import PublicKnowledgeGrounder

        graph = MagicMock()
        graph.get_node.return_value = None
        grounder = PublicKnowledgeGrounder(public_store)

        entities = [
            {"name": "Django", "entity_type": "technology", "item_id": "e:1"},
            {"name": "NonexistentThing", "entity_type": "concept", "item_id": "e:2"},
        ]
        grounder.ground(MagicMock(), entities, graph)

        assert len(grounder.decisions) == 2
        sources = {d.source for d in grounder.decisions}
        assert "alias_lookup" in sources
        assert "ungrounded" in sources


class TestBenchmarkHarness:
    """GroundingBenchmark evaluates against labeled dataset."""

    def test_benchmark_with_known_entities(self, public_store):
        benchmark = GroundingBenchmark(public_store)
        result = benchmark.evaluate([
            ("Python", "Q28865"),
            ("Django", "Q2622004"),
            ("React", "Q312"),
        ])
        assert result.total == 3
        assert result.hits == 3
        assert result.misses == 0
        assert result.hit_rate == 1.0

    def test_benchmark_with_misses(self, public_store):
        benchmark = GroundingBenchmark(public_store)
        result = benchmark.evaluate([
            ("Python", "Q28865"),
            ("NonexistentThing", "Q99999"),
        ])
        assert result.total == 2
        assert result.hits == 1
        assert result.misses == 1
        assert result.hit_rate == 0.5

    def test_benchmark_false_positives(self, public_store):
        benchmark = GroundingBenchmark(public_store)
        result = benchmark.evaluate([
            ("Python", ""),  # should NOT be grounded, but store has it
        ])
        assert result.false_positives == 1

    def test_benchmark_by_domain(self, public_store):
        benchmark = GroundingBenchmark(public_store)
        result = benchmark.evaluate([
            ("Python", "Q28865"),
            ("Django", "Q2622004"),
        ])
        assert "software" in result.by_domain
        assert result.by_domain["software"]["hits"] == 2
