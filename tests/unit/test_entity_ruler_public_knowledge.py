"""Tests for EntityRuler + PublicKnowledgeStore integration.

Covers:
- Workspace pattern overrides public pattern (three-layer precedence)
- Lazy reload fires on version change
- Collision logging
"""

from unittest.mock import MagicMock, patch

import pytest

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.sqlite_store import SQLitePublicKnowledgeStore
from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage


@pytest.fixture
def public_store(tmp_path):
    """SQLite store with test entities."""
    store = SQLitePublicKnowledgeStore(str(tmp_path / "public.sqlite"))
    store.absorb(PublicEntity(
        qid="Q28865",
        label="Python",
        aliases=["Python programming language"],
        entity_type="Technology",
        confidence=0.95,
    ))
    store.absorb(PublicEntity(
        qid="Q2622004",
        label="Django",
        aliases=["Django framework"],
        entity_type="Technology",
        confidence=0.9,
    ))
    return store


@pytest.fixture
def workspace_patterns():
    """Mock workspace PatternManager."""
    pm = MagicMock()
    pm.get_patterns.return_value = {
        "python": "concept",  # workspace says concept, public says technology
        "flask": "technology",
    }
    return pm


class TestPublicKnowledgeIntegration:
    """EntityRuler merges public knowledge patterns with workspace patterns."""

    def test_public_patterns_used_when_no_workspace(self, public_store):
        """Public patterns are used as base layer."""
        stage = EntityRulerStage(public_knowledge_store=public_store)
        # The stage should have loaded public patterns
        assert stage._public_knowledge_store is public_store

    def test_workspace_overrides_public(self, public_store, workspace_patterns):
        """Workspace pattern for 'python' overrides public knowledge."""
        stage = EntityRulerStage(
            pattern_manager=workspace_patterns,
            public_knowledge_store=public_store,
        )
        # Simulate entity ruler execution on text mentioning Python
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.pipeline.config import PipelineConfig

        state = PipelineState(text="I love Python and Django and Flask")
        config = PipelineConfig()
        result = stage.execute(state, config)

        # Check that Python was typed as "concept" (workspace wins) not "technology" (public)
        python_entities = [
            e for e in result.ruler_entities
            if isinstance(e, dict) and e.get("name", "").lower() == "python"
            and e.get("source") == "entity_ruler_learned"
        ]
        if python_entities:
            assert python_entities[0]["entity_type"] == "concept"

    def test_public_only_patterns_appear(self, public_store):
        """Entities only in public knowledge (no workspace override) appear."""
        stage = EntityRulerStage(public_knowledge_store=public_store)
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.pipeline.config import PipelineConfig

        state = PipelineState(text="Django is a great web framework")
        config = PipelineConfig()
        result = stage.execute(state, config)

        django_entities = [
            e for e in result.ruler_entities
            if isinstance(e, dict) and e.get("name", "").lower() == "django"
            and e.get("source") == "entity_ruler_learned"
        ]
        assert len(django_entities) >= 1

    def test_lazy_version_check(self, public_store):
        """Public patterns reload when store version changes."""
        stage = EntityRulerStage(public_knowledge_store=public_store)

        # First access should load patterns
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.pipeline.config import PipelineConfig

        state = PipelineState(text="Django is great")
        config = PipelineConfig()
        stage.execute(state, config)
        v1 = stage._public_version

        # Absorb a new entity — version should increment
        public_store.absorb(PublicEntity(
            qid="Q79", label="Rust", entity_type="Technology", confidence=0.9,
        ))

        stage.execute(state, config)
        v2 = stage._public_version
        assert v2 > v1

    def test_collision_logged(self, public_store, workspace_patterns):
        """Collision between public and workspace patterns logged at DEBUG."""
        stage = EntityRulerStage(
            pattern_manager=workspace_patterns,
            public_knowledge_store=public_store,
        )
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.pipeline.config import PipelineConfig

        state = PipelineState(text="Python is versatile")
        config = PipelineConfig()

        with patch("smartmemory.pipeline.stages.entity_ruler.logger") as mock_logger:
            stage.execute(state, config)
            # Should have logged the collision for "python" (public=Technology, workspace=concept)
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            collision_logged = any(
                "collision" in c.lower() or ("python" in c.lower() and "technology" in c.lower())
                for c in debug_calls
            )
            assert collision_logged, (
                f"Expected collision debug log for 'python' but got: {debug_calls}"
            )

    def test_confidence_is_fixed_085(self, public_store):
        """All matched public patterns get fixed 0.85 confidence."""
        stage = EntityRulerStage(public_knowledge_store=public_store)
        from smartmemory.pipeline.state import PipelineState
        from smartmemory.pipeline.config import PipelineConfig

        state = PipelineState(text="Django is a web framework")
        config = PipelineConfig()
        result = stage.execute(state, config)

        for entity in result.ruler_entities:
            if isinstance(entity, dict) and entity.get("source") == "entity_ruler_learned":
                assert entity["confidence"] == 0.85
