"""Tests for SQLitePublicKnowledgeStore.

Covers CRUD round-trip, ruler pattern generation, ambiguity resolution
(deterministic winner), and the label-as-alias invariant.
"""

import json
import os
import tempfile

import pytest

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.sqlite_store import SQLitePublicKnowledgeStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh SQLite store for each test."""
    db_path = str(tmp_path / "test_public.sqlite")
    return SQLitePublicKnowledgeStore(db_path)


@pytest.fixture
def python_entity():
    return PublicEntity(
        qid="Q28865",
        label="Python",
        aliases=["Python programming language", "Python 3"],
        description="General-purpose programming language",
        entity_type="Technology",
        instance_of=["Q9143"],
        domain="software",
        confidence=0.95,
    )


@pytest.fixture
def python_snake_entity():
    return PublicEntity(
        qid="Q190391",
        label="Python",
        aliases=["Pythonidae"],
        description="Family of nonvenomous snakes",
        entity_type="Concept",
        instance_of=["Q16521"],
        domain="biology",
        confidence=0.85,
    )


class TestAbsorb:
    """absorb() stores entities and increments version."""

    def test_absorb_increments_version(self, store, python_entity):
        v0 = store.version
        store.absorb(python_entity)
        assert store.version == v0 + 1

    def test_absorb_multiple_increments(self, store, python_entity, python_snake_entity):
        store.absorb(python_entity)
        store.absorb(python_snake_entity)
        assert store.version == 2

    def test_count_after_absorb(self, store, python_entity):
        assert store.count() == 0
        store.absorb(python_entity)
        assert store.count() == 1

    def test_absorb_upsert(self, store, python_entity):
        """Absorbing same QID again updates, doesn't duplicate."""
        store.absorb(python_entity)
        python_entity.description = "Updated description"
        store.absorb(python_entity)
        assert store.count() == 1
        result = store.lookup_by_qid("Q28865")
        assert result.description == "Updated description"


class TestLookup:
    """lookup_by_alias and lookup_by_qid."""

    def test_lookup_by_qid(self, store, python_entity):
        store.absorb(python_entity)
        result = store.lookup_by_qid("Q28865")
        assert result is not None
        assert result.qid == "Q28865"
        assert result.label == "Python"
        assert result.entity_type == "Technology"

    def test_lookup_by_qid_not_found(self, store):
        assert store.lookup_by_qid("Q99999999") is None

    def test_lookup_by_alias_explicit(self, store, python_entity):
        store.absorb(python_entity)
        results = store.lookup_by_alias("Python programming language")
        assert len(results) == 1
        assert results[0].qid == "Q28865"

    def test_lookup_by_alias_case_insensitive(self, store, python_entity):
        store.absorb(python_entity)
        results = store.lookup_by_alias("python programming language")
        assert len(results) == 1
        assert results[0].qid == "Q28865"

    def test_label_as_alias_invariant(self, store, python_entity):
        """Primary label must be findable via lookup_by_alias."""
        store.absorb(python_entity)
        results = store.lookup_by_alias("Python")
        assert len(results) >= 1
        qids = {r.qid for r in results}
        assert "Q28865" in qids

    def test_ambiguous_alias(self, store, python_entity, python_snake_entity):
        """'Python' matches both the language and the snake."""
        store.absorb(python_entity)
        store.absorb(python_snake_entity)
        results = store.lookup_by_alias("Python")
        assert len(results) == 2
        qids = {r.qid for r in results}
        assert qids == {"Q28865", "Q190391"}

    def test_lookup_no_match(self, store):
        assert store.lookup_by_alias("nonexistent") == []


class TestRulerPatterns:
    """get_ruler_patterns() returns deterministic surface_form → type mapping."""

    def test_single_entity(self, store, python_entity):
        store.absorb(python_entity)
        patterns = store.get_ruler_patterns()
        # "Python" and "Python programming language" and "Python 3"
        assert "python" in patterns or "Python" in patterns

    def test_ambiguity_deterministic_winner(self, store, python_entity, python_snake_entity):
        """Two entities share 'Python' — highest confidence wins."""
        store.absorb(python_entity)   # confidence 0.95
        store.absorb(python_snake_entity)  # confidence 0.85
        patterns = store.get_ruler_patterns()
        # "Python" should map to Technology (higher confidence)
        python_type = None
        for sf, t in patterns.items():
            if sf.lower() == "python":
                python_type = t
                break
        assert python_type is not None
        assert python_type == "Technology"

    def test_ambiguity_tiebreak_by_qid(self, store):
        """Same confidence → lowest QID wins."""
        e1 = PublicEntity(qid="Q200", label="Foo", entity_type="Concept", confidence=0.9)
        e2 = PublicEntity(qid="Q100", label="Foo", entity_type="Technology", confidence=0.9)
        store.absorb(e1)
        store.absorb(e2)
        patterns = store.get_ruler_patterns()
        foo_type = None
        for sf, t in patterns.items():
            if sf.lower() == "foo":
                foo_type = t
                break
        assert foo_type == "Technology"  # Q100 < Q200

    def test_ambiguity_tiebreak_numeric_not_lexicographic(self, store):
        """QID sort must be numeric: Q9 < Q100 (not lexicographic where Q9 > Q100)."""
        e1 = PublicEntity(qid="Q100", label="Bar", entity_type="Concept", confidence=0.9)
        e2 = PublicEntity(qid="Q9", label="Bar", entity_type="Technology", confidence=0.9)
        store.absorb(e1)
        store.absorb(e2)
        patterns = store.get_ruler_patterns()
        bar_type = None
        for sf, t in patterns.items():
            if sf.lower() == "bar":
                bar_type = t
                break
        assert bar_type == "Technology"  # Q9 < Q100 numerically

    def test_patterns_include_aliases(self, store, python_entity):
        store.absorb(python_entity)
        patterns = store.get_ruler_patterns()
        surface_forms_lower = {sf.lower() for sf in patterns}
        assert "python 3" in surface_forms_lower
        assert "python programming language" in surface_forms_lower


class TestLoadSnapshot:
    """load_snapshot() bulk-loads from JSONL."""

    def test_load_jsonl(self, store, tmp_path):
        jsonl_path = tmp_path / "entities.jsonl"
        entities = [
            {
                "qid": "Q28865", "label": "Python",
                "aliases": ["Python 3"], "description": "Programming language",
                "entity_type": "Technology", "instance_of": ["Q9143"],
                "domain": "software", "confidence": 0.95,
            },
            {
                "qid": "Q2622004", "label": "Django",
                "aliases": ["Django framework"], "description": "Web framework",
                "entity_type": "Technology", "instance_of": ["Q271680"],
                "domain": "software", "confidence": 0.9,
            },
        ]
        with open(jsonl_path, "w") as f:
            for e in entities:
                f.write(json.dumps(e) + "\n")

        store.load_snapshot(str(jsonl_path))
        assert store.count() == 2
        assert store.lookup_by_qid("Q28865") is not None
        assert store.lookup_by_qid("Q2622004") is not None

    def test_load_snapshot_label_as_alias(self, store, tmp_path):
        """Labels from snapshot must be searchable as aliases."""
        jsonl_path = tmp_path / "entities.jsonl"
        entity = {
            "qid": "Q2622004", "label": "Django",
            "aliases": [], "description": "Web framework",
            "entity_type": "Technology", "instance_of": ["Q271680"],
            "domain": "software", "confidence": 0.9,
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(entity) + "\n")

        store.load_snapshot(str(jsonl_path))
        # "Django" has no explicit aliases but should be findable by label
        results = store.lookup_by_alias("Django")
        assert len(results) == 1
        assert results[0].qid == "Q2622004"


class TestLoadSnapshotEdgeCases:
    """Edge cases for load_snapshot."""

    def test_load_nonexistent_file_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_snapshot("/nonexistent/path/to/snapshot.jsonl")

    def test_close(self, store, python_entity):
        store.absorb(python_entity)
        store.close()
        # After close, operations should raise
        with pytest.raises(Exception):
            store.lookup_by_qid("Q28865")

    def test_absorb_no_lock_cleans_stale_aliases(self, store, tmp_path):
        """Snapshot reload should not accumulate stale aliases."""
        # Load a snapshot with entity having alias "OldAlias"
        jsonl_path = tmp_path / "v1.jsonl"
        entity_v1 = {
            "qid": "Q100", "label": "Foo",
            "aliases": ["OldAlias"], "entity_type": "Technology",
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(entity_v1) + "\n")
        store.load_snapshot(str(jsonl_path))
        assert len(store.lookup_by_alias("OldAlias")) == 1

        # Load v2 where "OldAlias" is removed
        jsonl_v2 = tmp_path / "v2.jsonl"
        entity_v2 = {
            "qid": "Q100", "label": "Foo",
            "aliases": ["NewAlias"], "entity_type": "Technology",
        }
        with open(jsonl_v2, "w") as f:
            f.write(json.dumps(entity_v2) + "\n")
        store.load_snapshot(str(jsonl_v2))

        # OldAlias should be gone
        assert len(store.lookup_by_alias("OldAlias")) == 0
        assert len(store.lookup_by_alias("NewAlias")) == 1


class TestFalkorDBBackendContract:
    """FalkorDBPublicKnowledgeStore must work with FalkorDBBackend.query() contract.

    FalkorDBBackend.query() returns List[Any] (a plain list of records),
    NOT an object with .result_set. This test class verifies the store
    works correctly with that contract using a mock backend.
    """

    def test_lookup_by_alias_with_list_return(self):
        """backend.query() returns a plain list — no .result_set."""
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        # FalkorDBBackend.query() returns List[Any]
        backend.query.return_value = [
            ["Q28865", "Python", "Programming language", "Technology", '["Q9143"]', "software", 0.95],
        ]
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        results = store.lookup_by_alias("Python")
        assert len(results) == 1
        assert results[0].qid == "Q28865"
        assert results[0].entity_type == "Technology"

    def test_lookup_by_alias_empty_list(self):
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = []
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        results = store.lookup_by_alias("nonexistent")
        assert results == []

    def test_lookup_by_qid_with_list_return(self):
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = [
            ["Q28865", "Python", "Programming language", "Technology", '["Q9143"]', "software", 0.95],
        ]
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        result = store.lookup_by_qid("Q28865")
        assert result is not None
        assert result.qid == "Q28865"

    def test_lookup_by_qid_empty(self):
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = []
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        result = store.lookup_by_qid("Q99999")
        assert result is None

    def test_get_ruler_patterns_with_list_return(self):
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = [
            ["python", "Technology", 0.95, "Q28865"],
            ["django", "Technology", 0.9, "Q2622004"],
        ]
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        patterns = store.get_ruler_patterns()
        assert patterns == {"python": "Technology", "django": "Technology"}

    def test_count_with_list_return(self):
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = [[5]]
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        assert store.count() == 5

    def test_count_empty(self):
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = []
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=MagicMock())
        assert store.count() == 0

    def test_absorb_calls_query_without_result_set(self):
        """absorb() should not access .result_set on query return."""
        from unittest.mock import MagicMock
        from smartmemory.grounding.falkordb_store import FalkorDBPublicKnowledgeStore

        backend = MagicMock()
        backend.query.return_value = []  # MERGE returns empty list
        redis_mock = MagicMock()
        store = FalkorDBPublicKnowledgeStore(backend=backend, redis_client=redis_mock)
        entity = PublicEntity(qid="Q1", label="Test", entity_type="Concept")
        store.absorb(entity)
        # Should not raise — absorb() doesn't read query results
        assert backend.query.call_count >= 2  # entity MERGE + alias MERGE


class TestSnapshotVersion:
    """snapshot_version() returns version metadata."""

    def test_default_version(self, store):
        assert store.snapshot_version() == ""

    def test_version_after_load(self, store, tmp_path):
        jsonl_path = tmp_path / "entities.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"qid": "Q1", "label": "Test"}) + "\n")
        store.load_snapshot(str(jsonl_path))
        # Version string is set to the file path or a hash
        assert store.snapshot_version() != ""
