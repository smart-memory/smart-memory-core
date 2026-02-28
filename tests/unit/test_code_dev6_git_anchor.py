"""Unit tests for CODE-DEV-6 — Git-Anchored Code Memory.

Tests:
- CodeEntity with commit_hash/indexed_at serializes via to_properties()
- CodeEntity without commit_hash omits field from to_properties()
- IndexResult.commit_hash is a declared field
- CodeIndexer stamps commit_hash on all entities
- CodeIndexer stamps indexed_at (ISO 8601 UTC) on all entities
- CodeIndexer with empty commit_hash leaves entities empty
- CodeIndexer populates IndexResult.commit_hash
- ingest_code() forwards commit_hash to CodeIndexer
- ingest_code() auto-detected commit_hash reaches CodeIndexer
"""

import datetime
from unittest.mock import MagicMock, patch

from smartmemory.code.models import CodeEntity, IndexResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(name: str = "my_func", entity_type: str = "function", file_path: str = "a/b.py") -> CodeEntity:
    return CodeEntity(
        name=name,
        entity_type=entity_type,
        file_path=file_path,
        line_number=1,
        repo="test-repo",
        docstring="Does something useful.",
    )


def _fake_graph(workspace_id=None):
    g = MagicMock()
    g.add_nodes_bulk.return_value = 1
    g.add_edges_bulk.return_value = 0
    scope_filters = {"workspace_id": workspace_id} if workspace_id is not None else {}
    g.get_scope_filters.return_value = scope_filters
    g.execute_query.return_value = []
    return g


# ---------------------------------------------------------------------------
# Test 1: CodeEntity.to_properties with commit_hash/indexed_at
# ---------------------------------------------------------------------------


class TestCodeEntitySerialization:
    """CodeEntity serializes git anchor fields conditionally."""

    def test_with_commit_hash_and_indexed_at(self):
        entity = _make_entity()
        entity.commit_hash = "abc123"
        entity.indexed_at = "2026-02-28T12:00:00+00:00"

        props = entity.to_properties()
        assert props["commit_hash"] == "abc123"
        assert props["indexed_at"] == "2026-02-28T12:00:00+00:00"

    def test_without_commit_hash_omits_field(self):
        entity = _make_entity()
        # commit_hash defaults to ""

        props = entity.to_properties()
        assert "commit_hash" not in props
        assert "indexed_at" not in props

    def test_with_commit_hash_only(self):
        entity = _make_entity()
        entity.commit_hash = "abc123"

        props = entity.to_properties()
        assert props["commit_hash"] == "abc123"
        assert "indexed_at" not in props


# ---------------------------------------------------------------------------
# Test 2: IndexResult.commit_hash is a declared field
# ---------------------------------------------------------------------------


class TestIndexResultCommitHash:
    """IndexResult has commit_hash as a proper dataclass field."""

    def test_default_empty_string(self):
        result = IndexResult(repo="test-repo")
        assert result.commit_hash == ""

    def test_set_via_constructor(self):
        result = IndexResult(repo="test-repo", commit_hash="abc123")
        assert result.commit_hash == "abc123"

    def test_assignable(self):
        result = IndexResult(repo="test-repo")
        result.commit_hash = "def456"
        assert result.commit_hash == "def456"


# ---------------------------------------------------------------------------
# Test 3: CodeIndexer stamps commit_hash and indexed_at on entities
# ---------------------------------------------------------------------------


class TestCodeIndexerStamping:
    """CodeIndexer stamps git anchor fields on all entities during index()."""

    def test_stamps_commit_hash_on_all_entities(self):
        from smartmemory.code.indexer import CodeIndexer

        entities = [_make_entity("func_a"), _make_entity("func_b")]

        with (
            patch("smartmemory.code.indexer.collect_python_files", return_value=["/tmp/a.py"]),
            patch("smartmemory.code.indexer.parse_file") as mock_parse,
        ):
            from smartmemory.code.models import ParseResult

            mock_parse.return_value = ParseResult(
                file_path="a.py",
                entities=entities,
            )

            indexer = CodeIndexer(
                graph=_fake_graph(),
                repo="test-repo",
                repo_root="/tmp",
                commit_hash="abc123def456",
            )
            result = indexer.index()

        for entity in result.entities:
            assert entity.commit_hash == "abc123def456"

    def test_stamps_indexed_at_iso8601_utc(self):
        from smartmemory.code.indexer import CodeIndexer

        entities = [_make_entity("func_a")]

        with (
            patch("smartmemory.code.indexer.collect_python_files", return_value=["/tmp/a.py"]),
            patch("smartmemory.code.indexer.parse_file") as mock_parse,
        ):
            from smartmemory.code.models import ParseResult

            mock_parse.return_value = ParseResult(
                file_path="a.py",
                entities=entities,
            )

            indexer = CodeIndexer(
                graph=_fake_graph(),
                repo="test-repo",
                repo_root="/tmp",
                commit_hash="abc123",
            )
            result = indexer.index()

        for entity in result.entities:
            assert entity.indexed_at != ""
            # Verify it parses as valid ISO 8601
            dt = datetime.datetime.fromisoformat(entity.indexed_at)
            assert dt.tzinfo is not None  # UTC-aware

    def test_empty_commit_hash_leaves_entities_empty(self):
        from smartmemory.code.indexer import CodeIndexer

        entities = [_make_entity("func_a")]

        with (
            patch("smartmemory.code.indexer.collect_python_files", return_value=["/tmp/a.py"]),
            patch("smartmemory.code.indexer.parse_file") as mock_parse,
        ):
            from smartmemory.code.models import ParseResult

            mock_parse.return_value = ParseResult(
                file_path="a.py",
                entities=entities,
            )

            indexer = CodeIndexer(
                graph=_fake_graph(),
                repo="test-repo",
                repo_root="/tmp",
                commit_hash="",
            )
            result = indexer.index()

        for entity in result.entities:
            assert entity.commit_hash == ""
            # indexed_at is still stamped even without commit_hash
            assert entity.indexed_at != ""

    def test_populates_index_result_commit_hash(self):
        from smartmemory.code.indexer import CodeIndexer

        with (
            patch("smartmemory.code.indexer.collect_python_files", return_value=[]),
            patch("smartmemory.code.indexer.collect_ts_files", return_value=[]),
        ):
            indexer = CodeIndexer(
                graph=_fake_graph(),
                repo="test-repo",
                repo_root="/tmp",
                commit_hash="abc123",
            )
            result = indexer.index()

        assert result.commit_hash == "abc123"


# ---------------------------------------------------------------------------
# Test 4: ingest_code() forwards commit_hash to CodeIndexer
# ---------------------------------------------------------------------------


class TestIngestCodeForwardsCommitHash:
    """ingest_code() passes commit_hash to CodeIndexer constructor."""

    def _make_sm(self):
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._graph = MagicMock()
        sm.scope_provider = MagicMock()
        sm.scope_provider.get_scope.return_value = MagicMock(workspace_id="default")
        sm._enable_ontology = False
        sm._code_pattern_manager = None
        sm._cache = None
        sm._vector_backend = None
        sm._observability = True
        sm._event_sink = None
        return sm

    def test_explicit_commit_hash_forwarded(self):
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer) as mock_cls,
            patch("smartmemory.smart_memory.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            sm.ingest_code(directory="/tmp/my-project", repo="test-repo", commit_hash="explicit-sha")

        # Verify CodeIndexer was constructed with commit_hash
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["commit_hash"] == "explicit-sha"

    def test_autodetected_commit_hash_forwarded(self):
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        git_proc = MagicMock()
        git_proc.returncode = 0
        git_proc.stdout = "autodetect123\n"

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer) as mock_cls,
            patch("smartmemory.smart_memory.subprocess.run", return_value=git_proc),
        ):
            sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["commit_hash"] == "autodetect123"

    def test_git_unavailable_passes_empty_commit_hash(self):
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer) as mock_cls,
            patch("smartmemory.smart_memory.subprocess.run", side_effect=FileNotFoundError("git not found")),
        ):
            sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["commit_hash"] == ""
