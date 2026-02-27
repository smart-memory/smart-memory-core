"""Unit tests for CODE-DEV-2/C — Embeddings in CodeIndexer.

Tests:
- test_indexer_generates_embeddings: mock embedding service, call index(), assert
  embeddings stored on nodes via VectorStore.upsert
- test_ingest_code_calls_seed_patterns: assert seed_patterns_from_code is called
  after indexing
- test_ingest_code_autodetects_commit_hash: mock subprocess.run, assert
  commit_hash captured
- test_seed_patterns_from_code: assert PatternManager receives entity name→type dict
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
# Test 1: CodeIndexer._generate_embeddings stores embeddings via VectorStore
# ---------------------------------------------------------------------------


class TestIndexerGeneratesEmbeddings:
    """_generate_embeddings calls create_embeddings and upserts to VectorStore."""

    def test_embedding_generated_per_entity_with_workspace(self):
        """When graph has a workspace_id scope filter, upsert uses is_global=False and includes workspace_id in metadata."""
        entities = [_make_entity("process_payment", "function", "payments/service.py")]

        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_vs = MagicMock()

        from smartmemory.code.indexer import CodeIndexer

        indexer = CodeIndexer(graph=_fake_graph(workspace_id="ws-acme"), repo="test-repo", repo_root="/tmp")
        with (
            patch("smartmemory.plugins.embedding.create_embeddings", return_value=mock_embedding),
            patch("smartmemory.stores.vector.vector_store.VectorStore", return_value=mock_vs),
        ):
            count = indexer._generate_embeddings(entities)

        assert count == 1
        mock_vs.upsert.assert_called_once()
        call_kwargs = mock_vs.upsert.call_args
        assert call_kwargs.kwargs["item_id"] == entities[0].item_id
        assert call_kwargs.kwargs["node_ids"] == [entities[0].item_id]
        assert call_kwargs.kwargs["is_global"] is False
        assert call_kwargs.kwargs["metadata"]["memory_type"] == "code"
        assert call_kwargs.kwargs["metadata"]["workspace_id"] == "ws-acme"

    def test_embedding_generated_per_entity_no_workspace(self):
        """When graph has no workspace_id (local/Lite mode), upsert falls back to is_global=True."""
        entities = [_make_entity("process_payment", "function", "payments/service.py")]

        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_vs = MagicMock()

        from smartmemory.code.indexer import CodeIndexer

        indexer = CodeIndexer(graph=_fake_graph(), repo="test-repo", repo_root="/tmp")
        with (
            patch("smartmemory.plugins.embedding.create_embeddings", return_value=mock_embedding),
            patch("smartmemory.stores.vector.vector_store.VectorStore", return_value=mock_vs),
        ):
            count = indexer._generate_embeddings(entities)

        assert count == 1
        mock_vs.upsert.assert_called_once()
        call_kwargs = mock_vs.upsert.call_args
        assert call_kwargs.kwargs["item_id"] == entities[0].item_id
        assert call_kwargs.kwargs["node_ids"] == [entities[0].item_id]
        assert call_kwargs.kwargs["is_global"] is True
        assert call_kwargs.kwargs["metadata"]["memory_type"] == "code"
        assert "workspace_id" not in call_kwargs.kwargs["metadata"]

    def test_embedding_text_composite(self):
        """Embedding input is name + entity_type + file_path + docstring."""
        entity_with_doc = CodeEntity(
            name="TokenValidator",
            entity_type="class",
            file_path="auth/validator.py",
            line_number=1,
            repo="test-repo",
            docstring="Validates JWT tokens.",
        )

        captured_texts = []

        def capture_embed(text):
            captured_texts.append(text)
            return np.array([0.5, 0.5])

        mock_vs = MagicMock()
        from smartmemory.code.indexer import CodeIndexer

        indexer = CodeIndexer(graph=_fake_graph(), repo="test-repo", repo_root="/tmp")
        with (
            patch("smartmemory.plugins.embedding.create_embeddings", side_effect=capture_embed),
            patch("smartmemory.stores.vector.vector_store.VectorStore", return_value=mock_vs),
        ):
            indexer._generate_embeddings([entity_with_doc])

        assert len(captured_texts) == 1
        expected = "TokenValidator class auth/validator.py Validates JWT tokens."
        assert captured_texts[0] == expected

    def test_empty_entities_returns_zero(self):
        """_generate_embeddings returns 0 and makes no API calls for empty input."""
        from smartmemory.code.indexer import CodeIndexer

        indexer = CodeIndexer(graph=_fake_graph(), repo="test-repo", repo_root="/tmp")
        with patch("smartmemory.plugins.embedding.create_embeddings") as mock_embed:
            count = indexer._generate_embeddings([])

        assert count == 0
        mock_embed.assert_not_called()

    def test_failed_entity_does_not_abort_rest(self):
        """An exception for one entity does not stop embedding of others."""
        entities = [
            _make_entity("good_func", "function", "a.py"),
            _make_entity("bad_func", "function", "b.py"),
            _make_entity("another_good", "function", "c.py"),
        ]

        call_count = [0]

        def flaky_embed(text):
            call_count[0] += 1
            if "bad_func" in text:
                raise RuntimeError("API error")
            return np.array([0.1, 0.2])

        mock_vs = MagicMock()
        from smartmemory.code.indexer import CodeIndexer

        indexer = CodeIndexer(graph=_fake_graph(), repo="test-repo", repo_root="/tmp")
        with (
            patch("smartmemory.plugins.embedding.create_embeddings", side_effect=flaky_embed),
            patch("smartmemory.stores.vector.vector_store.VectorStore", return_value=mock_vs),
        ):
            count = indexer._generate_embeddings(entities)

        assert count == 2  # good_func + another_good
        assert mock_vs.upsert.call_count == 2

    def test_index_result_embeddings_generated_populated(self):
        """IndexResult.embeddings_generated is set after index() completes."""
        from smartmemory.code.indexer import CodeIndexer

        mock_embedding = np.array([0.1])
        mock_vs = MagicMock()

        with (
            patch("smartmemory.code.indexer.collect_python_files", return_value=[]),
            patch("smartmemory.code.indexer.collect_ts_files", return_value=[]),
        ):
            indexer = CodeIndexer(graph=_fake_graph(), repo="test-repo", repo_root="/tmp")
            with (
                patch("smartmemory.plugins.embedding.create_embeddings", return_value=mock_embedding),
                patch("smartmemory.stores.vector.vector_store.VectorStore", return_value=mock_vs),
            ):
                result = indexer.index()

        # No files → no entities → 0 embeddings
        assert result.embeddings_generated == 0
        assert result.entities == []


# ---------------------------------------------------------------------------
# Test 2: SmartMemory.ingest_code calls seed_patterns_from_code
# ---------------------------------------------------------------------------


class TestIngestCodeCallsSeedPatterns:
    """ingest_code() calls seed_patterns_from_code after indexing."""

    def _make_sm(self):
        """Create a minimal SmartMemory stub for testing."""
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._graph = MagicMock()
        sm.scope_provider = MagicMock()
        sm.scope_provider.get_scope.return_value = MagicMock(workspace_id="default")
        sm._enable_ontology = False
        sm._code_pattern_manager = None
        return sm

    def test_seed_patterns_called_after_index(self):
        """seed_patterns_from_code is called with entities from the index result."""
        sm = self._make_sm()
        entity = _make_entity("PaymentService", "class", "payments/service.py")
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = [entity]
        mock_result.embeddings_generated = 1

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch.object(sm, "seed_patterns_from_code") as mock_seed,
            patch("smartmemory.smart_memory.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)  # git not available
            sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        mock_seed.assert_called_once_with([entity])

    def test_seed_not_called_when_no_entities(self):
        """seed_patterns_from_code is NOT called if index returns no entities."""
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch.object(sm, "seed_patterns_from_code") as mock_seed,
            patch("smartmemory.smart_memory.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2b: SmartMemory.ingest_code passes languages to indexer
# ---------------------------------------------------------------------------


class TestIngestCodePassesLanguages:
    """ingest_code() forwards the languages parameter to CodeIndexer.index()."""

    def _make_sm(self):
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._graph = MagicMock()
        sm.scope_provider = MagicMock()
        sm.scope_provider.get_scope.return_value = MagicMock(workspace_id="default")
        sm._enable_ontology = False
        sm._code_pattern_manager = None
        return sm

    def test_ingest_code_passes_languages_to_indexer(self):
        """languages kwarg is forwarded to CodeIndexer.index()."""
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch("smartmemory.smart_memory.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            sm.ingest_code(
                directory="/tmp/my-project",
                repo="test-repo",
                languages=["python", "typescript"],
            )

        mock_indexer.index.assert_called_once_with(languages=["python", "typescript"])


# ---------------------------------------------------------------------------
# Test 3: SmartMemory.ingest_code auto-detects commit hash
# ---------------------------------------------------------------------------


class TestIngestCodeAutodetectsCommitHash:
    """ingest_code() auto-detects commit_hash via git rev-parse HEAD."""

    def _make_sm(self):
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._graph = MagicMock()
        sm.scope_provider = MagicMock()
        sm.scope_provider.get_scope.return_value = MagicMock(workspace_id="default")
        sm._enable_ontology = False
        sm._code_pattern_manager = None
        return sm

    def test_commit_hash_captured_when_git_available(self):
        """When git returns 0, commit_hash is stored on the result."""
        sm = self._make_sm()
        expected_sha = "abc123def456"
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        git_proc = MagicMock()
        git_proc.returncode = 0
        git_proc.stdout = f"{expected_sha}\n"

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch("smartmemory.smart_memory.subprocess.run", return_value=git_proc) as mock_run,
        ):
            result = sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        mock_run.assert_called_once_with(
            ["git", "-C", "/tmp/my-project", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        assert getattr(result, "commit_hash", None) == expected_sha

    def test_commit_hash_none_when_git_unavailable(self):
        """When git returns non-zero, commit_hash is not set on the result."""
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        git_proc = MagicMock()
        git_proc.returncode = 128
        git_proc.stdout = ""

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch("smartmemory.smart_memory.subprocess.run", return_value=git_proc),
        ):
            result = sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        assert getattr(result, "commit_hash", None) is None

    def test_provided_commit_hash_used_without_git_call(self):
        """When commit_hash is provided explicitly, git is not invoked."""
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch("smartmemory.smart_memory.subprocess.run") as mock_run,
        ):
            result = sm.ingest_code(
                directory="/tmp/my-project",
                repo="test-repo",
                commit_hash="explicit-sha",
            )

        mock_run.assert_not_called()
        assert getattr(result, "commit_hash", None) == "explicit-sha"

    def test_commit_hash_none_when_git_not_on_path(self):
        """When git binary is missing (FileNotFoundError), commit_hash is not set."""
        sm = self._make_sm()
        mock_result = IndexResult(repo="test-repo")
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch("smartmemory.smart_memory.subprocess.run", side_effect=FileNotFoundError("git not found")),
        ):
            result = sm.ingest_code(directory="/tmp/my-project", repo="test-repo")

        assert getattr(result, "commit_hash", None) is None


# ---------------------------------------------------------------------------
# Test 4: SmartMemory.seed_patterns_from_code
# ---------------------------------------------------------------------------


class TestSeedPatternsFromCode:
    """seed_patterns_from_code passes name→type dict to PatternManager."""

    def test_patterns_forwarded_to_pattern_manager(self):
        """PatternManager.add_patterns receives {name: entity_type} for each entity."""
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._enable_ontology = True
        sm.scope_provider = MagicMock()
        sm.scope_provider.get_scope.return_value = MagicMock(workspace_id="ws-test")

        mock_pm = MagicMock()
        mock_pm.add_patterns.return_value = 2
        sm._code_pattern_manager = mock_pm

        entities = [
            _make_entity("TokenValidator", "class"),
            _make_entity("validate_token", "function"),
        ]
        sm.seed_patterns_from_code(entities)

        mock_pm.add_patterns.assert_called_once_with(
            {"TokenValidator": "class", "validate_token": "function"}
        )

    def test_empty_entities_is_noop(self):
        """seed_patterns_from_code with no entities does nothing."""
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._enable_ontology = True
        sm._code_pattern_manager = MagicMock()

        sm.seed_patterns_from_code([])

        sm._code_pattern_manager.add_patterns.assert_not_called()

    def test_nameless_entities_excluded(self):
        """Entities with empty name are skipped."""
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._enable_ontology = True
        mock_pm = MagicMock()
        mock_pm.add_patterns.return_value = 1
        sm._code_pattern_manager = mock_pm

        entities = [
            _make_entity("GoodClass", "class"),
            CodeEntity(name="", entity_type="function", file_path="x.py", line_number=1, repo="r"),
        ]
        sm.seed_patterns_from_code(entities)

        mock_pm.add_patterns.assert_called_once_with({"GoodClass": "class"})

    def test_ontology_disabled_is_noop(self):
        """When _enable_ontology=False, no pattern manager is created or called."""
        from smartmemory.smart_memory import SmartMemory

        sm = SmartMemory.__new__(SmartMemory)
        sm._enable_ontology = False
        sm._code_pattern_manager = None
        sm.scope_provider = MagicMock()
        sm.scope_provider.get_scope.return_value = MagicMock(workspace_id="default")

        entities = [_make_entity("SomeClass", "class")]

        # Should not raise, and _code_pattern_manager remains None after the call
        sm.seed_patterns_from_code(entities)
        assert sm._code_pattern_manager is None


# ---------------------------------------------------------------------------
# Test 5: PatternManager.add_patterns (unit test for the new method)
# ---------------------------------------------------------------------------


class TestPatternManagerAddPatterns:
    """PatternManager.add_patterns merges patterns into in-memory cache."""

    def _make_pm(self):
        from smartmemory.ontology.pattern_manager import PatternManager

        mock_ontology = MagicMock()
        mock_ontology.get_entity_patterns.return_value = []
        mock_ontology.add_entity_pattern.return_value = True

        pm = PatternManager(ontology_graph=mock_ontology, workspace_id="test-ws")
        return pm

    def test_new_patterns_are_accepted(self):
        pm = self._make_pm()
        accepted = pm.add_patterns({"TokenValidator": "class", "validate_token": "function"})
        assert accepted == 2
        patterns = pm.get_patterns()
        assert patterns["tokenvalidator"] == "class"
        assert patterns["validate_token"] == "function"

    def test_duplicate_pattern_not_re_accepted(self):
        pm = self._make_pm()
        pm.add_patterns({"TokenValidator": "class"})
        accepted2 = pm.add_patterns({"TokenValidator": "class"})
        # Second call for same key returns 0 (already in cache)
        assert accepted2 == 0

    def test_short_names_blocked(self):
        pm = self._make_pm()
        accepted = pm.add_patterns({"a": "class", "ab": "function"})
        # "a" is length 1 → blocked; "ab" is length 2 → accepted
        assert accepted == 1
        patterns = pm.get_patterns()
        assert "a" not in patterns
        assert "ab" in patterns

    def test_common_word_blocked(self):
        """Words in the common word blocklist are rejected."""
        from smartmemory.ontology.promotion import COMMON_WORD_BLOCKLIST

        pm = self._make_pm()
        if not COMMON_WORD_BLOCKLIST:
            pytest.skip("COMMON_WORD_BLOCKLIST is empty")

        blocked_word = next(iter(COMMON_WORD_BLOCKLIST))
        accepted = pm.add_patterns({blocked_word: "class"})
        assert accepted == 0

    def test_persisted_to_ontology_graph(self):
        from smartmemory.ontology.pattern_manager import PatternManager

        mock_ontology = MagicMock()
        mock_ontology.get_entity_patterns.return_value = []
        mock_ontology.add_entity_pattern.return_value = True

        pm = PatternManager(ontology_graph=mock_ontology, workspace_id="test-ws")
        pm.add_patterns({"PaymentService": "class"})

        mock_ontology.add_entity_pattern.assert_called_with(
            name="paymentservice",
            label="class",
            confidence=0.9,
            workspace_id="test-ws",
            source="code_index",
        )


# ---------------------------------------------------------------------------
# Test 6: MemoryItem.display_text for code nodes
# ---------------------------------------------------------------------------


class TestMemoryItemDisplayText:
    """display_text returns code metadata for code nodes, content for others."""

    def test_regular_item_returns_content(self):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content="Hello world", memory_type="semantic")
        assert item.display_text == "Hello world"

    def test_code_item_returns_name_and_path(self):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="",
            memory_type="code",
            metadata={"name": "process_payment", "file_path": "payments/service.py"},
        )
        assert item.display_text == "process_payment (payments/service.py)"

    def test_code_item_name_only_when_no_path(self):
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="",
            memory_type="code",
            metadata={"name": "process_payment"},
        )
        assert item.display_text == "process_payment"

    def test_code_item_with_content_returns_content(self):
        """If a code item somehow has content set, return it."""
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(
            content="some code content",
            memory_type="code",
            metadata={"name": "process_payment", "file_path": "p.py"},
        )
        # content is truthy → fall through to content path
        assert item.display_text == "some code content"
