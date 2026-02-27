"""Integration tests for CODE-DEV-3 — semantic_code_search with real-ish setup.

Tests the golden flow: ingest_code() → semantic search → verify results.
Uses mocked embeddings but exercises the full filter/hydration pipeline.

Marked as integration (auto-marked by conftest for tests/integration/).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartmemory.code.models import CodeEntity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(name: str, entity_type: str = "function", file_path: str = "src/app.py",
                 repo: str = "test-repo") -> CodeEntity:
    return CodeEntity(
        name=name,
        entity_type=entity_type,
        file_path=file_path,
        line_number=42,
        repo=repo,
        docstring=f"Does {name} things.",
    )


def _make_sm_with_di_context():
    """Create a SmartMemory stub with all _di_context() attributes initialized."""
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


# ---------------------------------------------------------------------------
# Tests: _di_context fix (Task 1)
# ---------------------------------------------------------------------------


class TestIngestCodeDiContext:
    """Verify _di_context() is entered during ingest_code() so Lite mode works."""

    def test_di_context_entered_during_ingest_code(self):
        """_di_context() is entered when ingest_code() runs."""
        sm = _make_sm_with_di_context()
        mock_result = MagicMock()
        mock_result.entities = []

        mock_indexer = MagicMock()
        mock_indexer.index.return_value = mock_result

        entered = []

        original_di = sm._di_context

        from contextlib import contextmanager

        @contextmanager
        def tracking_di():
            with original_di() as ctx:
                entered.append(True)
                yield ctx

        sm._di_context = tracking_di

        with (
            patch("smartmemory.code.indexer.CodeIndexer", return_value=mock_indexer),
            patch("smartmemory.smart_memory.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            sm.ingest_code(directory="/tmp/test", repo="test-repo")

        assert len(entered) == 1, "_di_context() was not entered during ingest_code()"

    def test_vector_backend_visible_inside_di_context(self):
        """When vector_backend is set, it's available via ContextVar inside _di_context()."""
        sm = _make_sm_with_di_context()
        mock_backend = MagicMock()
        sm._vector_backend = mock_backend

        with sm._di_context():
            from smartmemory.stores.vector.vector_store import _vector_backend_ctx
            assert _vector_backend_ctx.get() is mock_backend


# ---------------------------------------------------------------------------
# Tests: semantic_code_search golden flow (Task 2)
# ---------------------------------------------------------------------------


class TestSemanticSearchGoldenFlow:
    """Golden flow: vector search → post-filter → hydrate → response."""

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings")
    def test_full_pipeline(self, mock_embed, mock_vs_cls):
        """End-to-end: embed query → vector search → filter → hydrate → return."""
        from smartmemory.code.search import semantic_code_search

        # Setup: embedding returns a vector
        mock_embed.return_value = np.array([0.1, 0.2, 0.3])

        # Setup: vector search returns mixed results
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [
            {
                "id": "code::test-repo::auth.py::authenticate",
                "score": 0.95,
                "metadata": {"memory_type": "code", "entity_type": "function", "repo": "test-repo"},
            },
            {
                "id": "mem-semantic-1",
                "score": 0.90,
                "metadata": {"memory_type": "semantic"},
            },
            {
                "id": "code::test-repo::auth.py::validate",
                "score": 0.85,
                "metadata": {"memory_type": "code", "entity_type": "function", "repo": "test-repo"},
            },
        ]

        # Setup: graph returns nodes for code items
        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: {
            "item_id": item_id,
            "name": item_id.split("::")[-1],
            "entity_type": "function",
            "file_path": "auth.py",
            "line_number": 10,
            "docstring": "Auth logic",
            "repo": "test-repo",
            "http_method": "",
            "http_path": "",
        }

        results = semantic_code_search(mock_graph, "authentication functions")

        # Assertions
        assert len(results) == 2  # only code items
        assert results[0]["name"] == "authenticate"
        assert results[0]["score"] == 0.95
        assert results[1]["name"] == "validate"
        assert results[1]["score"] == 0.85

        # Verify embedding was called with the query
        mock_embed.assert_called_once_with("authentication functions")

        # Verify vector store received 3x oversampling
        vs_call = mock_vs.search.call_args
        assert vs_call.kwargs.get("top_k") == 60  # top_k=20 * 3

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings")
    def test_scope_provider_forwarded_to_vector_store(self, mock_embed, mock_vs_cls):
        """scope_provider is passed to VectorStore constructor."""
        from smartmemory.code.search import semantic_code_search

        mock_embed.return_value = [0.1, 0.2]
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = []

        mock_scope = MagicMock()
        mock_scope.workspace_id = "ws-isolated"

        semantic_code_search(MagicMock(), "auth", scope_provider=mock_scope)

        # VectorStore was constructed with our scope_provider
        vs_init_kwargs = mock_vs_cls.call_args
        assert vs_init_kwargs.kwargs.get("scope_provider") is mock_scope

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings")
    def test_combined_filters(self, mock_embed, mock_vs_cls):
        """entity_type + repo filters applied together."""
        from smartmemory.code.search import semantic_code_search

        mock_embed.return_value = [0.1]
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [
            {"id": "a", "score": 0.9, "metadata": {"memory_type": "code", "entity_type": "function", "repo": "repo-a"}},
            {"id": "b", "score": 0.8, "metadata": {"memory_type": "code", "entity_type": "class", "repo": "repo-a"}},
            {"id": "c", "score": 0.7, "metadata": {"memory_type": "code", "entity_type": "function", "repo": "repo-b"}},
        ]

        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: {
            "item_id": item_id, "name": item_id, "entity_type": "function",
            "file_path": "x.py", "line_number": 1, "docstring": "", "repo": "repo-a",
            "http_method": "", "http_path": "",
        }

        results = semantic_code_search(mock_graph, "auth", entity_type="function", repo="repo-a")

        assert len(results) == 1
        assert results[0]["item_id"] == "a"
