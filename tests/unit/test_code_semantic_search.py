"""Unit tests for CODE-DEV-3 — semantic_code_search() core function.

Tests:
- test_post_filter_only_code_type: mixed memory_type results, only code passes
- test_entity_type_filter: entity_type post-filter applied
- test_repo_filter: repo post-filter applied
- test_empty_query_returns_empty: blank/empty query returns []
- test_embedding_failure_returns_empty: create_embeddings() returns None → []
- test_vector_search_failure_returns_empty: VectorStore.search() raises → []
- test_deleted_node_skipped: graph.get_node() returns None → silently skipped
- test_score_included_in_results: score from vector hit carried into response
- test_top_k_respected: no more than top_k results returned
- test_oversampling_factor: VectorStore.search called with top_k * 3
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vector_hit(item_id: str, memory_type: str = "code", score: float = 0.9,
                     entity_type: str = "function", repo: str = "test-repo") -> dict:
    """Build a vector search result hit."""
    return {
        "id": item_id,
        "score": score,
        "metadata": {
            "memory_type": memory_type,
            "entity_type": entity_type,
            "repo": repo,
        },
    }


def _make_graph_node(item_id: str, name: str = "my_func", entity_type: str = "function",
                     file_path: str = "src/app.py", line_number: int = 10,
                     repo: str = "test-repo") -> dict:
    """Build a graph node dict matching what SmartGraph.get_node() returns."""
    return {
        "item_id": item_id,
        "name": name,
        "entity_type": entity_type,
        "file_path": file_path,
        "line_number": line_number,
        "docstring": f"Docstring for {name}",
        "repo": repo,
        "http_method": "",
        "http_path": "",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSemanticCodeSearchPostFilter:
    """Post-filter logic: only memory_type='code' results pass through."""

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_post_filter_only_code_type(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [
            _make_vector_hit("code-1", memory_type="code", score=0.95),
            _make_vector_hit("mem-1", memory_type="semantic", score=0.90),
            _make_vector_hit("code-2", memory_type="code", score=0.85),
        ]

        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: _make_graph_node(item_id)

        results = semantic_code_search(mock_graph, "authentication")

        assert len(results) == 2
        assert results[0]["item_id"] == "code-1"
        assert results[1]["item_id"] == "code-2"

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_entity_type_filter(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [
            _make_vector_hit("fn-1", entity_type="function", score=0.9),
            _make_vector_hit("cls-1", entity_type="class", score=0.85),
        ]

        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: _make_graph_node(item_id)

        results = semantic_code_search(mock_graph, "auth", entity_type="function")

        assert len(results) == 1
        assert results[0]["item_id"] == "fn-1"

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_repo_filter(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [
            _make_vector_hit("a", repo="my-repo", score=0.9),
            _make_vector_hit("b", repo="other-repo", score=0.85),
        ]

        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: _make_graph_node(item_id)

        results = semantic_code_search(mock_graph, "auth", repo="my-repo")

        assert len(results) == 1
        assert results[0]["item_id"] == "a"


class TestSemanticCodeSearchGracefulDegradation:
    """Graceful return [] on various failure modes."""

    def test_empty_query_returns_empty(self):
        from smartmemory.code.search import semantic_code_search

        assert semantic_code_search(MagicMock(), "") == []
        assert semantic_code_search(MagicMock(), "   ") == []

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=None)
    def test_embedding_failure_returns_empty(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        results = semantic_code_search(MagicMock(), "authentication")

        assert results == []
        mock_vs_cls.assert_not_called()  # shouldn't even reach vector search

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_vector_search_failure_returns_empty(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.side_effect = RuntimeError("connection refused")

        results = semantic_code_search(MagicMock(), "auth")

        assert results == []


class TestSemanticCodeSearchHydration:
    """Graph hydration: get_node() for each hit, skip deleted nodes."""

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_deleted_node_skipped(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [
            _make_vector_hit("exists", score=0.9),
            _make_vector_hit("deleted", score=0.8),
        ]

        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: (
            _make_graph_node(item_id) if item_id == "exists" else None
        )

        results = semantic_code_search(mock_graph, "auth")

        assert len(results) == 1
        assert results[0]["item_id"] == "exists"

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_score_included_in_results(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [_make_vector_hit("fn-1", score=0.87)]

        mock_graph = MagicMock()
        mock_graph.get_node.return_value = _make_graph_node("fn-1")

        results = semantic_code_search(mock_graph, "auth")

        assert len(results) == 1
        assert results[0]["score"] == 0.87

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_line_number_coerced_to_int(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = [_make_vector_hit("fn-1", score=0.9)]

        node = _make_graph_node("fn-1")
        node["line_number"] = "42"  # string from graph
        mock_graph = MagicMock()
        mock_graph.get_node.return_value = node

        results = semantic_code_search(mock_graph, "auth")

        assert results[0]["line_number"] == 42
        assert isinstance(results[0]["line_number"], int)


class TestSemanticCodeSearchOversampling:
    """Vector search is called with 3x oversampling."""

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_oversampling_factor(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.search.return_value = []

        semantic_code_search(MagicMock(), "auth", top_k=10)

        call_kwargs = mock_vs.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 30 or call_kwargs[1].get("top_k") == 30

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    @patch("smartmemory.plugins.embedding.create_embeddings", return_value=[0.1, 0.2])
    def test_top_k_respected(self, mock_embed, mock_vs_cls):
        from smartmemory.code.search import semantic_code_search

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        # Return more code hits than top_k
        mock_vs.search.return_value = [
            _make_vector_hit(f"fn-{i}", score=0.9 - i * 0.01) for i in range(10)
        ]

        mock_graph = MagicMock()
        mock_graph.get_node.side_effect = lambda item_id: _make_graph_node(item_id)

        results = semantic_code_search(mock_graph, "auth", top_k=3)

        assert len(results) == 3
