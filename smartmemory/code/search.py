"""Semantic code search using vector embeddings.

Provides intent-based search over code entities (classes, functions, modules,
routes, tests) indexed by ``CodeIndexer``.  Embeddings are generated during
``ingest_code()`` and stored in the shared ``semantic_memory`` vector collection
with ``memory_type="code"`` metadata.

Works with both FalkorDB (hosted) and usearch (Lite) backends — ``VectorStore``
resolves the active backend via ContextVar or config.
"""

import logging
from typing import Any, Optional

from smartmemory.graph.smartgraph import SmartGraph
from smartmemory.scope_provider import DefaultScopeProvider, ScopeProvider

logger = logging.getLogger(__name__)

# Fields extracted from Code graph nodes into the response dict.
# Matches the shape returned by GET /memory/code/search.
_CODE_FIELDS = (
    "item_id",
    "name",
    "entity_type",
    "file_path",
    "line_number",
    "docstring",
    "repo",
    "http_method",
    "http_path",
)


def semantic_code_search(
    graph: SmartGraph,
    query: str,
    top_k: int = 20,
    entity_type: Optional[str] = None,
    repo: Optional[str] = None,
    scope_provider: Optional[ScopeProvider] = None,
) -> list[dict[str, Any]]:
    """Semantic search over code entities using vector embeddings.

    Generates an embedding for *query*, searches the shared vector collection,
    post-filters to ``memory_type="code"`` results, and hydrates each hit from
    the graph backend.

    Args:
        graph: SmartGraph instance (for fetching full Code node properties).
        query: Natural language search query.
        top_k: Maximum results to return.
        entity_type: Optional filter (module, class, function, route, test).
        repo: Optional filter by repository name.
        scope_provider: Optional scope provider for workspace isolation.
            Passed to ``VectorStore`` for metadata-based filtering.

    Returns:
        List of dicts with ``item_id``, ``name``, ``entity_type``,
        ``file_path``, ``line_number``, ``docstring``, ``repo``,
        ``http_method``, ``http_path``, ``score``.
        Sorted by relevance score descending.  Empty list if embeddings
        are unavailable.
    """
    if not query or not query.strip():
        return []

    try:
        from smartmemory.plugins.embedding import create_embeddings
        from smartmemory.stores.vector.vector_store import VectorStore
    except ImportError as exc:
        logger.warning("Embedding dependencies unavailable: %s", exc)
        return []

    # 1. Embed the query
    query_embedding = create_embeddings(query)
    if query_embedding is None:
        logger.warning("Failed to generate query embedding for code search")
        return []
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    # 2. Vector search — 3x oversampling to account for mixed-type collection
    sp = scope_provider or DefaultScopeProvider()
    vector_store = VectorStore(scope_provider=sp)
    try:
        vector_results = vector_store.search(
            query_embedding,
            top_k=top_k * 3,
            query_text=query,
        )
    except Exception as exc:
        logger.error("Vector search failed for code semantic search: %s", exc)
        return []

    if not vector_results:
        return []

    # 3. Post-filter: memory_type="code" + optional entity_type/repo
    filtered: list[dict[str, Any]] = []
    for hit in vector_results:
        meta = hit.get("metadata", {})
        if meta.get("memory_type") != "code":
            continue
        if entity_type and meta.get("entity_type") != entity_type:
            continue
        if repo and meta.get("repo") != repo:
            continue
        filtered.append(hit)
        if len(filtered) >= top_k:
            break

    # 4. Hydrate from graph and build response dicts
    results: list[dict[str, Any]] = []
    for hit in filtered:
        item_id = hit.get("id")
        if not item_id:
            continue
        node = graph.get_node(item_id)
        if node is None:
            # Node deleted since embedding was written — skip silently
            continue

        # Extract properties — node can be a MemoryItem or raw dict
        props: dict[str, Any]
        if hasattr(node, "metadata"):
            # MemoryItem: code fields live in metadata (not system fields)
            props = {"item_id": getattr(node, "item_id", item_id)}
            for key in _CODE_FIELDS:
                if key == "item_id":
                    continue
                props[key] = getattr(node, key, None) or node.metadata.get(key, "")
        elif isinstance(node, dict):
            props = node
        else:
            props = {"item_id": item_id}

        result = {k: props.get(k, "") for k in _CODE_FIELDS}
        result["line_number"] = int(result.get("line_number") or 0)
        result["score"] = hit.get("score", 0.0)
        results.append(result)

    return results
