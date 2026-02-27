import json
from contextvars import ContextVar
from datetime import datetime
from time import perf_counter
from typing import Any, Optional

from smartmemory.observability.tracing import trace_span
from smartmemory.stores.vector.backends.base import create_backend
from smartmemory.utils import get_config
from smartmemory.interfaces import ScopeProvider
from smartmemory.scope_provider import DefaultScopeProvider


_DEFAULT_BACKEND = None

# CORE-DI-1: per-call override via ContextVar (set by SmartMemory._di_context())
_vector_backend_ctx: ContextVar[Optional[Any]] = ContextVar("_vector_backend_ctx", default=None)


class VectorStore:
    """
    Backend-agnostic vector store for semantic memory.
    Supports add, upsert, search, delete, get, and clear operations with metadata.

    Each instance is independent - no singleton pattern for better testability and isolation.

    Example usage:
        vector_store = VectorStore(collection_name="my_collection")
        vector_store.upsert(
            item_id="item123",
            embedding=embedding,
            node_ids=["nodeA", "nodeB"],
            chunk_ids=["chunk3", "chunk7"],
            metadata={"source": "summary"}
        )
        vector_store.clear()  # Deletes all vectors from the collection
    """

    # trace_span replaces the old VEC_EMIT = make_emitter(...) pattern

    @classmethod
    def set_default_backend(cls, backend) -> None:
        """Override the backend for all new VectorStore instances.

        Pass None to restore normal backend resolution from config.
        Intended for testing (DIST-LITE-1 P0-4) — not for production use.
        """
        global _DEFAULT_BACKEND
        _DEFAULT_BACKEND = backend

    def _vec_data(
        self,
        *,
        item_id=None,
        embedding=None,
        meta=None,
        top_k=None,
        returned=None,
        deleted_count=None,
        t0=None,
    ):
        """Build a standard payload for vector operations, omitting None values.

        This centralizes repeated logic like computing dim, counts, collection name,
        and duration, so call sites can pass only what's relevant.
        """

        def _count(val):
            if isinstance(val, list):
                return len(val)
            return 1 if val else 0

        node_ids = meta.get("node_ids") if isinstance(meta, dict) else None
        chunk_ids = meta.get("chunk_ids") if isinstance(meta, dict) else None

        duration_ms = None
        if t0 is not None:
            try:
                duration_ms = (perf_counter() - t0) * 1000.0
            except Exception:
                duration_ms = None

        data = {
            "collection": getattr(self.collection, "name", "unknown"),
            "id": str(item_id) if item_id is not None else None,
            "dim": (len(embedding) if hasattr(embedding, "__len__") else None) if embedding is not None else None,
            "node_ids_count": _count(node_ids) if node_ids is not None else None,
            "chunk_ids_count": _count(chunk_ids) if chunk_ids is not None else None,
            "top_k": top_k,
            "returned": returned,
            "deleted_count": deleted_count,
            "duration_ms": duration_ms,
        }

        # Omit None entries
        return {k: v for k, v in data.items() if v is not None}

    def __init__(self, collection_name=None, persist_directory=None, scope_provider: Optional[ScopeProvider] = None):
        """Construct a VectorStore that delegates to a configured backend."""
        self.scope_provider = scope_provider or DefaultScopeProvider()

        # CORE-DI-1: per-instance ContextVar override (highest priority)
        ctx_backend = _vector_backend_ctx.get()
        if ctx_backend is not None:
            self._backend = ctx_backend
            self._collection_name = collection_name or "default"
            self.collection = self._backend
            return

        # Process-global override (backward compat, lower priority)
        if _DEFAULT_BACKEND is not None:
            self._backend = _DEFAULT_BACKEND
            self._collection_name = collection_name or "default"
            self.collection = self._backend
            return  # skip all normal resolution

        vector_cfg = get_config("vector") or {}
        full_cfg = get_config()

        # Resolve effective collection name with namespace support
        ns = full_cfg.get("active_namespace") if isinstance(full_cfg, dict) else None
        use_ws_ns = bool(vector_cfg.get("use_workspace_namespace", False))

        # Use scope provider to get workspace context if needed for namespacing
        ws = None
        if use_ws_ns:
            ctx = self.scope_provider.get_isolation_filters()
            ws = ctx.get("workspace_id")

        base_collection = collection_name or vector_cfg.get("collection_name") or "semantic_memory"
        eff_collection = base_collection
        if ns:
            eff_collection = f"{eff_collection}_{ns}"
        if ws:
            eff_collection = f"{eff_collection}_{ws}"

        # Resolve persist directory (unused for FalkorDB but kept for API compatibility)
        if persist_directory is None:
            persist_directory = vector_cfg.get("persist_directory")

        backend_name = (vector_cfg.get("backend") or "falkordb").lower()
        self._backend = create_backend(backend_name, eff_collection, persist_directory)
        self._collection_name = eff_collection
        # Expose collection for compatibility with _vec_data method
        self.collection = self._backend

    def add(self, item_id, embedding, metadata=None, node_ids=None, chunk_ids=None, is_global=False):
        """
        Add an embedding to the vector store. Supports cross-referencing multiple node_ids and chunk_ids.
        - item_id: unique vector entry ID (string)
        - embedding: list of floats
        - metadata: dict of additional metadata
        - node_ids: single string or list of graph node IDs
        - chunk_ids: single string or list of chunk IDs
        - is_global: if True, skip scope_provider context injection

        Filtering is handled automatically by ScopeProvider.
        All IDs are stored in metadata as lists for robust cross-referencing.
        """
        meta = metadata.copy() if metadata else {}
        meta["item_id"] = str(item_id)

        # Inject write context from provider
        if not is_global:
            write_ctx = self.scope_provider.get_write_context()
            meta.update(write_ctx)

        # Flatten 'properties' dict if present
        node = meta.pop("_node", None)
        if node:
            properties = node.pop("properties", None)
            if properties and isinstance(properties, dict):
                for k, v in properties.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif isinstance(v, datetime):
                        meta[k] = v.isoformat()
                    else:
                        meta[k] = json.dumps(v)
        # Normalize node_ids and chunk_ids to lists
        if node_ids is not None:
            meta["node_ids"] = node_ids if isinstance(node_ids, list) else [node_ids]
        if chunk_ids is not None:
            meta["chunk_ids"] = chunk_ids if isinstance(chunk_ids, list) else [chunk_ids]

        # Ensure all metadata values are storage-compatible (str, int, float, bool)
        def storage_safe(val):
            if isinstance(val, (list, dict)):
                return json.dumps(val, default=str)
            if isinstance(val, datetime):
                return val.isoformat()
            return val

        meta = {k: storage_safe(v) for k, v in meta.items() if v is not None}
        meta = {k: v for k, v in meta.items() if v is not None}

        t0 = perf_counter()
        self._backend.add(item_id=str(item_id), embedding=embedding, metadata=meta)
        with trace_span("vector.add", self._vec_data(item_id=item_id, embedding=embedding, meta=meta, t0=t0)):
            pass

    def upsert(self, item_id, embedding, metadata=None, node_ids=None, chunk_ids=None, is_global=False):
        """
        Upsert an embedding to the vector store. Overwrites if the id exists, inserts if not.
        - is_global: if True, skip scope_provider context injection

        Filtering is handled automatically by ScopeProvider.
        """
        meta = metadata.copy() if metadata else {}

        # Inject write context from provider
        if not is_global:
            write_ctx = self.scope_provider.get_write_context()
            meta.update(write_ctx)

        if node_ids is not None:
            meta["node_ids"] = node_ids if isinstance(node_ids, list) else [node_ids]
        if chunk_ids is not None:
            meta["chunk_ids"] = chunk_ids if isinstance(chunk_ids, list) else [chunk_ids]

        # Ensure all metadata values are storage-compatible (str, int, float, bool)
        def storage_safe(val):
            if isinstance(val, (list, dict)):
                return json.dumps(val, default=str)
            if isinstance(val, datetime):
                return val.isoformat()
            return val

        meta = {k: storage_safe(v) for k, v in meta.items() if v is not None}
        meta = {k: v for k, v in meta.items() if v is not None}

        t0 = perf_counter()
        self._backend.upsert(item_id=str(item_id), embedding=embedding, metadata=meta)
        with trace_span("vector.upsert", self._vec_data(item_id=item_id, embedding=embedding, meta=meta, t0=t0)):
            pass

    def get(self, item_id, include_metadata: bool = True):
        """
        Fetch a single vector item by id. Returns a dict with keys:
        {"id", "embedding" (if backend provides), "metadata", "node_ids", "is_global"} when available,
        or None if not found or not supported by backend.
        """
        getter = getattr(self._backend, "get", None)
        if callable(getter):
            try:
                # Backend-specific signature may vary; prefer a simple get by id
                res = getter(item_id)
                if isinstance(res, list):
                    # Some backends return list; take first
                    res = res[0] if res else None
                return res
            except Exception as e:
                print(f"Warning: Vector backend get failed: {e}")
                return None
        # Not supported by backend
        return None

    def delete(self, item_id) -> bool:
        """
        Delete a single vector item by id. Returns True if deletion was attempted and backend acknowledged,
        False if not supported or failed.
        """
        deleter = getattr(self._backend, "delete", None)
        if callable(deleter):
            try:
                deleter(item_id)
                with trace_span("vector.delete", self._vec_data(item_id=item_id)):
                    pass
                return True
            except Exception as e:
                print(f"Warning: Vector backend delete failed: {e}")
                return False
        return False

    def search(self, query_embedding, top_k=5, is_global=False, query_text=None, rrf_k=60):
        """
        Search the vector store with automatic scope filtering.
        Supports hybrid retrieval if query_text is provided.

        Args:
            query_embedding: query vector
            top_k: number of results to return
            is_global: if True, return all results. Otherwise, filter by ScopeProvider.
            query_text: optional text query for hybrid search (BM25)
            rrf_k: constant for Reciprocal Rank Fusion (default 60)
        """
        t0 = perf_counter()

        # 1. Vector Search
        vector_results = self._backend.search(query_embedding=query_embedding, top_k=top_k * 2)

        backend_results = vector_results

        # 2. Hybrid Search (if query_text provided)
        if query_text and hasattr(self._backend, "search_by_text"):
            text_results = self._backend.search_by_text(query_text=query_text, top_k=top_k * 2)

            # Perform Reciprocal Rank Fusion (RRF)
            # score = 1 / (k + rank)
            rrf_scores = {}
            doc_map = {}  # Keep track of full doc objects

            for rank, res in enumerate(vector_results):
                doc_id = res["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
                doc_map[doc_id] = res

            for rank, res in enumerate(text_results):
                doc_id = res["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
                if doc_id not in doc_map:
                    doc_map[doc_id] = res

            # Sort by RRF score
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

            # Reconstruct backend_results
            backend_results = []
            for doc_id in sorted_ids:
                doc = doc_map[doc_id]
                doc["score"] = rrf_scores[doc_id]  # Replace score with RRF score
                backend_results.append(doc)

        hits = []
        count = 0

        # Get isolation filters from provider (single source of truth)
        # Provider returns field names and values without core library knowing what they mean
        filters = {} if is_global else self.scope_provider.get_isolation_filters()

        for i, res in enumerate(backend_results):
            id_ = res.get("id")
            meta = res.get("metadata", {})

            # Apply all isolation filters generically
            skip = False
            for filter_key, filter_value in filters.items():
                if filter_value is not None and meta.get(filter_key) != filter_value:
                    skip = True
                    break
            if skip:
                continue

            hit = {"id": id_}
            hit["metadata"] = self.deserialize_metadata(meta)
            if "score" in res:
                hit["score"] = res["score"]
            hits.append(hit)
            count += 1
            if count >= top_k:
                break
        with trace_span("vector.search", self._vec_data(top_k=top_k, returned=len(hits), t0=t0)):
            pass
        return hits

    def clear(self):
        """
        Delete all embeddings from the vector store collection.
        This operation is idempotent and safe. Useful for tests or resetting state.
        """
        try:
            t0 = perf_counter()
            # Delegate to backend; not all backends return a count
            self._backend.clear()
            with trace_span("vector.clear", self._vec_data(deleted_count=None, t0=t0)):
                pass
        except Exception as e:
            print(f"Warning: Vector store clear encountered error: {e}")

    @staticmethod
    def deserialize_metadata(meta):
        """
        Convert metadata values back to their original types if possible.
        Attempts to json.loads strings that look like lists/dicts, and parse ISO8601 dates.
        """
        import json
        from dateutil.parser import parse as parse_date

        def try_load(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    try:
                        return parse_date(val)
                    except Exception:
                        return val
            return val

        return {k: try_load(v) for k, v in meta.items()}
