from typing import Dict, List, Optional
import math

from falkordb import FalkorDB

from smartmemory.utils import get_config
from .base import VectorBackend


class FalkorVectorBackend(VectorBackend):
    """
    FalkorDB-backed vector store (Phase 1):
    - Stores embeddings JSON-encoded to avoid Cypher array issues
    - Performs cosine similarity on the app side over a bounded candidate set
    - Pushes only collection label filtering to the DB (tenancy filters remain in VectorStore)

    Phase 2 will switch to native vector indexing/KNN when available.
    """

    def __init__(self, collection_name: str, persist_directory: Optional[str]):  # persist_directory unused
        if FalkorDB is None:
            raise RuntimeError("falkordb package is not installed but vector backend=falkordb")
        graph_cfg = get_config("graph_db") or {}
        # graph_cfg is a ValidatedConfigDict; support dict access too for safety
        host = getattr(graph_cfg, "host", None) or (graph_cfg.get("host") if isinstance(graph_cfg, dict) else None) or "localhost"
        port = getattr(graph_cfg, "port", None) or (graph_cfg.get("port") if isinstance(graph_cfg, dict) else None) or 9010
        graph_name = getattr(graph_cfg, "graph_name", None) or (graph_cfg.get("graph_name") if isinstance(graph_cfg, dict) else None) or "smartmemory"
        self.db = FalkorDB(host=host, port=port)
        self.graph = self.db.select_graph(graph_name)
        # Use a label derived from collection to isolate per-collection vectors
        # Sanitize to alphanumeric + underscore
        safe = "".join(ch if ch.isalnum() else "_" for ch in collection_name)
        self.label = f"Vec_{safe}" if not safe.startswith("Vec_") else safe

        # Vector config
        vector_cfg = get_config("vector") or {}
        self.dimension = (
                getattr(vector_cfg, "dimension", None)
                or (vector_cfg.get("dimension") if isinstance(vector_cfg, dict) else None)
                or 1536
        )
        self.metric = (
                getattr(vector_cfg, "metric", None)
                or (vector_cfg.get("metric") if isinstance(vector_cfg, dict) else None)
                or "cosine"
        )
        self.hnsw_m = (
                getattr(vector_cfg, "hnsw_m", None)
                or (vector_cfg.get("hnsw_m") if isinstance(vector_cfg, dict) else None)
                or 16
        )
        self.hnsw_ef_construction = (
                getattr(vector_cfg, "hnsw_ef_construction", None)
                or (vector_cfg.get("hnsw_ef_construction") if isinstance(vector_cfg, dict) else None)
                or 200
        )
        # efRuntime is used at query time by the index implementation; exposed here for index options
        self.hnsw_ef_runtime = (
                getattr(vector_cfg, "hnsw_ef_runtime", None)
                or (vector_cfg.get("hnsw_ef_runtime") if isinstance(vector_cfg, dict) else None)
                or 64
        )

        # Always attempt to create vector index (idempotency depends on server; ignore errors)
        options = [
            f"dimension: {int(self.dimension)}",
            f"similarityFunction: '{self.metric}'",
        ]
        if self.hnsw_m is not None:
            options.append(f"M: {int(self.hnsw_m)}")
        if self.hnsw_ef_construction is not None:
            options.append(f"efConstruction: {int(self.hnsw_ef_construction)}")
        if self.hnsw_ef_runtime is not None:
            options.append(f"efRuntime: {int(self.hnsw_ef_runtime)}")
        ddl = (
            f"CREATE VECTOR INDEX FOR (n:{self.label}) ON (n.embedding) "
            f"OPTIONS {{{', '.join(options)}}}"
        )
        try:
            self.graph.query(ddl)
        except Exception:
            # Let add/search fail later if server truly lacks support
            pass
            
        # Create full-text index for keyword search
        ft_ddl = f"CREATE FULLTEXT INDEX FOR (n:{self.label}) ON EACH [n.content, n.name, n.id]"
        try:
            self.graph.query(ft_ddl)
        except Exception:
            pass

    # ---------- Helpers ----------
    @staticmethod
    def _to_scalar_props(meta: Dict) -> Dict:
        out: Dict = {}
        for k, v in (meta or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
        return out

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b, strict=False):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0 or nb == 0:
            return -1.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    # ---------- CRUD ----------
    def add(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        self._upsert_impl(item_id=item_id, embedding=embedding, metadata=metadata)

    def upsert(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        self._upsert_impl(item_id=item_id, embedding=embedding, metadata=metadata)

    def _upsert_impl(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        scalars = self._to_scalar_props(metadata)
        # Remove 'item_id' from scalars since it's already in the MERGE clause
        scalars.pop('item_id', None)
        scalars.pop('id', None)  # Also remove 'id' to avoid conflicts
        
        # Convert numpy array to list if needed (FalkorDB vecf32 requires list)
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        params: Dict = {"item_id": item_id, "embedding": embedding}
        # Ensure vector literal creation via vecf32($embedding)
        set_parts = ["n.embedding = vecf32($embedding)"]
        for k, v in scalars.items():
            # Escape property names with backticks to avoid reserved word conflicts
            set_parts.append(f"n.`{k}` = $prop_{k}")
            params[f"prop_{k}"] = v
        set_clause = ", ".join(set_parts)
        query = f"MERGE (n:{self.label} {{id: $item_id}}) SET {set_clause}"
        self.graph.query(query, params)

    # ---------- Query ----------
    def search(self, *, query_embedding: List[float], top_k: int) -> List[Dict]:
        # Use index proc for vector KNN search
        q = (
            "CALL db.idx.vector.queryNodes($label, $property, $k, vecf32($q)) "
            "YIELD node, score "
            "RETURN node.id AS id, node AS node, score "
            "ORDER BY score DESC"
        )
        params = {"label": self.label, "property": "embedding", "k": int(top_k), "q": query_embedding}
        res = self.graph.query(q, params)
        out: List[Dict] = []
        for row in getattr(res, "result_set", res) or []:
            try:
                cid = row[0]
                node_obj = row[1]
                score = row[2]
                if hasattr(node_obj, "properties"):
                    meta = dict(node_obj.properties)
                else:
                    meta = {k: v for k, v in vars(node_obj).items() if not k.startswith("_") and k != "properties"}
                meta.pop("embedding", None)
                out.append({"id": cid, "metadata": meta, "score": score})
            except Exception:
                continue
        return out

    def search_by_text(self, *, query_text: str, top_k: int) -> List[Dict]:
        """Full-text search using FalkorDB's fulltext index."""
        # Sanitize query text (simple escape)
        safe_query = query_text.replace("'", "\\'")
        
        # Use CALL db.idx.fulltext.queryNodes
        q = (
            "CALL db.idx.fulltext.queryNodes($label, $q) "
            "YIELD node, score "
            "RETURN node.id AS id, node AS node, score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )
        params = {"label": self.label, "q": safe_query, "k": int(top_k) * 2}
        
        try:
            res = self.graph.query(q, params)
            out: List[Dict] = []
            for row in getattr(res, "result_set", res) or []:
                try:
                    cid = row[0]
                    node_obj = row[1]
                    score = row[2]
                    if hasattr(node_obj, "properties"):
                        meta = dict(node_obj.properties)
                    else:
                        meta = {k: v for k, v in vars(node_obj).items() if not k.startswith("_") and k != "properties"}
                    meta.pop("embedding", None)
                    out.append({"id": cid, "metadata": meta, "score": score})
                except Exception:
                    continue
            return out
        except Exception as e:
            print(f"Warning: Full-text search failed: {e}")
            return []

    # ---------- Maintenance ----------
    def clear(self) -> None:
        query = f"MATCH (n:{self.label}) DELETE n"
        try:
            self.graph.query(query)
        except Exception:
            # best-effort
            pass
