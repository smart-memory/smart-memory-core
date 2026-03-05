"""UsearchVectorBackend — zero-infra vector backend for SmartMemory Lite (DIST-LITE-1).

Storage layout (all under persist_directory/):
  {collection_name}.usearch  — usearch index file
  {collection_name}.json     — id_map, rev_map, metadata, next_key, dim
  fts.db                     — SQLite FTS5 for text search (one virtual table per collection)
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from smartmemory.stores.vector.backends.base import VectorBackend

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import usearch.index as _usearch_index

    _USEARCH_AVAILABLE = True
except ImportError:
    _USEARCH_AVAILABLE = False


class UsearchVectorBackend(VectorBackend):
    """Vector backend using usearch for ANN search and SQLite FTS5 for text search.

    - usearch.Index uses integer keys internally; this class maintains bidirectional
      maps between integer keys and string item_ids.
    - Maps, metadata, and index dimension are persisted to a companion JSON file so
      the backend can be reconstructed after process restart.
    - FTS5 is stored in a shared SQLite database (fts.db) under persist_directory.
      Each collection owns a dedicated virtual table (fts_{sanitized_collection_name})
      so that clear(), search_by_text(), and add() never affect sibling collections.
    """

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None) -> None:
        if not _USEARCH_AVAILABLE:
            raise ImportError(
                "usearch and numpy are required for UsearchVectorBackend. Install with: pip install usearch numpy"
            )
        self._collection_name = collection_name
        self._persist_dir: Optional[Path] = None
        if persist_directory is not None:
            self._persist_dir = Path(persist_directory).expanduser()
            self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._index_path: Optional[str] = (
            str(self._persist_dir / f"{collection_name}.usearch") if self._persist_dir else None
        )
        self._map_path: Optional[str] = (
            str(self._persist_dir / f"{collection_name}.json") if self._persist_dir else None
        )
        fts_path: str = str(self._persist_dir / "fts.db") if self._persist_dir else ":memory:"

        # Per-collection FTS table name — sanitize to valid SQL identifier characters.
        # Using a dedicated virtual table per collection prevents cross-collection data
        # leakage from search_by_text() and clear() operating on a shared table.
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", collection_name)
        self._fts_table: str = f"fts_{sanitized}"

        # In-memory state
        self._id_map: Dict[int, str] = {}  # int key  → item_id
        self._rev_map: Dict[str, int] = {}  # item_id  → int key
        self._metadata: Dict[str, Dict] = {}  # item_id  → metadata dict
        self._next_key: int = 0
        self._dim: Optional[int] = None
        self._index: Any = None  # usearch.index.Index or None

        self._fts_conn: sqlite3.Connection = self._init_fts(fts_path, self._fts_table)
        self._load_or_create()

    # ── FTS setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _init_fts(fts_path: str, table_name: str) -> sqlite3.Connection:
        conn = sqlite3.connect(fts_path, check_same_thread=False)
        conn.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table_name}" USING fts5(item_id UNINDEXED, content)'
        )
        conn.commit()
        return conn

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        """Load persisted state if it exists; otherwise start fresh."""
        if self._map_path is None or not Path(self._map_path).exists():
            return
        try:
            with open(self._map_path) as fh:
                state = json.load(fh)
            self._id_map = {int(k): v for k, v in state.get("id_map", {}).items()}
            self._rev_map = state.get("rev_map", {})
            self._metadata = state.get("metadata", {})
            self._next_key = state.get("next_key", len(self._id_map))
            self._dim = state.get("dim")
            if self._dim and self._index_path and Path(self._index_path).exists():
                self._index = _usearch_index.Index(ndim=self._dim, metric="cos")
                self._index.load(self._index_path)
        except Exception as exc:
            logger.warning("UsearchVectorBackend: failed to load persisted state — starting fresh. %s", exc)
            self._id_map = {}
            self._rev_map = {}
            self._metadata = {}
            self._next_key = 0
            self._dim = None
            self._index = None

    def _save(self) -> None:
        """Flush in-memory state and usearch index to disk."""
        if self._map_path is not None:
            state = {
                "id_map": {str(k): v for k, v in self._id_map.items()},
                "rev_map": self._rev_map,
                "metadata": self._metadata,
                "next_key": self._next_key,
                "dim": self._dim,
            }
            with open(self._map_path, "w") as fh:
                json.dump(state, fh)
        if self._index is not None and self._index_path is not None:
            self._index.save(self._index_path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_index(self, dim: int) -> None:
        """Lazily create the usearch index on first add."""
        if self._index is None:
            self._dim = dim
            self._index = _usearch_index.Index(ndim=dim, metric="cos")

    def _fts_upsert(self, item_id: str, content: str) -> None:
        self._fts_conn.execute(f'DELETE FROM "{self._fts_table}" WHERE item_id = ?', (item_id,))
        self._fts_conn.execute(
            f'INSERT INTO "{self._fts_table}"(item_id, content) VALUES (?, ?)', (item_id, content)
        )
        self._fts_conn.commit()

    def _fts_delete(self, item_id: str) -> None:
        self._fts_conn.execute(f'DELETE FROM "{self._fts_table}" WHERE item_id = ?', (item_id,))
        self._fts_conn.commit()

    def _remove_from_index(self, item_id: str) -> None:
        """Remove an item from all in-memory structures (does not save)."""
        old_key = self._rev_map.pop(item_id, None)
        if old_key is not None:
            self._id_map.pop(old_key, None)
            if self._index is not None:
                try:
                    self._index.remove(old_key)
                except Exception:
                    pass
        self._metadata.pop(item_id, None)

    # ── VectorBackend abstract methods ─────────────────────────────────────────

    def add(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        """Add a new item. Overwrites silently if item_id already exists."""
        vec = np.array(embedding, dtype=np.float32)
        self._ensure_index(len(embedding))
        # Remove stale entry if present (idempotent add)
        self._remove_from_index(item_id)

        key = self._next_key
        self._next_key += 1
        self._id_map[key] = item_id
        self._rev_map[item_id] = key
        meta = dict(metadata) if metadata else {}
        # Store embedding in metadata so get() can return it (usearch doesn't
        # expose stored vectors). Stored as list[float] for JSON serialization.
        meta["_embedding"] = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        self._metadata[item_id] = meta
        assert self._index is not None
        self._index.add(key, vec)

        if metadata and "content" in metadata:
            self._fts_upsert(item_id, str(metadata["content"]))

        self._save()

    def upsert(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        """Insert or replace an item by item_id."""
        self.add(item_id=item_id, embedding=embedding, metadata=metadata)

    def get(self, item_id: str) -> Optional[Dict]:
        """Retrieve a stored item by id, including its embedding vector."""
        if item_id not in self._rev_map:
            return None
        meta = self._metadata.get(item_id, {})
        embedding = meta.get("_embedding")
        # Return in the same shape that SSG/VectorStore expects
        result: Dict[str, Any] = {"id": item_id, "metadata": {k: v for k, v in meta.items() if k != "_embedding"}}
        if embedding is not None:
            result["embedding"] = embedding
        return result

    def search(self, *, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Return up to top_k nearest neighbours by cosine similarity."""
        if self._index is None or not self._id_map:
            return []
        vec = np.array(query_embedding, dtype=np.float32)
        k = min(top_k, len(self._id_map))
        results = self._index.search(vec, k)
        hits: List[Dict] = []
        for key, distance in zip(results.keys, results.distances, strict=False):
            item_id = self._id_map.get(int(key))
            if item_id is not None:
                hits.append(
                    {
                        "id": item_id,
                        "score": float(1.0 - distance),  # cosine distance → similarity
                        "metadata": self._metadata.get(item_id, {}),
                    }
                )
        return hits

    def search_by_text(self, *, query_text: str, top_k: int) -> List[Dict]:
        """Full-text search via SQLite FTS5. Returns dicts with at least 'id'."""
        try:
            rows = self._fts_conn.execute(
                f'SELECT item_id FROM "{self._fts_table}" WHERE "{self._fts_table}" MATCH ? LIMIT ?',
                (query_text, top_k),
            ).fetchall()
            return [{"id": row[0]} for row in rows]
        except Exception as exc:
            logger.warning("UsearchVectorBackend: FTS search failed: %s", exc)
            return []

    def clear(self) -> None:
        """Remove all items from the index and FTS database."""
        self._id_map = {}
        self._rev_map = {}
        self._metadata = {}
        self._next_key = 0
        self._index = None
        self._dim = None

        # Remove persisted files
        for path in (self._index_path, self._map_path):
            if path is not None:
                p = Path(path)
                if p.exists():
                    p.unlink()

        # Clear only this collection's FTS rows (other collections have their own table)
        self._fts_conn.execute(f'DELETE FROM "{self._fts_table}"')
        self._fts_conn.commit()

        # Write empty state file so _next_key is preserved at 0
        self._save()
