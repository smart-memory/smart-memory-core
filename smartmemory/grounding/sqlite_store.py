"""SQLite-backed PublicKnowledgeStore for local/bundled snapshots.

Uses WAL mode for concurrent reads and a threading lock for write safety.
This is the default store for local/CLI usage where FalkorDB is unavailable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.store import PublicKnowledgeStore

logger = logging.getLogger(__name__)


class SQLitePublicKnowledgeStore(PublicKnowledgeStore):
    """PublicKnowledgeStore backed by a local SQLite database.

    Thread-safe via threading.Lock on all writes. Reads are concurrent
    thanks to WAL mode.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._version_counter = 0
        self._snapshot_version = ""
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                qid TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT DEFAULT '',
                entity_type TEXT DEFAULT '',
                instance_of TEXT DEFAULT '[]',
                domain TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS aliases (
                surface_form TEXT NOT NULL COLLATE NOCASE,
                qid TEXT NOT NULL,
                PRIMARY KEY (surface_form, qid),
                FOREIGN KEY (qid) REFERENCES entities(qid)
            );

            CREATE INDEX IF NOT EXISTS idx_aliases_surface
                ON aliases(surface_form COLLATE NOCASE);

            CREATE TABLE IF NOT EXISTS snapshot_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self._conn.commit()

    def lookup_by_alias(self, surface_form: str) -> list[PublicEntity]:
        cursor = self._conn.execute(
            """
            SELECT e.qid, e.label, e.description, e.entity_type,
                   e.instance_of, e.domain, e.confidence
            FROM aliases a
            JOIN entities e ON a.qid = e.qid
            WHERE a.surface_form = ? COLLATE NOCASE
            """,
            (surface_form,),
        )
        results = []
        for row in cursor:
            # Collect all aliases for this entity
            alias_cursor = self._conn.execute(
                "SELECT surface_form FROM aliases WHERE qid = ?", (row["qid"],)
            )
            aliases = [r["surface_form"] for r in alias_cursor if r["surface_form"] != row["label"]]
            results.append(PublicEntity(
                qid=row["qid"],
                label=row["label"],
                description=row["description"],
                entity_type=row["entity_type"],
                instance_of=json.loads(row["instance_of"]),
                domain=row["domain"],
                confidence=row["confidence"],
                aliases=aliases,
            ))
        return results

    def lookup_by_qid(self, qid: str) -> PublicEntity | None:
        cursor = self._conn.execute(
            """
            SELECT qid, label, description, entity_type,
                   instance_of, domain, confidence
            FROM entities WHERE qid = ?
            """,
            (qid,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        alias_cursor = self._conn.execute(
            "SELECT surface_form FROM aliases WHERE qid = ?", (qid,)
        )
        aliases = [r["surface_form"] for r in alias_cursor if r["surface_form"] != row["label"]]
        return PublicEntity(
            qid=row["qid"],
            label=row["label"],
            description=row["description"],
            entity_type=row["entity_type"],
            instance_of=json.loads(row["instance_of"]),
            domain=row["domain"],
            confidence=row["confidence"],
            aliases=aliases,
        )

    def absorb(self, entity: PublicEntity) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entities
                    (qid, label, description, entity_type, instance_of, domain, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.qid,
                    entity.label,
                    entity.description,
                    entity.entity_type,
                    json.dumps(entity.instance_of),
                    entity.domain,
                    entity.confidence,
                ),
            )
            # Remove old aliases for this QID and re-insert
            self._conn.execute("DELETE FROM aliases WHERE qid = ?", (entity.qid,))
            # Label-as-alias invariant: primary label is always an alias
            all_surface_forms = set(entity.aliases)
            all_surface_forms.add(entity.label)
            for sf in all_surface_forms:
                self._conn.execute(
                    "INSERT OR REPLACE INTO aliases (surface_form, qid) VALUES (?, ?)",
                    (sf, entity.qid),
                )
            self._conn.commit()
            self._version_counter += 1

    def get_ruler_patterns(self) -> dict[str, str]:
        """Return surface_form -> entity_type for EntityRuler.

        Deterministic: for ambiguous surface forms, pick the entity with
        highest confidence, tie-break by lowest QID.
        """
        cursor = self._conn.execute(
            """
            SELECT a.surface_form, e.entity_type, e.confidence, e.qid
            FROM aliases a
            JOIN entities e ON a.qid = e.qid
            WHERE e.entity_type != ''
            ORDER BY a.surface_form COLLATE NOCASE, e.confidence DESC, CAST(SUBSTR(e.qid, 2) AS INTEGER) ASC
            """
        )
        patterns: dict[str, str] = {}
        for row in cursor:
            sf_lower = row["surface_form"].lower()
            if sf_lower not in patterns:
                patterns[sf_lower] = row["entity_type"]
        return patterns

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
        return cursor.fetchone()[0]

    def snapshot_version(self) -> str:
        return self._snapshot_version

    @property
    def version(self) -> int:
        return self._version_counter

    def load_snapshot(self, jsonl_path: str) -> None:
        """Bulk load entities from a JSONL file."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {jsonl_path}")

        count = 0
        with self._lock:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entity = PublicEntity(
                        qid=data["qid"],
                        label=data["label"],
                        aliases=data.get("aliases", []),
                        description=data.get("description", ""),
                        entity_type=data.get("entity_type", ""),
                        instance_of=data.get("instance_of", []),
                        domain=data.get("domain", ""),
                        confidence=data.get("confidence", 1.0),
                    )
                    self._absorb_no_lock(entity)
                    count += 1
            self._conn.commit()
            self._version_counter += 1

        self._snapshot_version = path.name
        logger.info("Loaded %d entities from snapshot %s", count, jsonl_path)

    def _absorb_no_lock(self, entity: PublicEntity) -> None:
        """Internal absorb without lock (caller holds lock)."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO entities
                (qid, label, description, entity_type, instance_of, domain, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.qid,
                entity.label,
                entity.description,
                entity.entity_type,
                json.dumps(entity.instance_of),
                entity.domain,
                entity.confidence,
            ),
        )
        # Remove old aliases before re-inserting (prevent stale alias accumulation)
        self._conn.execute("DELETE FROM aliases WHERE qid = ?", (entity.qid,))
        # Label-as-alias invariant
        all_surface_forms = set(entity.aliases)
        all_surface_forms.add(entity.label)
        for sf in all_surface_forms:
            self._conn.execute(
                "INSERT OR REPLACE INTO aliases (surface_form, qid) VALUES (?, ?)",
                (sf, entity.qid),
            )

    def close(self) -> None:
        self._conn.close()
