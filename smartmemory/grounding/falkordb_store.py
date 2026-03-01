"""FalkorDB-backed PublicKnowledgeStore for service mode.

Uses a dedicated global graph ('__public_knowledge') separate from
per-workspace ontology and data graphs. Version tracking via Redis INCR.

Follows the OntologyGraph._get_backend() lazy-load pattern at
ontology_graph.py:60-71.
"""

from __future__ import annotations

import json
import logging

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.store import PublicKnowledgeStore

logger = logging.getLogger(__name__)

_VERSION_KEY = "smartmemory:public_knowledge:version"


class FalkorDBPublicKnowledgeStore(PublicKnowledgeStore):
    """PublicKnowledgeStore backed by FalkorDB with a dedicated global graph.

    All surface forms are stored lowercased for case-insensitive matching.
    Version tracking uses Redis INCR for cross-process visibility.
    """

    def __init__(self, backend=None, redis_client=None):
        self._backend = backend
        self._redis = redis_client
        self._local_version = 0
        self._snapshot_version_str = ""

    def _get_backend(self):
        """Lazy-load or return injected backend."""
        if self._backend is not None:
            return self._backend
        from smartmemory.graph.backends.falkordb import FalkorDBBackend

        self._backend = FalkorDBBackend(graph_name="__public_knowledge")
        return self._backend

    def _get_redis(self):
        """Lazy-load Redis client."""
        if self._redis is not None:
            return self._redis
        try:
            import redis
            import os

            self._redis = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "9012")),
                decode_responses=True,
            )
            return self._redis
        except Exception:
            logger.debug("Redis unavailable, using local version counter")
            return None

    def lookup_by_alias(self, surface_form: str) -> list[PublicEntity]:
        backend = self._get_backend()
        query = """
        MATCH (a:Alias)-[:REFERS_TO]->(e:PublicEntity)
        WHERE a.surface_form = toLower($sf)
        RETURN e.qid AS qid, e.label AS label, e.description AS description,
               e.entity_type AS entity_type, e.instance_of AS instance_of,
               e.domain AS domain, e.confidence AS confidence
        """
        try:
            rows = backend.query(query, {"sf": surface_form})
            entities = []
            for record in rows:
                entities.append(PublicEntity(
                    qid=record[0],
                    label=record[1],
                    description=record[2] or "",
                    entity_type=record[3] or "",
                    instance_of=json.loads(record[4]) if record[4] else [],
                    domain=record[5] or "",
                    confidence=float(record[6]) if record[6] else 1.0,
                ))
            return entities
        except Exception as e:
            logger.error("FalkorDB lookup_by_alias failed: %s", e)
            return []

    def lookup_by_qid(self, qid: str) -> PublicEntity | None:
        backend = self._get_backend()
        query = """
        MATCH (e:PublicEntity {qid: $qid})
        RETURN e.qid AS qid, e.label AS label, e.description AS description,
               e.entity_type AS entity_type, e.instance_of AS instance_of,
               e.domain AS domain, e.confidence AS confidence
        """
        try:
            rows = backend.query(query, {"qid": qid})
            if not rows:
                return None
            record = rows[0]
            return PublicEntity(
                qid=record[0],
                label=record[1],
                description=record[2] or "",
                entity_type=record[3] or "",
                instance_of=json.loads(record[4]) if record[4] else [],
                domain=record[5] or "",
                confidence=float(record[6]) if record[6] else 1.0,
            )
        except Exception as e:
            logger.error("FalkorDB lookup_by_qid failed: %s", e)
            return None

    def absorb(self, entity: PublicEntity) -> None:
        backend = self._get_backend()
        instance_of_json = json.dumps(entity.instance_of)

        # MERGE entity node
        backend.query(
            """
            MERGE (e:PublicEntity {qid: $qid})
            SET e.label = $label,
                e.description = $description,
                e.entity_type = $entity_type,
                e.instance_of = $instance_of,
                e.domain = $domain,
                e.confidence = $confidence
            """,
            {
                "qid": entity.qid,
                "label": entity.label,
                "description": entity.description,
                "entity_type": entity.entity_type,
                "instance_of": instance_of_json,
                "domain": entity.domain,
                "confidence": entity.confidence,
            },
        )

        # Label-as-alias invariant: primary label is always an alias
        all_surface_forms = set(entity.aliases)
        all_surface_forms.add(entity.label)
        for sf in all_surface_forms:
            backend.query(
                """
                MATCH (e:PublicEntity {qid: $qid})
                MERGE (a:Alias {surface_form: toLower($sf)})
                MERGE (a)-[:REFERS_TO]->(e)
                """,
                {"qid": entity.qid, "sf": sf},
            )

        # Increment version
        r = self._get_redis()
        if r:
            try:
                r.incr(_VERSION_KEY)
            except Exception:
                self._local_version += 1
        else:
            self._local_version += 1

    def get_ruler_patterns(self) -> dict[str, str]:
        backend = self._get_backend()
        query = """
        MATCH (a:Alias)-[:REFERS_TO]->(e:PublicEntity)
        WHERE e.entity_type <> ''
        RETURN a.surface_form AS surface_form, e.entity_type AS entity_type,
               e.confidence AS confidence, e.qid AS qid
        ORDER BY a.surface_form, e.confidence DESC, toInteger(substring(e.qid, 1)) ASC
        """
        try:
            rows = backend.query(query)
            patterns: dict[str, str] = {}
            for record in rows:
                sf = record[0]  # already lowercased on write
                if sf not in patterns:
                    patterns[sf] = record[1]
            return patterns
        except Exception as e:
            logger.error("FalkorDB get_ruler_patterns failed: %s", e)
            return {}

    def count(self) -> int:
        backend = self._get_backend()
        try:
            rows = backend.query("MATCH (e:PublicEntity) RETURN count(e)")
            return rows[0][0] if rows else 0
        except Exception:
            return 0

    def snapshot_version(self) -> str:
        return self._snapshot_version_str

    @property
    def version(self) -> int:
        r = self._get_redis()
        if r:
            try:
                val = r.get(_VERSION_KEY)
                return int(val) if val else 0
            except Exception:
                pass
        return self._local_version
