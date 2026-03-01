"""OntologyGraph — manages entity type definitions in a separate FalkorDB graph.

Stores entity types with three-tier status: seed → provisional → confirmed.
Separate from the data graph so ontology reads never contend with data writes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Seed entity types shipped with SmartMemory
SEED_TYPES: List[str] = [
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Event",
    "Location",
    "Document",
    "Tool",
    "Skill",
    "Decision",
    "Claim",
    "Action",
    "Metric",
    "Process",
]

VALID_STATUSES = {"seed", "provisional", "confirmed", "rejected"}


# Seed relation types derived from canonical relation vocabulary (ONTO-PUB-3)
def _load_seed_relation_types() -> list[str]:
    try:
        from smartmemory.relations.schema import CANONICAL_RELATION_TYPES

        return list(CANONICAL_RELATION_TYPES.keys())
    except ImportError:
        return []


SEED_RELATION_TYPES: list[str] = _load_seed_relation_types()


class OntologyGraph:
    """Manage entity type definitions in a dedicated FalkorDB graph.

    The ontology graph is named ``ws_{workspace_id}_ontology`` and stores
    ``EntityType`` nodes with ``name`` and ``status`` properties.

    Usage::

        og = OntologyGraph(workspace_id="acme")
        og.seed_types()   # idempotent — seeds 14 base types
        og.add_provisional("Metric")
        og.promote("Metric")
        types = og.get_entity_types()
    """

    def __init__(self, workspace_id: str = "default", backend=None):
        """
        Args:
            workspace_id: Workspace identifier for graph naming.
            backend: Optional FalkorDB backend instance. If None, resolved from config.
        """
        self.workspace_id = workspace_id
        self._graph_name = f"ws_{workspace_id}_ontology"
        self._backend = backend

    def _get_backend(self):
        """Lazy-load or return injected backend."""
        if self._backend is not None:
            return self._backend
        try:
            from smartmemory.graph.backends.falkordb import FalkorDBBackend

            self._backend = FalkorDBBackend(graph_name=self._graph_name)
            return self._backend
        except Exception as e:
            logger.error("Failed to initialize ontology graph backend: %s", e)
            raise

    def seed_types(self, types: Optional[List[str]] = None) -> int:
        """Seed initial entity types with status='seed'. Idempotent.

        Returns:
            Number of types actually created (skips existing).
        """
        types_to_seed = types or SEED_TYPES
        backend = self._get_backend()
        created = 0

        for type_name in types_to_seed:
            try:
                existing = self._query_type(backend, type_name)
                if existing:
                    continue
                self._create_type(backend, type_name, "seed")
                created += 1
            except Exception as e:
                logger.warning("Failed to seed type '%s': %s", type_name, e)

        logger.info("Seeded %d/%d entity types in %s", created, len(types_to_seed), self._graph_name)
        return created

    def get_entity_types(self) -> List[Dict[str, Any]]:
        """Return all entity types with their status."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (t:EntityType) RETURN t.name AS name, t.status AS status ORDER BY t.name",
                graph_name=self._graph_name,
            )
            return [{"name": row[0], "status": row[1]} for row in (result or [])]
        except Exception as e:
            logger.warning("Failed to query entity types: %s", e)
            return []

    def get_type_status(self, name: str) -> Optional[str]:
        """Return the status of a type, or None if not found."""
        backend = self._get_backend()
        existing = self._query_type(backend, name)
        if existing:
            return existing[0][1]  # status column
        return None

    def add_provisional(self, name: str) -> bool:
        """Add a new type with status='provisional'. No-op if already exists.

        Returns:
            True if created, False if already existed.
        """
        backend = self._get_backend()
        existing = self._query_type(backend, name)
        if existing:
            return False
        self._create_type(backend, name, "provisional")
        return True

    def promote(self, name: str) -> bool:
        """Promote a type to 'confirmed'. Returns False if type not found."""
        backend = self._get_backend()
        try:
            backend.query(
                "MATCH (t:EntityType {name: $name}) SET t.status = 'confirmed'",
                params={"name": name},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to promote type '%s': %s", name, e)
            return False

    def reject(self, name: str) -> bool:
        """Reject a type — sets status to 'rejected'. Returns False if type not found."""
        backend = self._get_backend()
        try:
            backend.query(
                "MATCH (t:EntityType {name: $name}) SET t.status = 'rejected'",
                params={"name": name},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to reject type '%s': %s", name, e)
            return False

    # ------------------------------------------------------------------ #
    # Frequency tracking & entity patterns (Phase 4)
    # ------------------------------------------------------------------ #

    def increment_frequency(self, name: str, confidence: float) -> None:
        """Atomically increment type frequency and update running avg confidence."""
        backend = self._get_backend()
        try:
            backend.query(
                "MERGE (t:EntityType {name: $name}) "
                "SET t.frequency = COALESCE(t.frequency, 0) + 1, "
                "t.avg_confidence = (COALESCE(t.avg_confidence, 0) * COALESCE(t.frequency, 0) + $conf) "
                "/ (COALESCE(t.frequency, 0) + 1)",
                params={"name": name, "conf": confidence},
                graph_name=self._graph_name,
            )
        except Exception as e:
            logger.warning("Failed to increment frequency for '%s': %s", name, e)

    def get_frequency(self, name: str) -> int:
        """Return the frequency count for a type, or 0 if not found."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (t:EntityType {name: $name}) RETURN COALESCE(t.frequency, 0)",
                params={"name": name},
                graph_name=self._graph_name,
            )
            if result and result[0]:
                return int(result[0][0])
        except Exception as e:
            logger.warning("Failed to get frequency for '%s': %s", name, e)
        return 0

    def get_all_frequencies(self) -> Dict[str, int]:
        """Return frequency counts for all types in a single query."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (t:EntityType) RETURN t.name, COALESCE(t.frequency, 0)",
                graph_name=self._graph_name,
            )
            return {row[0]: int(row[1]) for row in (result or [])}
        except Exception as e:
            logger.warning("Failed to get all frequencies: %s", e)
            return {}

    def get_type_assignments(self, entity_name: str) -> List[Dict]:
        """Query EntityPattern nodes for this entity name, return [{type, count}]."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (p:EntityPattern {name: $name})-[:IS_INSTANCE_OF]->(t:EntityType) "
                "RETURN t.name AS type, COALESCE(p.count, 1) AS count",
                params={"name": entity_name},
                graph_name=self._graph_name,
            )
            return [{"type": row[0], "count": int(row[1])} for row in (result or [])]
        except Exception as e:
            logger.warning("Failed to get type assignments for '%s': %s", entity_name, e)
            return []

    def add_entity_pattern(
        self,
        name: str,
        label: str,
        confidence: float,
        workspace_id: str | None = None,
        is_global: bool = False,
        source: str = "llm_discovery",
        initial_count: int = 1,
    ) -> bool:
        """Create an EntityPattern node linked to its EntityType via :IS_INSTANCE_OF.

        Args:
            initial_count: Starting count for new patterns. Use 2 for AST-validated
                code patterns to bypass the ``count >= 2`` discovery threshold in
                ``get_entity_patterns()``. Default 1 preserves the existing threshold
                for LLM-discovered patterns.

        Returns:
            True if created, False on error.
        """
        backend = self._get_backend()
        try:
            backend.query(
                "MERGE (t:EntityType {name: $label}) "
                "MERGE (p:EntityPattern {name: $name, label: $label}) "
                "ON CREATE SET p.confidence = $conf, p.workspace_id = $ws, "
                "p.is_global = $glob, p.source = $source, p.discovered_at = timestamp(), p.count = $initial_count "
                "ON MATCH SET p.count = COALESCE(p.count, 1) + 1, "
                "p.confidence = (COALESCE(p.confidence, 0) + $conf) / 2 "
                "MERGE (p)-[:IS_INSTANCE_OF]->(t)",
                params={
                    "name": name.lower(),
                    "label": label,
                    "conf": confidence,
                    "ws": workspace_id or self.workspace_id,
                    "glob": is_global,
                    "source": source,
                    "initial_count": initial_count,
                },
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to add entity pattern '%s' -> '%s': %s", name, label, e)
            return False

    def get_entity_patterns(self, workspace_id: str | None = None) -> List[Dict]:
        """Load all patterns for a workspace (including global patterns)."""
        backend = self._get_backend()
        ws = workspace_id or self.workspace_id
        try:
            result = backend.query(
                "MATCH (p:EntityPattern) "
                "WHERE (p.is_global = true OR p.workspace_id = $ws) "
                "AND COALESCE(p.count, 1) >= 2 "
                "RETURN p.name, p.label, COALESCE(p.confidence, 0.5), p.source",
                params={"ws": ws},
                graph_name=self._graph_name,
            )
            return [
                {"name": row[0], "label": row[1], "confidence": float(row[2]), "source": row[3]}
                for row in (result or [])
            ]
        except Exception as e:
            logger.warning("Failed to get entity patterns for workspace '%s': %s", ws, e)
            return []

    def seed_entity_patterns(self, patterns: Dict[str, str] | None = None) -> int:
        """Seed common entity patterns for seed types. Idempotent.

        Args:
            patterns: Optional dict of {entity_name: entity_type}. Defaults to common entities.

        Returns:
            Number of patterns created.
        """
        defaults: Dict[str, str] = {
            "python": "Technology",
            "javascript": "Technology",
            "typescript": "Technology",
            "react": "Technology",
            "docker": "Technology",
            "kubernetes": "Technology",
            "aws": "Organization",
            "google": "Organization",
            "microsoft": "Organization",
            "github": "Organization",
            "openai": "Organization",
        }
        to_seed = patterns or defaults
        created = 0
        for name, label in to_seed.items():
            if self.add_entity_pattern(name, label, 0.95, is_global=True, source="seed", initial_count=2):
                created += 1
        return created

    def get_pattern_stats(self) -> Dict[str, int]:
        """Return counts of patterns by source layer: {seed, promoted, llm_discovery}."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (p:EntityPattern) RETURN p.source, count(p) ORDER BY p.source",
                graph_name=self._graph_name,
            )
            stats: Dict[str, int] = {"seed": 0, "promoted": 0, "llm_discovery": 0}
            for row in result or []:
                source = row[0] or "llm_discovery"
                stats[source] = int(row[1])
            return stats
        except Exception as e:
            logger.warning("Failed to get pattern stats: %s", e)
            return {"seed": 0, "promoted": 0, "llm_discovery": 0}

    # ------------------------------------------------------------------ #
    # Pipeline config storage (Phase 5)
    # ------------------------------------------------------------------ #

    def save_pipeline_config(self, name: str, config_json: str, workspace_id: str, description: str = "") -> bool:
        """Save or update a named pipeline config as a PipelineConfig node.

        Returns:
            True if saved successfully.
        """
        backend = self._get_backend()
        try:
            backend.query(
                "MERGE (c:PipelineConfig {name: $name, workspace_id: $ws}) "
                "ON CREATE SET c.config_json = $cfg, c.description = $desc, "
                "c.created_at = timestamp(), c.updated_at = timestamp() "
                "ON MATCH SET c.config_json = $cfg, c.description = $desc, "
                "c.updated_at = timestamp()",
                params={"name": name, "ws": workspace_id, "cfg": config_json, "desc": description},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to save pipeline config '%s': %s", name, e)
            return False

    def get_pipeline_config(self, name: str, workspace_id: str) -> Dict[str, Any] | None:
        """Load a named pipeline config by name and workspace.

        Returns:
            Config dict with name, description, config_json, workspace_id, created_at, updated_at.
            None if not found.
        """
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (c:PipelineConfig {name: $name, workspace_id: $ws}) "
                "RETURN c.name, c.description, c.config_json, c.workspace_id, "
                "c.created_at, c.updated_at",
                params={"name": name, "ws": workspace_id},
                graph_name=self._graph_name,
            )
            if not result:
                return None
            row = result[0]
            return {
                "name": row[0],
                "description": row[1] or "",
                "config_json": row[2],
                "workspace_id": row[3],
                "created_at": str(row[4]) if row[4] else "",
                "updated_at": str(row[5]) if row[5] else "",
            }
        except Exception as e:
            logger.warning("Failed to get pipeline config '%s': %s", name, e)
            return None

    def list_pipeline_configs(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List all pipeline configs for a workspace."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (c:PipelineConfig {workspace_id: $ws}) "
                "RETURN c.name, c.description, c.config_json, c.workspace_id, "
                "c.created_at, c.updated_at ORDER BY c.name",
                params={"ws": workspace_id},
                graph_name=self._graph_name,
            )
            return [
                {
                    "name": row[0],
                    "description": row[1] or "",
                    "config_json": row[2],
                    "workspace_id": row[3],
                    "created_at": str(row[4]) if row[4] else "",
                    "updated_at": str(row[5]) if row[5] else "",
                }
                for row in (result or [])
            ]
        except Exception as e:
            logger.warning("Failed to list pipeline configs for workspace '%s': %s", workspace_id, e)
            return []

    def delete_pipeline_config(self, name: str, workspace_id: str) -> bool:
        """Delete a named pipeline config. Returns True if deleted."""
        backend = self._get_backend()
        try:
            backend.query(
                "MATCH (c:PipelineConfig {name: $name, workspace_id: $ws}) DELETE c",
                params={"name": name, "ws": workspace_id},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to delete pipeline config '%s': %s", name, e)
            return False

    # ------------------------------------------------------------------ #
    # Entity pattern deletion (Phase 5)
    # ------------------------------------------------------------------ #

    def delete_entity_pattern(self, name: str, label: str, workspace_id: str | None = None) -> bool:
        """Delete an EntityPattern node by name and label.

        Returns:
            True if deleted successfully.
        """
        backend = self._get_backend()
        ws = workspace_id or self.workspace_id
        try:
            backend.query(
                "MATCH (p:EntityPattern {name: $name, label: $label}) "
                "WHERE p.workspace_id = $ws OR p.is_global = true "
                "DETACH DELETE p",
                params={"name": name.lower(), "label": label, "ws": ws},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to delete entity pattern '%s' -> '%s': %s", name, label, e)
            return False

    # ------------------------------------------------------------------ #
    # Relation type management (ONTO-PUB-3)
    # ------------------------------------------------------------------ #

    def seed_relation_types(self) -> int:
        """Seed canonical relation types as RelationType nodes. Idempotent.

        Returns:
            Number of types actually created (skips existing).
        """
        try:
            from smartmemory.relations.schema import CANONICAL_RELATION_TYPES
        except ImportError:
            logger.warning("smartmemory.relations.schema not available; skipping relation type seeding")
            return 0

        backend = self._get_backend()
        created = 0

        for name, typedef in CANONICAL_RELATION_TYPES.items():
            try:
                existing = backend.query(
                    "MATCH (t:RelationType {name: $name}) RETURN t.name",
                    params={"name": name},
                    graph_name=self._graph_name,
                )
                if existing:
                    continue
                backend.query(
                    "CREATE (t:RelationType {name: $name, status: 'seed', category: $category})",
                    params={"name": name, "category": typedef.category},
                    graph_name=self._graph_name,
                )
                created += 1
            except Exception as e:
                logger.warning("Failed to seed relation type '%s': %s", name, e)

        logger.info("Seeded %d/%d relation types in %s", created, len(CANONICAL_RELATION_TYPES), self._graph_name)
        return created

    def get_relation_types(self, status: str | None = None) -> list[dict]:
        """Return all relation types, optionally filtered by status."""
        backend = self._get_backend()
        try:
            if status:
                result = backend.query(
                    "MATCH (t:RelationType {status: $status}) RETURN t.name AS name, t.status AS status, "
                    "t.category AS category ORDER BY t.name",
                    params={"status": status},
                    graph_name=self._graph_name,
                )
            else:
                result = backend.query(
                    "MATCH (t:RelationType) RETURN t.name AS name, t.status AS status, "
                    "t.category AS category ORDER BY t.name",
                    graph_name=self._graph_name,
                )
            return [{"name": row[0], "status": row[1], "category": row[2]} for row in (result or [])]
        except Exception as e:
            logger.warning("Failed to query relation types: %s", e)
            return []

    def add_provisional_relation_type(self, name: str, category: str = "unknown") -> bool:
        """Add a provisional relation type. No-op if already exists.

        Returns:
            True if created, False if already existed.
        """
        backend = self._get_backend()
        try:
            existing = backend.query(
                "MATCH (t:RelationType {name: $name}) RETURN t.name",
                params={"name": name},
                graph_name=self._graph_name,
            )
            if existing:
                return False
            backend.query(
                "CREATE (t:RelationType {name: $name, status: 'provisional', category: $category})",
                params={"name": name, "category": category},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to add provisional relation type '%s': %s", name, e)
            return False

    def seed_type_pair_edges(self) -> int:
        """Ensure VALID_FOR edges exist between EntityType nodes for type-pair priors.

        Graph model: EntityType -[VALID_FOR {relation: "works_at"}]-> EntityType
        Wildcard pairs ("*", "*") are skipped — they apply universally.

        Uses MERGE (idempotent) — safe to call repeatedly.

        Returns:
            Number of edges ensured (includes already-existing edges).
        """
        try:
            from smartmemory.relations.schema import CANONICAL_RELATION_TYPES
        except ImportError:
            return 0

        backend = self._get_backend()
        ensured = 0

        for name, typedef in CANONICAL_RELATION_TYPES.items():
            for src, tgt in typedef.type_pairs:
                if src == "*" or tgt == "*":
                    continue  # wildcards are implicit, not stored as edges
                try:
                    backend.query(
                        "MERGE (s:EntityType {name: $src}) "
                        "MERGE (t:EntityType {name: $tgt}) "
                        "MERGE (s)-[r:VALID_FOR {relation: $rel}]->(t)",
                        params={"src": src.title(), "tgt": tgt.title(), "rel": name},
                        graph_name=self._graph_name,
                    )
                    ensured += 1
                except Exception as e:
                    logger.warning("Failed to ensure VALID_FOR edge %s->%s for '%s': %s", src, tgt, name, e)

        logger.info("Ensured %d type-pair edges in %s", ensured, self._graph_name)
        return ensured

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _query_type(self, backend, name: str):
        """Query a single type by name."""
        try:
            return backend.query(
                "MATCH (t:EntityType {name: $name}) RETURN t.name, t.status",
                params={"name": name},
                graph_name=self._graph_name,
            )
        except Exception:
            return None

    def _create_type(self, backend, name: str, status: str):
        """Create an EntityType node."""
        backend.query(
            "CREATE (t:EntityType {name: $name, status: $status})",
            params={"name": name, "status": status},
            graph_name=self._graph_name,
        )
