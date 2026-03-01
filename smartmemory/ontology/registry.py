import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from uuid import uuid4

from smartmemory.configuration import MemoryConfig
from smartmemory.ontology.ir_models import Audit
from smartmemory.ontology.ir_models import OntologyIR

logger = logging.getLogger(__name__)


class OntologyRegistry:
    """FalkorDB-based ontology registry manager with graph storage."""

    def __init__(self, config: MemoryConfig = None, scope_provider=None):
        if config is None:
            config = MemoryConfig()

        self.config = config
        self.scope_provider = scope_provider
        self._init_falkordb_connection()
        self._ensure_registry_schema()
    
    def _get_audit_user(self) -> str:
        """Get user identifier for audit trail from scope_provider."""
        if self.scope_provider:
            ctx = self.scope_provider.get_write_context()
            # Use whatever user identifier the scope_provider provides
            for key in ctx:
                if 'user' in key.lower() or 'created_by' in key.lower():
                    return str(ctx[key])
        return "unknown"

    def _init_falkordb_connection(self):
        """Initialize FalkorDB connection."""
        try:
            import falkordb as _falkordb
        except ImportError:
            raise ImportError(
                "falkordb is required for server mode. "
                "Install it with: pip install smartmemory-core[server]"
            ) from None
        try:
            host = self.config.graph_db.get("host", "localhost")
            port = self.config.graph_db.get("port", 9010)
            graph_name = self.config.graph_db.get("graph_name", "ontology")

            self.client = _falkordb.FalkorDB(host=host, port=port)
            self.graph = self.client.select_graph(graph_name)
            logger.info(f"FalkorDB connected for ontology registries: {host}:{port}/{graph_name}")
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            self.client = None
            self.graph = None

    def _ensure_registry_schema(self):
        """Ensure registry graph schema and indexes exist."""
        if not self.graph:
            return

        try:
            # Create indexes for registry operations
            indexes = [
                "CREATE INDEX FOR (r:Registry) ON (r.name)",
                "CREATE INDEX FOR (r:Registry) ON (r.id)",
                "CREATE INDEX FOR (s:Snapshot) ON (s.registry_id)",
                "CREATE INDEX FOR (s:Snapshot) ON (s.version)",
                "CREATE INDEX FOR (c:ChangelogEntry) ON (c.registry_id)",
                "CREATE INDEX FOR (c:ChangelogEntry) ON (c.version)"
            ]

            for index_query in indexes:
                try:
                    self.graph.query(index_query)
                except Exception:
                    pass  # Index might already exist

            # Create default registry if it doesn't exist
            self._ensure_default_registry()

        except Exception as e:
            logger.warning(f"Failed to ensure registry schema: {e}")

    def _ensure_default_registry(self):
        """Ensure default registry exists."""
        try:
            # Check if default registry exists
            query = "MATCH (r:Registry {name: 'default'}) RETURN r"
            result = self.graph.query(query)

            if not result.result_set:
                # Create default registry
                now = datetime.now().isoformat()
                query = f"""
                CREATE (r:Registry {{
                    id: 'default',
                    name: 'default',
                    description: 'Default ontology registry',
                    domain: 'general',
                    created_at: '{now}',
                    updated_at: '{now}',
                    created_by: 'system',
                    current_version: 'v1.0.0'
                }})
                """
                self.graph.query(query)

                # Create initial empty snapshot
                empty_ontology = OntologyIR(
                    concepts=[], relations=[], taxonomy=[],
                    audit=Audit(version="v1.0.0")
                )
                self._save_snapshot("default", "v1.0.0", empty_ontology, "Initial default registry")
                logger.info("Created default ontology registry in FalkorDB")

        except Exception as e:
            logger.error(f"Failed to ensure default registry: {e}")

    def _escape_string(self, s: str) -> str:
        """Escape string for Cypher queries."""
        if not s:
            return ""
        return s.replace("'", "\\'").replace('"', '\\"')

    def create_registry(self, name: str, description: str, domain: str) -> str:
        """Create a new ontology registry in FalkorDB."""
        if not self.graph:
            raise RuntimeError("FalkorDB not available")

        registry_id = name.lower().replace(" ", "_")

        # Check if registry already exists
        escaped_name = self._escape_string(name)
        check_query = f"MATCH (r:Registry {{name: '{escaped_name}'}}) RETURN r"
        result = self.graph.query(check_query)

        if result.result_set:
            raise ValueError(f"Registry '{name}' already exists")

        # Create registry node
        now = datetime.now().isoformat()
        escaped_desc = self._escape_string(description)
        audit_user = self._get_audit_user()
        escaped_user = self._escape_string(audit_user)

        query = f"""
        CREATE (r:Registry {{
            id: '{registry_id}',
            name: '{escaped_name}',
            description: '{escaped_desc}',
            domain: '{domain}',
            created_at: '{now}',
            updated_at: '{now}',
            created_by: '{escaped_user}',
            current_version: 'v1.0.0'
        }})
        RETURN r.id as id
        """

        result = self.graph.query(query)

        # Create initial empty ontology snapshot
        empty_ontology = OntologyIR(
            concepts=[], relations=[], taxonomy=[],
            audit=Audit(version="v1.0.0")
        )
        self._save_snapshot(registry_id, "v1.0.0", empty_ontology, f"Initial version of {name}")

        # Add initial changelog entry
        self._add_changelog_entry(registry_id, "v1.0.0", "create", {}, f"Created registry '{name}'")

        return registry_id

    def get_registry(self, registry_id: str) -> Dict[str, Any]:
        """Get registry by ID from FalkorDB."""
        if not self.graph:
            raise RuntimeError("FalkorDB not available")

        try:
            escaped_id = self._escape_string(registry_id)
            query = f"""
            MATCH (r:Registry {{id: '{escaped_id}'}})
            RETURN r.id as id, r.name as name, r.description as description,
                   r.domain as domain, r.created_at as created_at, 
                   r.updated_at as updated_at, r.created_by as created_by,
                   r.current_version as current_version
            """

            result = self.graph.query(query)

            if result.result_set:
                row = result.result_set[0]

                # Get versions from snapshots
                versions_query = f"""
                MATCH (s:Snapshot {{registry_id: '{escaped_id}'}})
                RETURN s.version as version
                ORDER BY s.created_at ASC
                """
                versions_result = self.graph.query(versions_query)
                versions = [row[0] for row in versions_result.result_set]

                # Get changelog
                changelog_query = f"""
                MATCH (c:ChangelogEntry {{registry_id: '{escaped_id}'}})
                RETURN c.version as version, c.timestamp as timestamp, c.changed_by as changed_by,
                       c.message as message, c.action as action, c.changes_summary as changes_summary
                ORDER BY c.timestamp DESC
                """
                changelog_result = self.graph.query(changelog_query)
                changelog = []
                for cl_row in changelog_result.result_set:
                    changelog.append({
                        "version": cl_row[0],
                        "timestamp": cl_row[1],
                        "changed_by": cl_row[2],
                        "message": cl_row[3],
                        "action": cl_row[4],
                        "changes_summary": json.loads(cl_row[5]) if cl_row[5] else {}
                    })

                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "domain": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "created_by": row[6],
                    "current_version": row[7],
                    "versions": versions,
                    "changelog": changelog
                }

            raise ValueError(f"Registry '{registry_id}' not found")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to get registry {registry_id}: {e}")
            raise ValueError(f"Registry '{registry_id}' not found")

    def list_registries(self) -> List[Dict[str, Any]]:
        """List all registries from FalkorDB."""
        if not self.graph:
            return []

        try:
            query = """
            MATCH (r:Registry)
            RETURN r.id as id, r.name as name, r.description as description,
                   r.domain as domain, r.created_at as created_at,
                   r.updated_at as updated_at, r.created_by as created_by,
                   r.current_version as current_version
            ORDER BY r.created_at DESC
            """

            result = self.graph.query(query)

            registries = []
            for row in result.result_set:
                registries.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "domain": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "created_by": row[6],
                    "current_version": row[7]
                })

            return registries

        except Exception as e:
            logger.error(f"Failed to list registries: {e}")
            return []

    def get_snapshot(self, registry_id: str, version: str = None) -> Dict[str, Any]:
        """Get a specific snapshot of a registry from FalkorDB."""
        if not self.graph:
            raise RuntimeError("FalkorDB not available")

        registry = self.get_registry(registry_id)
        target_version = version or registry["current_version"]

        try:
            escaped_id = self._escape_string(registry_id)
            escaped_version = self._escape_string(target_version)

            query = f"""
            MATCH (s:Snapshot {{registry_id: '{escaped_id}', version: '{escaped_version}'}})
            RETURN s.ontology_data as ontology_data, s.created_at as created_at,
                   s.created_by as created_by, s.message as message
            """

            result = self.graph.query(query)

            if result.result_set:
                row = result.result_set[0]
                ontology_data = json.loads(row[0]) if row[0] else {}

                return {
                    "registry_id": registry_id,
                    "snapshot": ontology_data,
                    "version": target_version,
                    "metadata": {
                        "created_at": row[1],
                        "created_by": row[2],
                        "message": row[3] or f"Snapshot {target_version}"
                    }
                }
            else:
                # Return empty snapshot for missing versions
                empty_ontology = OntologyIR(
                    concepts=[], relations=[], taxonomy=[],
                    audit=Audit(version=target_version)
                )
                return {
                    "registry_id": registry_id,
                    "snapshot": empty_ontology.to_dict(),
                    "version": target_version,
                    "metadata": {
                        "created_at": registry["created_at"],
                        "created_by": registry.get("created_by", "system")
                    }
                }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to get snapshot {registry_id}@{target_version}: {e}")
            # Return empty snapshot on error
            empty_ontology = OntologyIR(
                concepts=[], relations=[], taxonomy=[],
                audit=Audit(version=target_version)
            )
            return {
                "registry_id": registry_id,
                "snapshot": empty_ontology.to_dict(),
                "version": target_version,
                "metadata": {"error": str(e)}
            }

    def _save_snapshot(self, registry_id: str, version: str, ontology: OntologyIR, message: str):
        """Save a snapshot to FalkorDB."""
        if not self.graph:
            raise RuntimeError("FalkorDB not available")

        try:
            escaped_id = self._escape_string(registry_id)
            escaped_version = self._escape_string(version)
            audit_user = self._get_audit_user()
            escaped_user = self._escape_string(audit_user)
            escaped_message = self._escape_string(message)
            ontology_json = self._escape_string(json.dumps(ontology.to_dict()))
            now = datetime.now().isoformat()

            # Create snapshot node
            query = f"""
            CREATE (s:Snapshot {{
                id: '{str(uuid4())}',
                registry_id: '{escaped_id}',
                version: '{escaped_version}',
                created_at: '{now}',
                created_by: '{escaped_user}',
                message: '{escaped_message}',
                ontology_data: '{ontology_json}',
                concept_count: {len(ontology.concepts)},
                relation_count: {len(ontology.relations)},
                taxonomy_count: {len(ontology.taxonomy)}
            }})
            """

            self.graph.query(query)

            # Link to registry
            link_query = f"""
            MATCH (r:Registry {{id: '{escaped_id}'}})
            MATCH (s:Snapshot {{registry_id: '{escaped_id}', version: '{escaped_version}'}})
            CREATE (r)-[:HAS_SNAPSHOT]->(s)
            """

            self.graph.query(link_query)

        except Exception as e:
            logger.error(f"Failed to save snapshot {registry_id}@{version}: {e}")
            raise

    def _add_changelog_entry(self, registry_id: str, version: str, action: str, changes_summary: Dict, message: str):
        """Add changelog entry to FalkorDB."""
        if not self.graph:
            return

        try:
            escaped_id = self._escape_string(registry_id)
            escaped_version = self._escape_string(version)
            escaped_action = self._escape_string(action)
            audit_user = self._get_audit_user()
            escaped_user = self._escape_string(audit_user)
            escaped_message = self._escape_string(message)
            changes_json = self._escape_string(json.dumps(changes_summary))
            now = datetime.now().isoformat()

            query = f"""
            CREATE (c:ChangelogEntry {{
                id: '{str(uuid4())}',
                registry_id: '{escaped_id}',
                version: '{escaped_version}',
                timestamp: '{now}',
                changed_by: '{escaped_user}',
                message: '{escaped_message}',
                action: '{escaped_action}',
                changes_summary: '{changes_json}'
            }})
            """

            self.graph.query(query)

            # Link to registry
            link_query = f"""
            MATCH (r:Registry {{id: '{escaped_id}'}})
            MATCH (c:ChangelogEntry {{registry_id: '{escaped_id}', version: '{escaped_version}'}})
            CREATE (r)-[:HAS_CHANGELOG]->(c)
            """

            self.graph.query(link_query)

        except Exception as e:
            logger.error(f"Failed to add changelog entry: {e}")

    def apply_changeset(self, registry_id: str, base_version: str, changeset_dict: Dict[str, Any], message: str = "") -> str:
        """Apply a changeset to create a new version."""
        registry = self.get_registry(registry_id)

        # Get base snapshot
        base_snapshot = self.get_snapshot(registry_id, base_version)
        base_ontology = OntologyIR.from_dict(base_snapshot["snapshot"])

        # Apply changeset (merge concepts, relations, taxonomy)
        changeset = OntologyIR.from_dict(changeset_dict)
        merged_ontology = self._merge_ontology(base_ontology, changeset)

        # Create new version
        new_version = self._increment_version(base_version)
        merged_ontology.version = new_version

        # Save snapshot
        self._save_snapshot(registry_id, new_version, merged_ontology, message or f"Applied changeset from {base_version}")

        # Update registry current version
        self._update_registry_version(registry_id, new_version)

        # Add changelog entry
        changes_summary = {
            "concepts_added": len(changeset.concepts),
            "relations_added": len(changeset.relations),
            "taxonomy_added": len(changeset.taxonomy)
        }

        self._add_changelog_entry(
            registry_id, new_version, "apply_changeset", changes_summary,
            message or f"Applied changeset from {base_version}"
        )

        return new_version

    def _update_registry_version(self, registry_id: str, new_version: str):
        """Update registry current version in FalkorDB."""
        if not self.graph:
            return

        try:
            escaped_id = self._escape_string(registry_id)
            escaped_version = self._escape_string(new_version)
            now = datetime.now().isoformat()

            query = f"""
            MATCH (r:Registry {{id: '{escaped_id}'}})
            SET r.current_version = '{escaped_version}', r.updated_at = '{now}'
            """

            self.graph.query(query)

        except Exception as e:
            logger.error(f"Failed to update registry version: {e}")

    def _merge_ontology(self, base: OntologyIR, changeset: OntologyIR) -> OntologyIR:
        """Merge base ontology with changeset."""
        # Simple merge - add new concepts/relations, update existing ones
        merged_concepts = {c.id: c for c in base.concepts}
        merged_relations = {r.id: r for r in base.relations}
        merged_taxonomy = {f"{t.child}-{t.parent}": t for t in base.taxonomy}

        # Apply changeset
        for concept in changeset.concepts:
            merged_concepts[concept.id] = concept
        for relation in changeset.relations:
            merged_relations[relation.id] = relation
        for taxonomy in changeset.taxonomy:
            key = f"{taxonomy.child}-{taxonomy.parent}"
            merged_taxonomy[key] = taxonomy

        return OntologyIR(
            concepts=list(merged_concepts.values()),
            relations=list(merged_relations.values()),
            taxonomy=list(merged_taxonomy.values()),
            ir_version=base.version if hasattr(base, 'version') else base.ir_version,
            registry_id=base.registry_id if hasattr(base, 'registry_id') else "default"
        )

    def _increment_version(self, version: str) -> str:
        """Increment version number."""
        try:
            # Simple semantic versioning
            if version.startswith('v'):
                version = version[1:]
            parts = version.split('.')
            if len(parts) >= 3:
                parts[2] = str(int(parts[2]) + 1)
            else:
                parts.append('1')
            return f"v{'.'.join(parts)}"
        except Exception:
            # Fallback to timestamp-based versioning
            return f"v1.0.{int(datetime.now().timestamp())}"
