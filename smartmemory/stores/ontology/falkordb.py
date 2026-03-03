"""
FalkorDB Graph Materialization Service

Materializes approved ontology concepts and relations as graph nodes and edges.
"""
import logging
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from smartmemory.configuration import MemoryConfig
from smartmemory.ontology.ir_models import OntologyIR, Concept, Relation, TaxonomyRelation, Status

logger = logging.getLogger(__name__)


class FalkorDBGraphService:
    """FalkorDB service for ontology graph materialization"""

    def __init__(self, config: MemoryConfig = None):
        if config is None:
            config = MemoryConfig()

        self.config = config
        host = config.graph_db.get("host", "localhost")
        port = config.graph_db.get("port", 9010)
        graph_name = config.graph_db.get("graph_name", "ontology")
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.client = None
        self.graph = None

        try:
            import falkordb as _falkordb
        except ImportError:
            raise ImportError(
                "falkordb is required for server mode. "
                "Install it with: pip install smartmemory-core[server]"
            ) from None
        try:
            self.client = _falkordb.FalkorDB(host=host, port=port)
            self.graph = self.client.select_graph(graph_name)
            logger.info(f"FalkorDB connected: {host}:{port}/{graph_name}")
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            self.client = None
            self.graph = None

    def materialize_ontology(self, ontology: OntologyIR, min_confidence: float = 0.0) -> Dict[str, int]:
        """Materialize approved ontology elements to graph"""
        if not self.graph:
            logger.warning("FalkorDB not available, skipping materialization")
            return {"concepts": 0, "relations": 0, "taxonomy": 0}

        try:
            # Get approved elements above confidence threshold
            approved_concepts = ontology.get_approved_concepts(min_confidence)
            approved_relations = ontology.get_approved_relations(min_confidence)
            approved_taxonomy = [
                t for t in ontology.taxonomy
                if t.status == Status.APPROVED and t.confidence >= min_confidence
            ]

            # Clear existing data for this version (optional - could be configurable)
            self._clear_version_data(ontology.audit.version if ontology.audit else "unknown")

            # Materialize concepts as nodes
            concept_count = self._materialize_concepts(approved_concepts, ontology.audit.version if ontology.audit else "v1.0.0")

            # Materialize relations as edges
            relation_count = self._materialize_relations(approved_relations, ontology.audit.version if ontology.audit else "v1.0.0")

            # Materialize taxonomy as IS_A edges
            taxonomy_count = self._materialize_taxonomy(approved_taxonomy, ontology.audit.version if ontology.audit else "v1.0.0")

            # Create indexes for performance
            self._create_indexes()

            logger.info(f"Materialized ontology: {concept_count} concepts, {relation_count} relations, {taxonomy_count} taxonomy")

            return {
                "concepts": concept_count,
                "relations": relation_count,
                "taxonomy": taxonomy_count
            }

        except Exception as e:
            logger.error(f"Graph materialization failed: {e}")
            raise

    def _materialize_concepts(self, concepts: List[Concept], version: str) -> int:
        """Materialize concepts as graph nodes"""
        if not concepts:
            return 0

        try:
            # Build batch query for concepts
            queries = []
            for concept in concepts:
                # Escape strings for Cypher
                label = self._escape_string(concept.label)
                synonyms = [self._escape_string(s) for s in concept.synonyms]

                query = f"""
                CREATE (c:Concept {{
                    id: '{concept.id}',
                    canonical_label: '{label}',
                    synonyms: {synonyms},
                    status: '{concept.status.value}',
                    confidence: {concept.confidence},
                    origin: '{concept.origin.value}',
                    pinned: {str(concept.pinned).lower()},
                    snapshot_version: '{version}',
                    created_at: '{datetime.now(UTC).isoformat()}'
                }})
                """
                queries.append(query)

            # Execute batch
            for query in queries:
                self.graph.query(query)

            return len(concepts)

        except Exception as e:
            logger.error(f"Failed to materialize concepts: {e}")
            return 0

    def _materialize_relations(self, relations: List[Relation], version: str) -> int:
        """Materialize relations as graph edges"""
        if not relations:
            return 0

        try:
            relation_count = 0

            for relation in relations:
                if not relation.domain or not relation.range:
                    continue  # Skip relations without proper domain/range

                # Create relation edge between domain and range concepts
                domain_id = self._escape_string(relation.domain)
                range_id = self._escape_string(relation.range)
                label = self._escape_string(relation.label)
                aliases = [self._escape_string(a) for a in relation.aliases]

                query = f"""
                MATCH (domain:Concept {{id: '{domain_id}'}})
                MATCH (range:Concept {{id: '{range_id}'}})
                CREATE (domain)-[r:RELATION {{
                    name: '{label}',
                    relation_id: '{relation.id}',
                    aliases: {aliases},
                    confidence: {relation.confidence},
                    snapshot_version: '{version}',
                    created_at: '{datetime.now(UTC).isoformat()}'
                }}]->(range)
                """

                try:
                    result = self.graph.query(query)
                    if result.relationships_created > 0:
                        relation_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create relation {relation.id}: {e}")

            return relation_count

        except Exception as e:
            logger.error(f"Failed to materialize relations: {e}")
            return 0

    def _materialize_taxonomy(self, taxonomy: List[TaxonomyRelation], version: str) -> int:
        """Materialize taxonomy as IS_A edges"""
        if not taxonomy:
            return 0

        try:
            taxonomy_count = 0

            for tax in taxonomy:
                parent_id = self._escape_string(tax.parent)
                child_id = self._escape_string(tax.child)

                query = f"""
                MATCH (parent:Concept {{id: '{parent_id}'}})
                MATCH (child:Concept {{id: '{child_id}'}})
                CREATE (child)-[r:IS_A {{
                    confidence: {tax.confidence},
                    origin: '{tax.origin.value}',
                    snapshot_version: '{version}',
                    created_at: '{datetime.now(UTC).isoformat()}'
                }}]->(parent)
                """

                try:
                    result = self.graph.query(query)
                    if result.relationships_created > 0:
                        taxonomy_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create taxonomy {tax.child} -> {tax.parent}: {e}")

            return taxonomy_count

        except Exception as e:
            logger.error(f"Failed to materialize taxonomy: {e}")
            return 0

    def _clear_version_data(self, version: str):
        """Clear existing data for a specific version"""
        try:
            # Delete nodes and relationships for this version
            query = f"""
            MATCH (n {{snapshot_version: '{version}'}})
            DETACH DELETE n
            """
            self.graph.query(query)

            # Delete relationships for this version
            query = f"""
            MATCH ()-[r {{snapshot_version: '{version}'}}]-()
            DELETE r
            """
            self.graph.query(query)

        except Exception as e:
            logger.warning(f"Failed to clear version data: {e}")

    def _create_indexes(self):
        """Create performance indexes"""
        try:
            indexes = [
                "CREATE INDEX FOR (c:Concept) ON (c.id)",
                "CREATE INDEX FOR (c:Concept) ON (c.canonical_label)",
                "CREATE INDEX FOR (c:Concept) ON (c.snapshot_version)",
            ]

            for index_query in indexes:
                try:
                    self.graph.query(index_query)
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index creation skipped: {e}")

        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def query_graph(self, cypher_query: str) -> Dict[str, Any]:
        """Execute custom Cypher query"""
        if not self.graph:
            raise RuntimeError("FalkorDB not available")

        try:
            result = self.graph.query(cypher_query)

            # Convert result to dictionary format
            return {
                "nodes": getattr(result, 'nodes_created', 0),
                "relations": getattr(result, 'relationships_created', 0),  # FalkorDB uses 'relationships_created'
                "properties": getattr(result, 'properties_set', 0),
                "result_set": result.result_set if hasattr(result, 'result_set') else []
            }

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            raise

    def get_concept_neighbors(self, concept_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get concept neighbors up to specified depth"""
        if not self.graph:
            return {"nodes": [], "edges": []}

        try:
            escaped_id = self._escape_string(concept_id)
            query = f"""
            MATCH (c:Concept {{id: '{escaped_id}'}})
            CALL apoc.path.subgraphNodes(c, {{
                relationshipFilter: 'RELATION|IS_A',
                minLevel: 0,
                maxLevel: {depth}
            }}) YIELD node
            RETURN node
            """

            # Fallback query if APOC not available
            fallback_query = f"""
            MATCH (c:Concept {{id: '{escaped_id}'}})-[r*1..{depth}]-(neighbor:Concept)
            RETURN c, r, neighbor
            """

            try:
                result = self.graph.query(query)
            except Exception:
                result = self.graph.query(fallback_query)

            # Process result into nodes/edges format
            nodes = []
            edges = []

            if hasattr(result, 'result_set'):
                for record in result.result_set:
                    # Extract nodes and relationships from result
                    # This would need to be adapted based on FalkorDB result format
                    pass

            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            logger.error(f"Failed to get concept neighbors: {e}")
            return {"nodes": [], "edges": []}

    def get_ontology_stats(self, version: Optional[str] = None) -> Dict[str, int]:
        """Get ontology statistics from graph"""
        if not self.graph:
            return {}

        try:
            version_filter = f" {{snapshot_version: '{version}'}}" if version else ""

            queries = {
                "concepts": f"MATCH (c:Concept{version_filter}) RETURN count(c) as count",
                "relations": f"MATCH ()-[r:RELATION{version_filter}]-() RETURN count(r) as count",
                "taxonomy": f"MATCH ()-[r:IS_A{version_filter}]-() RETURN count(r) as count"
            }

            stats = {}
            for stat_name, query in queries.items():
                try:
                    result = self.graph.query(query)
                    if result.result_set:
                        stats[stat_name] = result.result_set[0][0]
                    else:
                        stats[stat_name] = 0
                except Exception as e:
                    logger.warning(f"Failed to get {stat_name} count: {e}")
                    stats[stat_name] = 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get ontology stats: {e}")
            return {}

    def _escape_string(self, s: str) -> str:
        """Escape string for Cypher query"""
        if not s:
            return ""
        # Escape single quotes and backslashes
        return s.replace("'", "\\'").replace("\\", "\\\\")

    def close(self):
        """Close FalkorDB connection"""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing FalkorDB connection: {e}")
            finally:
                self.client = None
                self.graph = None
