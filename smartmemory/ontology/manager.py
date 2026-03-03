"""
Ontology Management System for Hybrid Freeform + Structured Entity Extraction

Supports:
- Freeform LLM-driven extraction (current mode)
- Ontology-guided extraction with validation/enrichment
- HITL (Human-In-The-Loop) ontology editing
- LLM-driven ontology evolution and rule generation
- Migration between freeform and structured modes
"""

import logging
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

from smartmemory.ontology.models import EntityTypeDefinition, RelationshipTypeDefinition, OntologyRule, Ontology
from smartmemory.stores.ontology import OntologyStorage, FileSystemOntologyStorage
from smartmemory.utils import get_config


class OntologyManager:
    """Main interface for ontology management operations."""

    def __init__(self, storage: OntologyStorage = None, scope_provider=None):
        self.scope_provider = scope_provider
        self.storage = storage or FileSystemOntologyStorage()
        self._active_ontology: Optional[Ontology] = None

    def create_ontology(self, name: str, domain: str = "", description: str = "") -> Ontology:
        """Create a new ontology."""
        ontology = Ontology(name)
        ontology.domain = domain
        ontology.description = description
        self.storage.save_ontology(ontology)
        return ontology

    def load_ontology(self, ontology_id: str) -> Optional[Ontology]:
        """Load an ontology by ID."""
        return self.storage.load_ontology(ontology_id)

    def set_active_ontology(self, ontology: Ontology) -> None:
        """Set the active ontology for extraction guidance."""
        self._active_ontology = ontology

    def get_active_ontology(self) -> Optional[Ontology]:
        """Get the currently active ontology."""
        return self._active_ontology

    def infer_ontology_from_extractions(self, extraction_history: List[Dict[str, Any]],
                                        ontology_name: str = "inferred") -> Ontology:
        """Infer an ontology from historical extraction patterns."""
        # Attempt OntoGPT integration if enabled; fallback to frequency-based inference
        try:
            cfg = get_config('ontology')
            inf_cfg = cfg.get('inference') or {}
            engine_name = str(inf_cfg.get('stages', 'basic')).lower() if hasattr(inf_cfg, 'get') else 'basic'
            ontogpt_cfg = inf_cfg.get('ontogpt') or {} if hasattr(inf_cfg, 'get') else {}
            # ConfigDict.get returns ConfigDict for dicts; use getattr/get semantics to read safely
            try:
                og_enabled = bool(ontogpt_cfg.get('enabled', False))  # type: ignore[attr-defined]
            except Exception:
                og_enabled = False
            ontogpt_enabled = og_enabled or engine_name == 'ontogpt'
        except Exception:
            ontogpt_enabled = False
            ontogpt_cfg = {}
        if ontogpt_enabled:
            try:
                from smartmemory.ontology.inference.ontogpt import OntoGPTInferenceEngine
                engine = OntoGPTInferenceEngine(config=dict(ontogpt_cfg) if hasattr(ontogpt_cfg, 'items') else ontogpt_cfg)
                onto = engine.infer_ontology_from_extractions(extraction_history, ontology_name=ontology_name)
                if onto is not None:
                    self.storage.save_ontology(onto)
                    return onto
            except Exception as e:
                logging.getLogger(__name__).warning("OntoGPT inference failed; using basic inference instead: %s", e, exc_info=True)

        ontology = Ontology(ontology_name)
        ontology.created_by = "inferred"
        ontology.description = f"Ontology inferred from {len(extraction_history)} extraction examples"

        # Analyze entity types
        entity_type_stats = {}
        relationship_type_stats = {}

        for extraction in extraction_history:
            # Count entity types and their properties
            for entity in extraction.get('entities', []):
                entity_type = entity.get('type', '').lower()
                if entity_type:
                    if entity_type not in entity_type_stats:
                        entity_type_stats[entity_type] = {
                            'count': 0,
                            'properties': {},
                            'examples': []
                        }

                    entity_type_stats[entity_type]['count'] += 1
                    entity_type_stats[entity_type]['examples'].append(entity.get('name', ''))

                    # Track property usage
                    for prop_name, prop_value in entity.get('properties') or {}.items():
                        if prop_name not in entity_type_stats[entity_type]['properties']:
                            entity_type_stats[entity_type]['properties'][prop_name] = 0
                        entity_type_stats[entity_type]['properties'][prop_name] += 1

            # Count relationship types
            for relation in extraction.get('relations', []):
                rel_type = relation.get('relation_type', '').lower()
                if rel_type:
                    if rel_type not in relationship_type_stats:
                        relationship_type_stats[rel_type] = {
                            'count': 0,
                            'source_types': set(),
                            'target_types': set()
                        }

                    relationship_type_stats[rel_type]['count'] += 1
                    # Note: We'd need source/target entity types to populate these

        # Create entity type definitions from patterns
        for entity_type, stats in entity_type_stats.items():
            if stats['count'] >= 2:  # Only include types seen multiple times
                # Determine required properties (present in >80% of instances)
                total_instances = stats['count']
                required_props = {
                    prop for prop, count in stats['properties'].items()
                    if count / total_instances > 0.8
                }

                entity_def = EntityTypeDefinition(
                    name=entity_type,
                    description=f"Inferred entity type (seen {stats['count']} times)",
                    properties={prop: "string" for prop in stats['properties'].keys()},
                    required_properties=required_props,
                    parent_types=set(),
                    aliases=set(),
                    examples=stats['examples'][:5],  # Keep top 5 examples
                    created_by="inferred",
                    created_at=datetime.now(UTC),
                    confidence=min(1.0, stats['count'] / 10)  # Higher confidence for more examples
                )
                ontology.add_entity_type(entity_def)

        # Create relationship type definitions from patterns
        for rel_type, stats in relationship_type_stats.items():
            if stats['count'] >= 2:  # Only include types seen multiple times
                rel_def = RelationshipTypeDefinition(
                    name=rel_type,
                    description=f"Inferred relationship type (seen {stats['count']} times)",
                    source_types=stats['source_types'],
                    target_types=stats['target_types'],
                    properties={},
                    created_by="inferred",
                    created_at=datetime.now(UTC),
                    confidence=min(1.0, stats['count'] / 10)
                )
                ontology.add_relationship_type(rel_def)

        self.storage.save_ontology(ontology)
        return ontology

    def migrate_freeform_to_ontology(self, extraction_history: List[Dict[str, Any]],
                                     base_ontology: Optional[Ontology] = None) -> Ontology:
        """Migrate from freeform extraction to ontology-guided extraction."""
        if base_ontology:
            # Extend existing ontology with patterns from freeform extractions
            ontology = base_ontology
            ontology.description += f" Extended with {len(extraction_history)} freeform extractions"
        else:
            # Create new ontology from freeform patterns
            ontology = self.infer_ontology_from_extractions(extraction_history, "migrated_from_freeform")

        # Add migration metadata
        migration_rule = OntologyRule(
            id=str(uuid.uuid4()),
            name="freeform_migration",
            description="Rule created during migration from freeform to ontology-guided extraction",
            rule_type="migration",
            conditions={"source": "freeform_extraction"},
            actions={"validate_against_ontology": True, "suggest_improvements": True},
            created_by="migration",
            created_at=datetime.now(UTC)
        )
        ontology.add_rule(migration_rule)

        self.storage.save_ontology(ontology)
        return ontology

    def list_ontologies(self) -> List[Dict[str, str]]:
        """List all available ontologies."""
        return self.storage.list_ontologies()

    def delete_ontology(self, ontology_id: str) -> bool:
        """Delete an ontology."""
        return self.storage.delete_ontology(ontology_id)
