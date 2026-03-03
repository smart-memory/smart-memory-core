"""Ontology template catalog: built-in JSON templates + custom tenant templates."""

import copy
import json
import logging
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional

from smartmemory.ontology.models import Ontology, TemplateInfo, TemplatePreview
from smartmemory.stores.ontology import OntologyStorage

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
MANIFEST_FILE = TEMPLATES_DIR / "_manifest.json"


class TemplateService:
    """Browse, preview, and clone ontology templates.

    Built-in templates are loaded from JSON files in the templates/ directory.
    Custom templates are ontologies stored via OntologyStorage with is_template metadata.
    """

    def __init__(self, storage: OntologyStorage) -> None:
        self.storage = storage
        self._builtin_cache: Optional[List[TemplateInfo]] = None
        self._builtin_data_cache: Dict[str, Dict] = {}

    def list_builtin_templates(self) -> List[TemplateInfo]:
        """Load manifest from templates/ directory. Results are cached."""
        if self._builtin_cache is not None:
            return self._builtin_cache

        if not MANIFEST_FILE.exists():
            logger.warning("Template manifest not found at %s", MANIFEST_FILE)
            self._builtin_cache = []
            return self._builtin_cache

        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)

        templates: List[TemplateInfo] = []
        for entry in manifest:
            templates.append(
                TemplateInfo(
                    id=entry["id"],
                    name=entry["name"],
                    domain=entry["domain"],
                    description=entry["description"],
                    version=entry.get("version", "1.0.0"),
                    entity_count=entry["entity_count"],
                    relationship_count=entry["relationship_count"],
                    is_builtin=True,
                    created_by="system",
                )
            )
        self._builtin_cache = templates
        return templates

    def list_custom_templates(self, tenant_id: str) -> List[TemplateInfo]:
        """List ontologies with is_template=True for this tenant."""
        all_ontologies = self.storage.list_ontologies()
        templates: List[TemplateInfo] = []
        for info in all_ontologies:
            if not info.get("is_template"):
                continue
            if info.get("tenant_id") != tenant_id:
                continue
            templates.append(
                TemplateInfo(
                    id=f"custom:{info['id']}",
                    name=info["name"],
                    domain=info.get("domain", ""),
                    description=info.get("description", ""),
                    version=info.get("version", "1.0.0"),
                    entity_count=int(info.get("entity_count", 0)),
                    relationship_count=int(info.get("relationship_count", 0)),
                    is_builtin=False,
                    created_by=info.get("created_by", "unknown"),
                )
            )
        return templates

    def list_all_templates(self, tenant_id: str) -> List[TemplateInfo]:
        """Combined: built-in + custom, sorted by name."""
        builtin = self.list_builtin_templates()
        custom = self.list_custom_templates(tenant_id)
        return sorted(builtin + custom, key=lambda t: t.name)

    def get_template_preview(self, template_id: str, tenant_id: str) -> Optional[TemplatePreview]:
        """Load full template for preview (entity types, relationships, rules)."""
        data = self._load_template_data(template_id, tenant_id)
        if data is None:
            return None

        metadata = data.get("template_metadata", {})
        entity_types = list(data.get("entity_types", {}).keys())
        relationship_types = list(data.get("relationship_types", {}).keys())
        rules = data.get("rules", {})

        return TemplatePreview(
            id=template_id,
            name=metadata.get("name", data.get("name", "")),
            domain=metadata.get("domain", data.get("domain", "")),
            description=metadata.get("description", data.get("description", "")),
            version=metadata.get("version", data.get("version", "1.0.0")),
            entity_count=len(entity_types),
            relationship_count=len(relationship_types),
            is_builtin=template_id.startswith("builtin:"),
            created_by=metadata.get("created_by", data.get("created_by", "system")),
            entity_types=entity_types,
            relationship_types=relationship_types,
            rules_count=len(rules),
        )

    def clone_template(
        self,
        template_id: str,
        target_name: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        """Deep-copy template into workspace as a new independent ontology. Returns new ontology_id."""
        data = self._load_template_data(template_id, tenant_id)
        if data is None:
            raise ValueError(f"Template not found: {template_id}")

        ontology = Ontology.from_dict(data)
        ontology.id = str(uuid.uuid4())
        ontology.name = target_name
        ontology.created_at = datetime.now(UTC)
        ontology.updated_at = datetime.now(UTC)
        ontology.created_by = user_id
        ontology.tenant_id = tenant_id
        ontology.is_template = False
        ontology.source_template = template_id

        self.storage.save_ontology(ontology)
        return ontology.id

    def save_as_template(
        self,
        registry_id: str,
        template_name: str,
        description: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        """Save existing ontology as a custom template. Returns template_id."""
        source = self.storage.load_ontology(registry_id)
        if source is None:
            raise ValueError(f"Ontology not found: {registry_id}")

        # Deep copy via round-trip serialization to avoid shared references
        source_data = source.to_dict()
        template = Ontology.from_dict(source_data)
        template.id = str(uuid.uuid4())
        template.name = template_name
        template.description = description
        template.created_by = user_id
        template.tenant_id = tenant_id
        template.is_template = True
        template.source_template = ""
        template.created_at = datetime.now(UTC)
        template.updated_at = datetime.now(UTC)

        self.storage.save_ontology(template)
        return template.id

    def delete_custom_template(self, template_id: str, tenant_id: str) -> bool:
        """Delete a custom template. Built-in templates cannot be deleted.

        Verifies the template belongs to the requesting tenant before deleting.
        """
        if template_id.startswith("builtin:"):
            raise ValueError("Cannot delete built-in templates")

        ontology_id = template_id.removeprefix("custom:")

        # Verify tenant ownership before deleting
        all_ontologies = self.storage.list_ontologies()
        match = next((o for o in all_ontologies if o["id"] == ontology_id), None)
        if match is None:
            return False
        if match.get("tenant_id") and match["tenant_id"] != tenant_id:
            logger.warning("Tenant %s attempted to delete template owned by %s", tenant_id, match.get("tenant_id"))
            return False

        return self.storage.delete_ontology(ontology_id)

    def _load_template_data(self, template_id: str, tenant_id: str) -> Optional[Dict]:
        """Load raw template data from file (builtin) or storage (custom).

        For custom templates, verifies tenant ownership before returning.
        Returns a deep copy so callers can mutate without corrupting the cache.
        """
        if template_id.startswith("builtin:"):
            data = self._load_builtin_data(template_id)
            return copy.deepcopy(data) if data else None
        elif template_id.startswith("custom:"):
            ontology_id = template_id.removeprefix("custom:")
            # Verify tenant ownership via metadata listing
            all_ontologies = self.storage.list_ontologies()
            match = next((o for o in all_ontologies if o["id"] == ontology_id), None)
            if match is None:
                return None
            if match.get("tenant_id") and match["tenant_id"] != tenant_id:
                logger.warning("Tenant %s attempted to access template owned by %s", tenant_id, match.get("tenant_id"))
                return None
            ontology = self.storage.load_ontology(ontology_id)
            if ontology is None:
                return None
            return ontology.to_dict()
        return None

    def _load_builtin_data(self, template_id: str) -> Optional[Dict]:
        """Load a built-in template JSON file by template_id."""
        if template_id in self._builtin_data_cache:
            return self._builtin_data_cache[template_id]

        if not TEMPLATES_DIR.exists():
            return None

        for json_file in TEMPLATES_DIR.glob("*.json"):
            if json_file.name.startswith("_"):
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                meta = data.get("template_metadata", {})
                if meta.get("id") == template_id:
                    self._builtin_data_cache[template_id] = data
                    return data
            except (json.JSONDecodeError, KeyError):
                continue
        return None
