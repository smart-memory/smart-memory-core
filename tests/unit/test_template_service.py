"""Tests for ontology template service."""

import json
from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.unit

from smartmemory.ontology.models import Ontology, TemplateInfo, TemplatePreview
from smartmemory.ontology.template_service import TemplateService, TEMPLATES_DIR
from smartmemory.stores.ontology import FileSystemOntologyStorage


@pytest.fixture
def tmp_storage(tmp_path):
    """Create a FileSystemOntologyStorage using a temp directory."""
    return FileSystemOntologyStorage(str(tmp_path / "ontologies"))


@pytest.fixture
def service(tmp_storage):
    """Create a TemplateService with temp storage."""
    return TemplateService(tmp_storage)


class TestListBuiltinTemplates:
    def test_loads_from_manifest(self, service):
        templates = service.list_builtin_templates()
        assert len(templates) == 3
        ids = {t.id for t in templates}
        assert ids == {"builtin:general", "builtin:software", "builtin:business"}

    def test_all_are_builtin(self, service):
        for t in service.list_builtin_templates():
            assert t.is_builtin is True
            assert t.created_by == "system"

    def test_caches_result(self, service):
        first = service.list_builtin_templates()
        second = service.list_builtin_templates()
        assert first is second

    def test_missing_manifest_returns_empty(self, tmp_storage):
        svc = TemplateService(tmp_storage)
        with patch("smartmemory.ontology.template_service.MANIFEST_FILE", Path("/nonexistent/_manifest.json")):
            svc._builtin_cache = None  # Reset cache
            result = svc.list_builtin_templates()
            assert result == []

    def test_entity_counts_match(self, service):
        for t in service.list_builtin_templates():
            assert t.entity_count > 0
            assert t.relationship_count > 0


class TestListCustomTemplates:
    def test_empty_when_no_templates(self, service):
        assert service.list_custom_templates("tenant-1") == []

    def test_finds_custom_templates(self, service, tmp_storage):
        ontology = Ontology("My Template")
        ontology.is_template = True
        ontology.domain = "custom"
        ontology.description = "A custom template"
        # FileSystemOntologyStorage doesn't store tenant_id natively,
        # so custom template listing won't find it without tenant metadata in the file.
        # This tests the filtering logic with a mock that provides tenant_id.
        tmp_storage.save_ontology(ontology)

        # Patch list_ontologies to include tenant_id in the listing
        original_list = tmp_storage.list_ontologies

        def patched_list():
            result = original_list()
            for item in result:
                item["tenant_id"] = "tenant-1"
            return result

        tmp_storage.list_ontologies = patched_list
        templates = service.list_custom_templates("tenant-1")
        assert len(templates) == 1
        assert templates[0].name == "My Template"
        assert templates[0].is_builtin is False
        assert templates[0].id.startswith("custom:")

    def test_excludes_other_tenant(self, service, tmp_storage):
        ontology = Ontology("Other Tenant Template")
        ontology.is_template = True
        tmp_storage.save_ontology(ontology)

        original_list = tmp_storage.list_ontologies

        def patched_list():
            result = original_list()
            for item in result:
                item["tenant_id"] = "tenant-2"
            return result

        tmp_storage.list_ontologies = patched_list
        assert service.list_custom_templates("tenant-1") == []

    def test_excludes_non_templates(self, service, tmp_storage):
        ontology = Ontology("Regular Ontology")
        ontology.is_template = False
        tmp_storage.save_ontology(ontology)

        original_list = tmp_storage.list_ontologies

        def patched_list():
            result = original_list()
            for item in result:
                item["tenant_id"] = "tenant-1"
            return result

        tmp_storage.list_ontologies = patched_list
        assert service.list_custom_templates("tenant-1") == []


class TestListAllTemplates:
    def test_combines_builtin_and_custom(self, service):
        templates = service.list_all_templates("tenant-1")
        # Should have at least the 3 built-in templates
        assert len(templates) >= 3
        names = [t.name for t in templates]
        assert "General Purpose" in names
        assert "Software Engineering" in names

    def test_sorted_by_name(self, service):
        templates = service.list_all_templates("tenant-1")
        names = [t.name for t in templates]
        assert names == sorted(names)


class TestGetTemplatePreview:
    def test_builtin_preview(self, service):
        preview = service.get_template_preview("builtin:general", "tenant-1")
        assert preview is not None
        assert isinstance(preview, TemplatePreview)
        assert preview.id == "builtin:general"
        assert preview.name == "General Purpose"
        assert len(preview.entity_types) == 12
        assert len(preview.relationship_types) == 8
        assert preview.is_builtin is True

    def test_software_preview(self, service):
        preview = service.get_template_preview("builtin:software", "tenant-1")
        assert preview is not None
        assert preview.name == "Software Engineering"
        assert len(preview.entity_types) == 15
        assert len(preview.relationship_types) == 10

    def test_business_preview(self, service):
        preview = service.get_template_preview("builtin:business", "tenant-1")
        assert preview is not None
        assert preview.name == "Business & Finance"
        assert len(preview.entity_types) == 14
        assert len(preview.relationship_types) == 9

    def test_nonexistent_returns_none(self, service):
        assert service.get_template_preview("builtin:nonexistent", "tenant-1") is None

    def test_custom_preview(self, service, tmp_storage):
        ontology = Ontology("Custom")
        ontology.domain = "test"
        ontology.description = "Test custom"
        ontology.is_template = True
        tmp_storage.save_ontology(ontology)

        preview = service.get_template_preview(f"custom:{ontology.id}", "tenant-1")
        assert preview is not None
        assert preview.name == "Custom"
        assert preview.is_builtin is False


class TestCloneTemplate:
    def test_clone_builtin(self, service, tmp_storage):
        new_id = service.clone_template(
            template_id="builtin:general",
            target_name="My Workspace Ontology",
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert new_id

        loaded = tmp_storage.load_ontology(new_id)
        assert loaded is not None
        assert loaded.name == "My Workspace Ontology"
        assert loaded.created_by == "user-1"
        assert loaded.source_template == "builtin:general"
        assert loaded.is_template is False
        assert len(loaded.entity_types) == 12
        assert len(loaded.relationship_types) == 8

    def test_clone_creates_independent_copy(self, service, tmp_storage):
        id1 = service.clone_template("builtin:software", "Clone 1", "t1", "u1")
        id2 = service.clone_template("builtin:software", "Clone 2", "t1", "u2")
        assert id1 != id2

        clone1 = tmp_storage.load_ontology(id1)
        clone2 = tmp_storage.load_ontology(id2)
        assert clone1.name != clone2.name
        assert clone1.id != clone2.id

    def test_clone_nonexistent_raises(self, service):
        with pytest.raises(ValueError, match="Template not found"):
            service.clone_template("builtin:nonexistent", "Name", "t1", "u1")

    def test_clone_preserves_entity_types(self, service, tmp_storage):
        new_id = service.clone_template("builtin:software", "SW Clone", "t1", "u1")
        loaded = tmp_storage.load_ontology(new_id)
        assert "service" in loaded.entity_types
        assert "api" in loaded.entity_types
        assert "database" in loaded.entity_types


class TestSaveAsTemplate:
    def test_save_existing_as_template(self, service, tmp_storage):
        source = Ontology("Source Ontology")
        source.domain = "medical"
        source.description = "Medical ontology"
        from smartmemory.ontology.models import EntityTypeDefinition
        from datetime import datetime

        source.add_entity_type(
            EntityTypeDefinition(
                name="Patient",
                description="A medical patient",
                properties={"name": "str"},
                required_properties={"name"},
                parent_types=set(),
                aliases=set(),
                examples=["John Doe"],
                created_by="user",
                created_at=datetime.now(UTC),
            )
        )
        tmp_storage.save_ontology(source)

        template_id = service.save_as_template(
            registry_id=source.id,
            template_name="Medical Template",
            description="Reusable medical ontology",
            tenant_id="tenant-1",
            user_id="user-1",
        )
        assert template_id

        loaded = tmp_storage.load_ontology(template_id)
        assert loaded.name == "Medical Template"
        assert loaded.is_template is True
        assert loaded.created_by == "user-1"
        assert "patient" in loaded.entity_types

    def test_save_nonexistent_raises(self, service):
        with pytest.raises(ValueError, match="Ontology not found"):
            service.save_as_template("nonexistent", "Name", "Desc", "t1", "u1")


class TestDeleteCustomTemplate:
    def test_delete_custom(self, service, tmp_storage):
        ontology = Ontology("To Delete")
        ontology.is_template = True
        tmp_storage.save_ontology(ontology)

        result = service.delete_custom_template(f"custom:{ontology.id}", "tenant-1")
        assert result is True
        assert tmp_storage.load_ontology(ontology.id) is None

    def test_delete_builtin_raises(self, service):
        with pytest.raises(ValueError, match="Cannot delete built-in"):
            service.delete_custom_template("builtin:general", "tenant-1")

    def test_delete_nonexistent_returns_false(self, service):
        result = service.delete_custom_template("custom:nonexistent-id", "tenant-1")
        assert result is False


class TestTemplateModels:
    def test_template_info_to_dict(self):
        info = TemplateInfo(
            id="test:1",
            name="Test",
            domain="test",
            description="A test",
            version="1.0.0",
            entity_count=5,
            relationship_count=3,
            is_builtin=True,
        )
        d = info.to_dict()
        assert d["id"] == "test:1"
        assert d["entity_count"] == 5
        assert d["is_builtin"] is True

    def test_template_preview_inherits_info(self):
        preview = TemplatePreview(
            id="test:1",
            name="Test",
            domain="test",
            description="A test",
            version="1.0.0",
            entity_count=5,
            relationship_count=3,
            is_builtin=True,
            entity_types=["person", "org"],
            relationship_types=["works_at"],
            rules_count=0,
        )
        d = preview.to_dict()
        assert d["entity_types"] == ["person", "org"]
        assert d["relationship_types"] == ["works_at"]
        assert d["rules_count"] == 0


class TestBuiltinTemplateIntegrity:
    """Validate that the built-in template JSON files are well-formed."""

    def test_general_template_parseable(self):
        data = self._load_template("general.json")
        ontology = Ontology.from_dict(data)
        assert ontology.name == "General Purpose"
        assert len(ontology.entity_types) == 12

    def test_software_template_parseable(self):
        data = self._load_template("software.json")
        ontology = Ontology.from_dict(data)
        assert ontology.name == "Software Engineering"
        assert len(ontology.entity_types) == 15

    def test_business_template_parseable(self):
        data = self._load_template("business.json")
        ontology = Ontology.from_dict(data)
        assert ontology.name == "Business & Finance"
        assert len(ontology.entity_types) == 14

    def test_manifest_matches_templates(self):
        with open(TEMPLATES_DIR / "_manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest) == 3
        for entry in manifest:
            data = self._load_template_by_id(entry["id"])
            assert data is not None, f"Template file not found for {entry['id']}"
            assert len(data.get("entity_types", {})) == entry["entity_count"]
            assert len(data.get("relationship_types", {})) == entry["relationship_count"]

    @staticmethod
    def _load_template(filename: str) -> dict:
        with open(TEMPLATES_DIR / filename) as f:
            return json.load(f)

    @staticmethod
    def _load_template_by_id(template_id: str) -> dict | None:
        for json_file in TEMPLATES_DIR.glob("*.json"):
            if json_file.name.startswith("_"):
                continue
            with open(json_file) as f:
                data = json.load(f)
            if data.get("template_metadata", {}).get("id") == template_id:
                return data
        return None
