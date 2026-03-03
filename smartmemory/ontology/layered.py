"""Merged read view of a base + overlay ontology pair."""

import copy
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Set

from smartmemory.ontology.models import (
    EntityTypeDefinition,
    LayerDiff,
    Ontology,
    RelationshipTypeDefinition,
    OntologyRule,
)


class LayeredOntology:
    """Provides a merged read view of an overlay ontology layered on a base.

    Merge semantics:
    - Entity types: union of base + overlay; overlay wins on name conflict.
    - Relationship types: union of base + overlay; overlay wins on name conflict.
    - Rules: overlay only (base rules are NOT inherited).
    - Hidden types: base entity types in the hidden set are excluded.
    - Override granularity: entity-level (not field-level merge).
    """

    def __init__(
        self,
        overlay: Ontology,
        base: Optional[Ontology] = None,
        pinned_version: Optional[str] = None,
        hidden_types: Optional[Set[str]] = None,
    ) -> None:
        self.overlay = overlay
        self.base = base
        self.pinned_version = pinned_version
        self.hidden_types: Set[str] = {t.lower() for t in (hidden_types or set())}

    # -- Proxy properties for compatibility --

    @property
    def id(self) -> str:
        return self.overlay.id

    @property
    def name(self) -> str:
        return self.overlay.name

    @property
    def version(self) -> str:
        return self.overlay.version

    @property
    def tenant_id(self) -> str:
        return self.overlay.tenant_id

    # -- Merged read accessors --

    @property
    def entity_types(self) -> Dict[str, EntityTypeDefinition]:
        """Base + overlay entity types. Overlay wins on conflict; hidden types excluded."""
        merged: Dict[str, EntityTypeDefinition] = {}
        if self.base:
            for name, defn in self.base.entity_types.items():
                if name.lower() not in self.hidden_types:
                    merged[name] = defn
        # Overlay wins — overwrites base entries on conflict
        for name, defn in self.overlay.entity_types.items():
            merged[name] = defn
        return merged

    @property
    def relationship_types(self) -> Dict[str, RelationshipTypeDefinition]:
        """Base + overlay relationship types. Overlay wins on conflict."""
        merged: Dict[str, RelationshipTypeDefinition] = {}
        if self.base:
            merged.update(self.base.relationship_types)
        merged.update(self.overlay.relationship_types)
        return merged

    @property
    def rules(self) -> Dict[str, OntologyRule]:
        """Overlay rules only — base rules are NOT inherited."""
        return dict(self.overlay.rules)

    # -- Lookup --

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """Look up entity type: overlay first, then base; hidden applies only to base types."""
        key = name.lower()
        # Overlay always wins — even if the type is in hidden_types
        overlay_hit = self.overlay.entity_types.get(key)
        if overlay_hit is not None:
            return overlay_hit
        # Hidden filter only applies to base-only types
        if key in self.hidden_types:
            return None
        if self.base:
            return self.base.entity_types.get(key)
        return None

    def get_relationship_type(self, name: str) -> Optional[RelationshipTypeDefinition]:
        """Look up relationship type: overlay first, then base."""
        key = name.lower()
        overlay_hit = self.overlay.relationship_types.get(key)
        if overlay_hit is not None:
            return overlay_hit
        if self.base:
            return self.base.relationship_types.get(key)
        return None

    # -- Provenance --

    def get_provenance(self, type_name: str) -> str:
        """Return provenance label for a type name.

        Returns one of: "local", "base", "override", "hidden", "unknown".
        """
        key = type_name.lower()
        in_overlay = key in self.overlay.entity_types
        in_base = self.base is not None and key in self.base.entity_types

        # Override takes precedence — overlay wins even if type is hidden
        if in_overlay and in_base:
            return "override"
        if in_overlay:
            return "local"
        if key in self.hidden_types and in_base:
            return "hidden"
        if in_base:
            return "base"
        return "unknown"

    def get_provenance_map(self) -> Dict[str, str]:
        """Provenance for ALL known type names (overlay + base + hidden)."""
        all_names: Set[str] = set()
        all_names.update(self.overlay.entity_types.keys())
        if self.base:
            all_names.update(self.base.entity_types.keys())
        return {name: self.get_provenance(name) for name in sorted(all_names)}

    # -- Diff & Detach --

    def compute_diff(self) -> LayerDiff:
        """Compute diff between base and overlay layers."""
        if not self.base:
            return LayerDiff(
                base_only=[],
                overlay_only=sorted(self.overlay.entity_types.keys()),
                overridden=[],
                hidden=[],
            )

        base_keys = set(self.base.entity_types.keys())
        overlay_keys = set(self.overlay.entity_types.keys())

        # Override takes precedence over hidden — a type in both overlay and hidden is "overridden"
        hidden_only = (base_keys & self.hidden_types) - overlay_keys
        return LayerDiff(
            base_only=sorted(base_keys - overlay_keys - self.hidden_types),
            overlay_only=sorted(overlay_keys - base_keys),
            overridden=sorted(base_keys & overlay_keys),
            hidden=sorted(hidden_only),
        )

    def detach(self) -> Ontology:
        """Flatten to a standalone ontology — copies visible base types into overlay, removes subscription."""
        data = self.overlay.to_dict()
        flat = Ontology.from_dict(data)

        # Copy in base entity types that aren't overridden or hidden
        if self.base:
            for name, defn in self.base.entity_types.items():
                if name not in flat.entity_types and name.lower() not in self.hidden_types:
                    flat.entity_types[name] = copy.deepcopy(defn)
            for name, defn in self.base.relationship_types.items():
                if name not in flat.relationship_types:
                    flat.relationship_types[name] = copy.deepcopy(defn)

        flat.subscription = None
        flat.is_base_layer = False
        flat.updated_at = datetime.now(UTC)
        return flat

    def to_dict(self) -> Dict[str, Any]:
        """Merged view as dict, with provenance map included."""
        result = self.overlay.to_dict()

        if self.base:
            base_dict = self.base.to_dict()

            # Merge in base entity types (overlay wins, hidden excluded)
            merged_entity_types = {}
            for name, defn in base_dict.get("entity_types", {}).items():
                if name.lower() not in self.hidden_types:
                    merged_entity_types[name] = defn
            for name, defn in result.get("entity_types", {}).items():
                merged_entity_types[name] = defn
            result["entity_types"] = merged_entity_types

            # Merge in base relationship types (overlay wins)
            merged_rel_types = {}
            for name, defn in base_dict.get("relationship_types", {}).items():
                merged_rel_types[name] = defn
            for name, defn in result.get("relationship_types", {}).items():
                merged_rel_types[name] = defn
            result["relationship_types"] = merged_rel_types

        result["provenance"] = self.get_provenance_map()
        return result
