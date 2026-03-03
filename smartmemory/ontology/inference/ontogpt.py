"""
OntoGPTIntegration wrapper for OntologyManager.

Phase 1: Conditional integration that attempts to use OntoGPT when configured,
with graceful fallback to the existing frequency-based inference.

Future phases will implement:
- Template management and schema-driven mapping to internal Ontology
- Validation against existing ontologies
- Confidence aggregation and caching
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple

from smartmemory.ontology.models import Ontology, EntityTypeDefinition, RelationshipTypeDefinition


class OntoGPTInferenceEngine:
    """Thin wrapper around OntoGPT to support optional ontology inference.

    Note: This wrapper currently uses the ontogpt CLI if available. A direct
    Python API may be integrated later as it stabilizes upstream.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.logger = logging.getLogger(__name__)

        # Core settings
        self.provider: str = self.config.get("provider", os.getenv("ONTOGPT_PROVIDER", "openai"))
        self.model: str = self.config.get("models", os.getenv("ONTOGPT_MODEL", ""))
        self.template: str = self.config.get("template", os.getenv("ONTOGPT_TEMPLATE", ""))
        self.template_dir: str = self.config.get("template_dir", os.getenv("ONTOGPT_TEMPLATE_DIR", ""))
        self.target_class: str = self.config.get("target_class", os.getenv("ONTOGPT_TARGET_CLASS", ""))

        # Optional API routing (e.g., Azure, custom proxies)
        self.api_base: str = self.config.get("api_base", os.getenv("ONTOGPT_API_BASE", ""))
        self.api_version: str = self.config.get("api_version", os.getenv("ONTOGPT_API_VERSION", ""))

        # Inference behavior
        self.confidence_threshold: float = float(
            self.config.get("confidence_threshold", os.getenv("ONTOLOGY_CONFIDENCE_THRESHOLD", 0.6))
        )

    # ---------------------- Public API ----------------------
    def infer_ontology_from_extractions(
            self, extraction_history: List[Dict[str, Any]], ontology_name: str = "inferred"
    ) -> Optional[Ontology]:
        """Attempt to infer an ontology using OntoGPT.

        Phase 2: Build a textual corpus from extraction history, run OntoGPT,
        parse the output (JSON or LinkML-like), and map to internal Ontology.
        If anything fails, return None to signal fallback to basic inference.
        """
        if not self._cli_available():
            raise RuntimeError(
                "OntoGPT CLI not found on PATH. Install with 'pip install ontogpt' to enable OntoGPT inference."
            )
        if not self.template:
            raise RuntimeError(
                "OntoGPT template not configured. Set config.ontology.inference.ontogpt.template or ONTOGPT_TEMPLATE."
            )

        try:
            corpus = self._compose_corpus_from_extractions(extraction_history)
            if not corpus.strip():
                self.logger.warning("OntoGPT inference: empty corpus from extraction history")
                return None

            raw = self.extract_entities_and_relationships(corpus)
            ontology = self._build_ontology_from_payload(raw, ontology_name)
            if ontology is None:
                self.logger.warning("OntoGPT inference: unable to parse payload, falling back")
                return None

            # Attach provenance
            ontology.created_by = "ontogpt"
            ontology.description = (
                f"Ontology inferred by OntoGPT (template={self.template}, models={self.model or 'default'}, "
                f"provider={self.provider}) from {len(extraction_history)} examples"
            )
            return ontology
        except Exception as e:
            self.logger.error("OntoGPT inference error: %s", e, exc_info=True)
            return None

    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """Run 'ontogpt extract' on provided text and return raw parsed JSON or stdout.

        This helper is for future phases; not used in Phase 1.
        """
        if not self._cli_available():
            raise RuntimeError("OntoGPT CLI not found on PATH.")
        if not self.template:
            raise RuntimeError("OntoGPT template not configured (template).")

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
            tmp.write(text)
            tmp.flush()
            input_path = tmp.name

        # Resolve template argument: prefer full path if template_dir is set
        tpl = self.template
        try:
            if tpl and not tpl.endswith((".yaml", ".yml")) and self.template_dir:
                cand1 = os.path.join(self.template_dir, f"{tpl}.yaml")
                cand2 = os.path.join(self.template_dir, f"{tpl}.yml")
                if os.path.exists(cand1):
                    tpl = cand1
                elif os.path.exists(cand2):
                    tpl = cand2
        except Exception:
            pass

        cmd = [
            "ontogpt",
            "extract",
            "-i",
            input_path,
            "-t",
            tpl,
            "-O",
            "json",
        ]
        if self.model:
            cmd += ["-m", self.model]
        # Provider / API routing may be handled via environment (litellm proxies)
        if self.api_base:
            cmd += ["--api-base", self.api_base]

        try:
            # Add target class if configured
            if self.target_class:
                cmd += ["-T", self.target_class]

            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stdout = proc.stdout.strip()
            try:
                return json.loads(stdout)
            except Exception:
                # Try YAML if available
                try:
                    import yaml  # type: ignore
                    return yaml.safe_load(stdout) or {"raw": stdout}
                except Exception:
                    return {"raw": stdout}
        except subprocess.CalledProcessError as cpe:
            stderr_msg = (cpe.stderr or "").strip()
            stdout_msg = (cpe.stdout or "").strip()
            raise RuntimeError(
                f"OntoGPT CLI failed (exit {cpe.returncode}).\nSTDERR:\n{stderr_msg}\nSTDOUT:\n{stdout_msg}"
            ) from cpe
        finally:
            try:
                os.unlink(input_path)
            except Exception:
                pass

    def validate_against_existing_ontologies(self, data: Dict[str, Any]) -> bool:
        """Placeholder for ontology validation with OntoGPT outputs (future phase)."""
        return True

    # ---------------------- Internals ----------------------
    def _cli_available(self) -> bool:
        return shutil.which("ontogpt") is not None

    # ---------------------- Helpers (Phase 2) ----------------------
    def _compose_corpus_from_extractions(self, extraction_history: List[Dict[str, Any]]) -> str:
        """Compose a lightweight textual corpus summarizing entities and relations.

        This allows us to feed structured extraction history into OntoGPT's text-driven CLI.
        """
        lines: List[str] = []
        for ex in extraction_history:
            # Entities
            for ent in ex.get("entities", []) or []:
                name = str(ent.get("name", "")).strip()
                etype = str(ent.get("type", "")).strip()
                props = ent.get("properties") or {} or {}
                if name and etype:
                    prop_str = "; ".join(f"{k}: {v}" for k, v in props.items())
                    if prop_str:
                        lines.append(f"{name} is a {etype}. Properties: {prop_str}.")
                    else:
                        lines.append(f"{name} is a {etype}.")
            # Relations
            for rel in ex.get("relations", []) or []:
                src = str(rel.get("source", "")).strip()
                tgt = str(rel.get("target", "")).strip()
                rtype = str(rel.get("relation_type", "")).strip()
                if src and tgt and rtype:
                    rlabel = self._normalize_relation_label(rtype)
                    lines.append(f"{src} {rlabel} {tgt}.")
        return "\n".join(lines)

    def _normalize_relation_label(self, label: str) -> str:
        """Normalize a relation label like 'DEVELOPED_BY' -> 'developed by'."""
        s = label.strip()
        if not s:
            return s
        # Replace underscores with spaces, lower-case
        s = re.sub(r"[_]+", " ", s).strip().lower()
        return s

    def _build_ontology_from_payload(self, payload: Any, ontology_name: str) -> Optional[Ontology]:
        """Build an Ontology instance from OntoGPT output payload.

        Supports multiple shapes:
        - Dict with 'entities'/'relations'
        - LinkML-like schema with 'classes' and 'slots'
        - Fallback to None if not recognized
        """
        try:
            if isinstance(payload, dict) and payload:
                # Direct entities/relations shape
                if any(k in payload for k in ("entities", "relations", "relationships")):
                    return self._ontology_from_entities_relations(payload, ontology_name)

                # LinkML-like schema
                if any(k in payload for k in ("classes", "slots", "schema")):
                    return self._ontology_from_linkml_schema(payload, ontology_name)

                # Sometimes ontogpt may return under a key like 'data' or 'extracted_object'
                for key in ("data", "result", "extracted_object", "extracted_objects"):
                    sub = payload.get(key)
                    if isinstance(sub, dict):
                        o = self._build_ontology_from_payload(sub, ontology_name)
                        if o:
                            return o
                    if isinstance(sub, list) and sub and isinstance(sub[0], dict):
                        # Try merge multiple dicts
                        for item in sub:
                            o = self._build_ontology_from_payload(item, ontology_name)
                            if o:
                                return o

            # If payload is raw string, try parse JSON/YAML
            if isinstance(payload, str):
                try:
                    obj = json.loads(payload)
                    return self._build_ontology_from_payload(obj, ontology_name)
                except Exception:
                    try:
                        import yaml  # type: ignore
                        obj = yaml.safe_load(payload)
                        if obj:
                            return self._build_ontology_from_payload(obj, ontology_name)
                    except Exception:
                        return None
            return None
        except Exception as e:
            self.logger.error("Failed to build ontology from payload: %s", e, exc_info=True)
            return None

    def _ontology_from_entities_relations(self, data: Dict[str, Any], ontology_name: str) -> Optional[Ontology]:
        """Construct ontology using an 'entities'/'relations' style payload."""
        try:
            entities = data.get("entities") or data.get("entity") or []
            relations = data.get("relations") or data.get("relationships") or []

            # If nested under another key, try to find
            if not entities and isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, dict):
                        entities = entities or v.get("entities") or []
                        relations = relations or v.get("relations") or v.get("relationships") or []

            if not entities and not relations:
                return None

            # Map name->type from entities
            name_to_type: Dict[str, str] = {}
            type_props: Dict[str, Dict[str, int]] = {}
            type_examples: Dict[str, List[str]] = {}

            for ent in entities:
                etype = str(ent.get("type", "")).strip().lower()
                name = str(ent.get("name", "")).strip()
                props = ent.get("properties") or {} or {}
                if not etype or not name:
                    continue
                name_to_type[name] = etype
                type_props.setdefault(etype, {})
                type_examples.setdefault(etype, [])
                type_examples[etype].append(name)
                for p in props.keys():
                    type_props[etype][p] = type_props[etype].get(p, 0) + 1

            # Relationship aggregates
            rel_types: Dict[str, Dict[str, Any]] = {}
            for rel in relations:
                rtype = str(rel.get("relation_type", rel.get("type", ""))).strip().lower()
                src = str(rel.get("source", "")).strip()
                tgt = str(rel.get("target", "")).strip()
                conf = rel.get("confidence")
                if not rtype or not src or not tgt:
                    continue
                entry = rel_types.setdefault(rtype, {"count": 0, "source_types": set(), "target_types": set(), "conf": []})
                entry["count"] += 1
                if src in name_to_type:
                    entry["source_types"].add(name_to_type[src])
                if tgt in name_to_type:
                    entry["target_types"].add(name_to_type[tgt])
                if isinstance(conf, (int, float)):
                    entry["conf"].append(float(conf))

            ontology = Ontology(ontology_name)
            ontology.created_by = "ontogpt"

            # Entity type definitions
            for etype, props_counts in type_props.items():
                total = max(1, len(type_examples.get(etype, [])))
                required = {p for p, c in props_counts.items() if (c / total) >= max(0.6, float(self.confidence_threshold))}
                ontology.add_entity_type(
                    entity_type=self._make_entity_type(
                        name=etype,
                        description="OntoGPT-inferred entity type from corpus",
                        properties={p: "string" for p in props_counts.keys()},
                        required=required,
                        examples=type_examples.get(etype, [])[:5],
                        confidence=min(1.0, total / 10.0),
                    )
                )

            # Relationship type definitions
            for rtype, agg in rel_types.items():
                avg_conf = sum(agg["conf"]) / len(agg["conf"]) if agg["conf"] else 0.8
                ontology.add_relationship_type(
                    rel_type=self._make_relationship_type(
                        name=self._snake_case(rtype),
                        description="OntoGPT-inferred relationship type",
                        source_types=set(agg["source_types"]),
                        target_types=set(agg["target_types"]),
                        confidence=avg_conf,
                    )
                )

            if not ontology.entity_types and not ontology.relationship_types:
                return None
            return ontology
        except Exception as e:
            self.logger.error("Failed to construct ontology from entities/relations: %s", e, exc_info=True)
            return None

    def _ontology_from_linkml_schema(self, data: Dict[str, Any], ontology_name: str) -> Optional[Ontology]:
        """Construct ontology from a LinkML-like schema payload (classes/slots)."""
        try:
            # Navigate if wrapped
            if "schema" in data and isinstance(data["schema"], dict):
                data = data["schema"]

            classes: Dict[str, Any] = data.get("classes") or {} or {}
            slots: Dict[str, Any] = data.get("slots") or {} or {}
            if not classes and isinstance(data, dict):
                # Sometimes nested under different key
                for v in data.values():
                    if isinstance(v, dict) and ("classes" in v or "slots" in v):
                        classes = v.get("classes", classes)
                        slots = v.get("slots", slots)

            if not classes:
                return None

            ontology = Ontology(ontology_name)
            ontology.created_by = "ontogpt"

            # Helper to resolve a slot to range and type
            def resolve_slot(slot_name: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
                sd = {}
                if slot_name in slots and isinstance(slots[slot_name], dict):
                    sd = slots[slot_name]
                rng = sd.get("range")
                return (slot_name, rng, sd)

            # Build entity and relationships
            rel_agg: Dict[str, Dict[str, Any]] = {}
            for cls_name, cls in classes.items():
                cls_props: Dict[str, str] = {}
                required: List[str] = []
                examples: List[str] = []

                # Attributes may be represented in different keys
                slot_names: List[str] = []
                if isinstance(cls, dict):
                    if isinstance(cls.get("attributes"), dict):
                        slot_names.extend(list(cls["attributes"].keys()))
                    # 'slots' as list of slot names
                    if isinstance(cls.get("slots"), list):
                        slot_names.extend(cls.get("slots", []))
                    # slot_usage with overrides
                    if isinstance(cls.get("slot_usage"), dict):
                        slot_names.extend(list(cls["slot_usage"].keys()))

                # Deduplicate
                slot_names = list(dict.fromkeys(x for x in slot_names if isinstance(x, str)))

                for s in slot_names:
                    sname, rng, sd = resolve_slot(s)
                    if not sname:
                        continue
                    # If range is another class, treat as relationship
                    if isinstance(rng, str) and rng in classes:
                        key = self._snake_case(sname)
                        agg = rel_agg.setdefault(key, {"source": set(), "target": set()})
                        agg["source"].add(cls_name.lower())
                        agg["target"].add(rng.lower())
                    else:
                        # Treat as attribute/property
                        dtype = str(rng or sd.get("range", "string") or "string").lower()
                        if dtype in ("integer", "int", "number", "float", "double"):
                            dtype = "integer" if dtype in ("integer", "int") else "number"
                        elif dtype in ("datetime", "date", "time"):
                            dtype = "date"
                        elif dtype in ("boolean", "bool"):
                            dtype = "boolean"
                        else:
                            dtype = "string"
                        cls_props[sname] = dtype
                        if sd.get("required") is True:
                            required.append(sname)

                ontology.add_entity_type(
                    entity_type=self._make_entity_type(
                        name=cls_name.lower(),
                        description="OntoGPT (LinkML) class",
                        properties=cls_props,
                        required=set(required),
                        examples=examples,
                        confidence=0.85,
                    )
                )

            # Add relationship types aggregated from slots that point to other classes
            for rname, agg in rel_agg.items():
                ontology.add_relationship_type(
                    rel_type=self._make_relationship_type(
                        name=rname,
                        description="OntoGPT (LinkML) relationship inferred from slot range",
                        source_types=set(agg["source"]),
                        target_types=set(agg["target"]),
                        confidence=0.85,
                    )
                )

            if not ontology.entity_types and not ontology.relationship_types:
                return None
            return ontology
        except Exception as e:
            self.logger.error("Failed to construct ontology from LinkML schema: %s", e, exc_info=True)
            return None

    def _make_entity_type(
            self,
            name: str,
            description: str,
            properties: Dict[str, str],
            required: set,
            examples: List[str],
            confidence: float,
    ) -> Any:

        return EntityTypeDefinition(
            name=name,
            description=description,
            properties=properties,
            required_properties=set(required),
            parent_types=set(),
            aliases=set(),
            examples=examples,
            created_by="ontogpt",
            created_at=datetime.now(UTC),
            confidence=confidence,
        )

    def _make_relationship_type(
            self,
            name: str,
            description: str,
            source_types: set,
            target_types: set,
            confidence: float,
    ) -> Any:

        return RelationshipTypeDefinition(
            name=name,
            description=description,
            source_types=source_types,
            target_types=target_types,
            properties={},
            bidirectional=False,
            aliases=set(),
            examples=[],
            created_by="ontogpt",
            created_at=datetime.now(UTC),
            confidence=confidence,
        )

    def _snake_case(self, s: str) -> str:
        s = re.sub(r"[^0-9A-Za-z]+", "_", s)
        s = re.sub(r"_{2,}", "_", s)
        return s.strip("_").lower()
