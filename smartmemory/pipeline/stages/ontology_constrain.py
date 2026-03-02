"""OntologyConstrain stage — merge, validate, and filter extraction results."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Set

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph
    from smartmemory.observability.events import RedisStreamQueue
    from smartmemory.ontology.entity_pair_cache import EntityPairCache
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

_RELATION_STOPWORDS: set[str] = {"is", "has", "does", "was", "are", "were", "been", "be"}


class OntologyConstrainStage:
    """Merge ruler + LLM entities, validate types against ontology, filter relations."""

    def __init__(
        self,
        ontology_graph: OntologyGraph,
        promotion_queue: RedisStreamQueue | None = None,
        entity_pair_cache: EntityPairCache | None = None,
        ontology_registry=None,
        ontology_model=None,
        relation_normalizer=None,
        type_pair_validator=None,
    ):
        self._ontology = ontology_graph
        self._promotion_queue = promotion_queue
        self._entity_pair_cache = entity_pair_cache
        self._registry = ontology_registry
        self._ontology_model = ontology_model
        self._relation_normalizer = relation_normalizer
        self._type_pair_validator = type_pair_validator

    @property
    def name(self) -> str:
        return "ontology_constrain"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        constrain_cfg = config.extraction.constrain
        promotion_cfg = config.extraction.promotion

        # OL-2: Capture ontology version from registry
        registry_id = ""
        version = ""
        if self._registry:
            try:
                reg = self._registry.get_registry("default")
                registry_id = reg.get("id", "default")
                version = reg.get("current_version", "")
            except Exception:
                pass

        # Step 1: Merge ruler + LLM entities
        merged = self._merge_entities(state.ruler_entities, state.llm_entities)

        # Step 2: Validate entity types against ontology + track frequency
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        promotion_candidates: List[Dict[str, Any]] = []
        accepted_names: Set[str] = set()
        unresolved: List[Dict[str, Any]] = []

        for entity in merged:
            entity_type = self._get_entity_type(entity)
            name = self._get_entity_name(entity)
            confidence = self._get_confidence(entity)
            status = self._ontology.get_type_status(entity_type.title())

            if status in ("seed", "confirmed", "provisional"):
                accepted.append(entity)
                accepted_names.add(name.lower())
                # Track frequency for accepted entities
                self._ontology.increment_frequency(entity_type.title(), confidence)
            elif confidence >= constrain_cfg.confidence_threshold:
                # Unknown type with sufficient confidence — add as provisional
                self._ontology.add_provisional(entity_type.title())
                self._ontology.increment_frequency(entity_type.title(), confidence)
                promotion_candidates.append(entity)
                accepted.append(entity)
                accepted_names.add(name.lower())
            else:
                # OL-4: Build structured unresolved entity
                rejected.append(entity)
                unresolved.append(
                    {
                        "entity_name": name,
                        "attempted_type": entity_type,
                        "reason": "unknown_type" if status is None else "low_confidence",
                        "confidence": confidence,
                        "source_content": (state.text or "")[:200],
                    }
                )

        # OL-3: Validate property constraints on accepted entities
        constraint_violations: List[Dict[str, Any]] = []
        if self._ontology_model:
            still_accepted: List[Dict[str, Any]] = []
            for entity in accepted:
                et = self._get_entity_type(entity)
                violations = self._validate_constraints(entity, et)
                hard = [v for v in violations if v["kind"] == "hard"]
                if hard:
                    name = self._get_entity_name(entity)
                    rejected.append(entity)
                    accepted_names.discard(name.lower())
                    unresolved.append(
                        {
                            "entity_name": name,
                            "attempted_type": et,
                            "reason": "constraint_violation",
                            "confidence": self._get_confidence(entity),
                            "source_content": (state.text or "")[:200],
                        }
                    )
                else:
                    still_accepted.append(entity)
                    constraint_violations.extend(violations)
            accepted = still_accepted

        # Step 2b: Inject cached entity-pair relations
        if self._entity_pair_cache and len(accepted) >= 2:
            cached_relations = self._lookup_cached_relations(accepted, state.workspace_id)
            if cached_relations:
                state = replace(state, llm_relations=list(state.llm_relations) + cached_relations)
                # Record avoided tokens — each cached relation pair saves ~100 tokens of LLM work
                if state.token_tracker:
                    state.token_tracker.record_avoided(
                        "ontology_constrain",
                        100 * len(cached_relations),
                        reason="cache_hit",
                    )

        # Step 3: Filter relations — keep only those with both endpoints accepted
        valid_relations = self._filter_relations(state.llm_relations, accepted, accepted_names)

        # Step 4: Apply limits
        accepted = accepted[: constrain_cfg.max_entities]
        valid_relations = valid_relations[: constrain_cfg.max_relations]

        # Step 5: Enqueue promotion candidates (async) or auto-promote (fallback)
        if not promotion_cfg.require_approval:
            self._handle_promotion(
                promotion_candidates,
                workspace_id=state.workspace_id,
                source_memory_id=state.item_id,
            )

        return replace(
            state,
            entities=accepted,
            relations=valid_relations,
            rejected=rejected,
            promotion_candidates=promotion_candidates,
            ontology_registry_id=registry_id,
            ontology_version=version,
            constraint_violations=constraint_violations,
            unresolved_entities=unresolved,
        )

    def _handle_promotion(
        self,
        candidates: List[Dict[str, Any]],
        workspace_id: str | None = None,
        source_memory_id: str | None = None,
    ) -> None:
        """Enqueue candidates to promotion stream, or auto-promote as fallback."""
        for candidate in candidates:
            entity_type = self._get_entity_type(candidate)
            entity_name = self._get_entity_name(candidate)
            confidence = self._get_confidence(candidate)

            if self._promotion_queue:
                try:
                    self._promotion_queue.enqueue(
                        {
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "confidence": confidence,
                            "workspace_id": workspace_id or "default",
                            "source_memory_id": source_memory_id,
                        }
                    )
                    continue
                except Exception as e:
                    logger.debug("Promotion queue unavailable, falling back to inline: %s", e)

            # Fallback: direct promote
            self._ontology.promote(entity_type.title())

    def _lookup_cached_relations(
        self, accepted: List[Dict[str, Any]], workspace_id: str | None
    ) -> List[Dict[str, Any]]:
        """Look up cached entity-pair relations for accepted entities."""
        cached: List[Dict[str, Any]] = []
        ws = workspace_id or "default"
        names = [self._get_entity_name(e) for e in accepted]
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                try:
                    relations = self._entity_pair_cache.lookup(name_a, name_b, ws)  # type: ignore[union-attr]
                    if relations:
                        for rel in relations:
                            cached.append(
                                {
                                    "source_id": name_a.lower(),
                                    "target_id": name_b.lower(),
                                    "relation_type": rel.get("relation_type", "RELATED"),
                                    "confidence": rel.get("confidence", 0.5),
                                    "source": "entity_pair_cache",
                                }
                            )
                except Exception:
                    pass
        return cached

    def _merge_entities(
        self,
        ruler_entities: List[Any],
        llm_entities: List[Any],
    ) -> List[Dict[str, Any]]:
        """Merge ruler and LLM entities by name. Higher confidence wins.

        When merging, preserve ``item_id`` from the LLM entity so that
        downstream relation filtering can match ``source_id`` / ``target_id``
        (SHA256 hashes assigned by the LLM extractor) to accepted entities.
        """
        merged: Dict[str, Dict[str, Any]] = {}

        # Index ruler entities first (preferred type source)
        for entity in ruler_entities:
            name = self._get_entity_name(entity).lower()
            merged[name] = self._normalize_entity(entity, source="ruler")

        # Merge LLM entities
        for entity in llm_entities:
            name = self._get_entity_name(entity).lower()
            normalized = self._normalize_entity(entity, source="llm")
            if name in merged:
                existing = merged[name]
                # Higher confidence wins — including entity_type
                if normalized.get("confidence", 0) > existing.get("confidence", 0):
                    existing["confidence"] = normalized["confidence"]
                    existing["entity_type"] = normalized.get("entity_type", existing.get("entity_type", "concept"))
                # Always preserve the LLM item_id — needed for relation ID resolution
                if normalized.get("item_id") and not existing.get("item_id"):
                    existing["item_id"] = normalized["item_id"]
            else:
                merged[name] = normalized

        return list(merged.values())

    def _normalize_entity(self, entity: Any, source: str) -> Dict[str, Any]:
        """Convert any entity format to a standard dict."""
        if isinstance(entity, dict):
            result = dict(entity)
            result.setdefault("source", source)
            return result

        # Handle MemoryItem objects from LLM extractor
        name = getattr(entity, "content", "") or ""
        metadata = getattr(entity, "metadata", {}) or {}
        return {
            "name": metadata.get("name", name),
            "entity_type": metadata.get("entity_type", "concept"),
            "confidence": metadata.get("confidence", 0.5),
            "source": source,
            "item_id": getattr(entity, "item_id", None),
        }

    def _get_entity_name(self, entity: Any) -> str:
        """Extract name from dict or MemoryItem."""
        if isinstance(entity, dict):
            return entity.get("name", "")
        metadata = getattr(entity, "metadata", {}) or {}
        return metadata.get("name", getattr(entity, "content", ""))

    def _get_entity_type(self, entity: Any) -> str:
        """Extract entity type from dict or MemoryItem."""
        if isinstance(entity, dict):
            return entity.get("entity_type", "concept")
        metadata = getattr(entity, "metadata", {}) or {}
        return metadata.get("entity_type", "concept")

    def _get_confidence(self, entity: Any) -> float:
        """Extract confidence from dict or MemoryItem."""
        if isinstance(entity, dict):
            return float(entity.get("confidence", 0.5))
        metadata = getattr(entity, "metadata", {}) or {}
        return float(metadata.get("confidence", 0.5))

    def _filter_relations(
        self,
        relations: List[Any],
        accepted_entities: List[Dict[str, Any]],
        accepted_names: Set[str],
    ) -> List[Dict[str, Any]]:
        """Keep only relations where both endpoints are in accepted entities."""
        # Build a set of accepted entity IDs
        accepted_ids: Set[str] = set()
        for entity in accepted_entities:
            eid = entity.get("item_id") or entity.get("name", "").lower()
            accepted_ids.add(eid)

        valid = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source = rel.get("source_id")
            target = rel.get("target_id")
            if not source or not target:
                logger.debug("Skipping relation with missing IDs: %s", rel)
                continue
            if source in accepted_ids and target in accepted_ids:
                valid.append(rel)

        # ONTO-PUB-3: Normalize predicates, validate type-pairs, score plausibility
        if self._relation_normalizer is not None or self._type_pair_validator is not None:
            valid = self._normalize_and_score_relations(valid, accepted_entities)

        return valid

    def _normalize_and_score_relations(
        self,
        valid: List[Dict[str, Any]],
        accepted: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize predicates, validate type-pairs, compute plausibility scores."""
        from smartmemory.relations.scorer import PlausibilityScorer

        scorer = PlausibilityScorer()

        # Build ID → entity type map from accepted entities
        id_to_type: Dict[str, str] = {}
        for entity in accepted:
            eid = entity.get("item_id") or entity.get("name", "").lower()
            etype = entity.get("entity_type", "concept").lower()
            id_to_type[eid] = etype

        enriched: List[Dict] = []
        for rel in valid:
            raw_predicate = rel.get("raw_predicate") or rel.get("relation_type", "related_to")

            # Normalize predicate
            if self._relation_normalizer is not None:
                canonical_type, norm_conf = self._relation_normalizer.normalize(raw_predicate)
            else:
                canonical_type = rel.get("relation_type", "related_to")
                norm_conf = 1.0

            # Validate type-pair
            src_type = id_to_type.get(rel.get("source_id", ""), "concept")
            tgt_type = id_to_type.get(rel.get("target_id", ""), "concept")

            if self._type_pair_validator is not None:
                is_valid, pair_score = self._type_pair_validator.validate(src_type, canonical_type, tgt_type)
            else:
                is_valid = True
                pair_score = 1.0

            # In strict mode, drop invalid relations entirely
            if not is_valid:
                continue

            # Compute plausibility
            plausibility = scorer.score(norm_conf, pair_score)

            # Attach metadata
            rel["canonical_type"] = canonical_type
            if "raw_predicate" not in rel:
                rel["raw_predicate"] = rel.get("relation_type", "related_to")
            rel["normalization_confidence"] = norm_conf
            rel["plausibility_score"] = plausibility
            # Update relation_type to canonical (for FalkorDB edge label)
            rel["relation_type"] = canonical_type

            # CORE-EXT-1c: Track relation type frequency and novel labels
            if self._ontology is not None:
                if norm_conf > 0.0:
                    self._ontology.increment_relation_frequency(canonical_type, norm_conf)
                elif norm_conf == 0.0:
                    from smartmemory.relations.normalizer import _normalize_key

                    key = _normalize_key(raw_predicate)
                    if key and len(key) >= 3 and key not in _RELATION_STOPWORDS:
                        self._ontology.add_provisional_relation_type(key)
                        self._ontology.increment_relation_frequency(key, 0.0)

            enriched.append(rel)

        # Sort by plausibility descending (best triples survive max_relations limit)
        enriched.sort(key=lambda r: r.get("plausibility_score", 0.0), reverse=True)
        return enriched

    def _validate_constraints(self, entity: Dict, entity_type: str) -> List[Dict[str, Any]]:
        """Check property constraints for an entity. Returns violation dicts."""
        if not self._ontology_model:
            return []
        type_def = self._ontology_model.get_entity_type(entity_type)
        if not type_def or not type_def.property_constraints:
            return []
        violations: List[Dict[str, Any]] = []
        entity_name = self._get_entity_name(entity)
        for prop_name, constraint in type_def.property_constraints.items():
            val = entity.get(prop_name)
            if constraint.required and val is None:
                violations.append(
                    {
                        "entity_name": entity_name,
                        "property_name": prop_name,
                        "constraint_type": "required",
                        "expected": "non-empty value",
                        "actual": str(val),
                        "kind": constraint.kind,
                    }
                )
            if val is not None and constraint.type == "number":
                try:
                    float(val)
                except (ValueError, TypeError):
                    violations.append(
                        {
                            "entity_name": entity_name,
                            "property_name": prop_name,
                            "constraint_type": "type",
                            "expected": "number",
                            "actual": str(val),
                            "kind": constraint.kind,
                        }
                    )
            if val is not None and constraint.type == "enum" and constraint.enum_values:
                if str(val) not in constraint.enum_values:
                    violations.append(
                        {
                            "entity_name": entity_name,
                            "property_name": prop_name,
                            "constraint_type": "enum",
                            "expected": str(constraint.enum_values),
                            "actual": str(val),
                            "kind": constraint.kind,
                        }
                    )
            if val is not None and constraint.cardinality == "one" and isinstance(val, list):
                violations.append(
                    {
                        "entity_name": entity_name,
                        "property_name": prop_name,
                        "constraint_type": "cardinality",
                        "expected": "single value",
                        "actual": f"list of {len(val)}",
                        "kind": constraint.kind,
                    }
                )
        return violations

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(
            state,
            entities=[],
            relations=[],
            rejected=[],
            promotion_candidates=[],
            ontology_registry_id="",
            ontology_version="",
            constraint_violations=[],
            unresolved_entities=[],
        )
