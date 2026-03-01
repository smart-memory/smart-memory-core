"""PublicKnowledgeGrounder — two-step entity grounding loop.

Replaces WikipediaGrounder in the grounding pipeline when a
PublicKnowledgeStore is available. Matches WikipediaGrounder.ground()
signature at wikipedia.py:37.

Step 1: Alias lookup in PublicKnowledgeStore
Step 2: SPARQL fallback (if sparql_client provided) → absorb on hit
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from smartmemory.grounding.models import GroundingDecision, PublicEntity
from smartmemory.grounding.store import PublicKnowledgeStore
from smartmemory.grounding.type_map import normalize_type, type_to_p31_qids

logger = logging.getLogger(__name__)


class PublicKnowledgeGrounder:
    """Grounds entities to canonical Wikidata QIDs via alias lookup + SPARQL fallback."""

    def __init__(self, store: PublicKnowledgeStore, sparql_client=None):
        self.store = store
        self.sparql = sparql_client
        self._graph_hits = 0  # Matches WikipediaGrounder._graph_hits for GroundStage token tracking
        self.decisions: list[GroundingDecision] = []

    def ground(self, item, entities, graph) -> list:
        """Ground entities to Wikidata QIDs.

        Signature matches WikipediaGrounder.ground(item, entities, graph).

        Returns:
            List of grounded node IDs (wikidata:{qid} format).
        """
        self._graph_hits = 0
        self.decisions = []

        if not entities:
            return []

        provenance_candidates = []

        for entity in entities:
            name = self._extract_name(entity)
            if not name:
                continue

            entity_type = self._extract_type(entity)
            item_id = self._extract_item_id(entity)

            # Step 1: Alias lookup (also checks for existing graph nodes)
            candidates = self.store.lookup_by_alias(name)

            # Check if already grounded in graph (reuse candidates to avoid double lookup)
            grounded_id = self._try_graph_hit_from_candidates(candidates, graph)
            if grounded_id:
                provenance_candidates.append(grounded_id)
                self._ensure_edge(graph, item_id, grounded_id)
                self._graph_hits += 1
                continue

            selected = self._select_candidate(candidates, entity_type)

            if selected:
                wikidata_id = self._create_wikidata_node(selected, graph)
                self._ensure_edge(graph, item_id, wikidata_id)
                provenance_candidates.append(wikidata_id)
                self._log_decision(name, len(candidates), selected, "alias_lookup",
                                   "single_candidate" if len(candidates) == 1 else "type_match")
                continue

            # Step 2: Type-filtered SPARQL fallback
            if self.sparql and self._sparql_available():
                type_filter_qids = type_to_p31_qids(entity_type) if entity_type else None
                sparql_results = self.sparql.query_entity(name, type_filter_qids=type_filter_qids)
                if sparql_results:
                    selected = self._select_candidate(sparql_results, entity_type)
                    if selected:
                        self.store.absorb(selected)
                        wikidata_id = self._create_wikidata_node(selected, graph)
                        self._ensure_edge(graph, item_id, wikidata_id)
                        provenance_candidates.append(wikidata_id)
                        self._log_decision(name, len(sparql_results), selected,
                                           "sparql_fallback", "type_match")
                        continue

            # Ungrounded
            self._log_decision(name, len(candidates), None, "ungrounded", "ambiguous")

        return provenance_candidates

    def _extract_name(self, entity) -> str | None:
        if isinstance(entity, dict):
            return entity.get("name") or (entity.get("metadata") or {}).get("name")
        elif hasattr(entity, "metadata") and entity.metadata:
            return entity.metadata.get("name")
        return None

    def _extract_type(self, entity) -> str:
        if isinstance(entity, dict):
            return entity.get("entity_type", "")
        return ""

    def _extract_item_id(self, entity) -> str | None:
        if isinstance(entity, dict):
            return entity.get("item_id")
        elif hasattr(entity, "item_id"):
            return entity.item_id
        return None

    def _try_graph_hit_from_candidates(
        self, candidates: list[PublicEntity], graph
    ) -> str | None:
        """Check if a wikidata node already exists for any candidate."""
        for candidate in candidates:
            wikidata_id = f"wikidata:{candidate.qid}"
            existing = graph.get_node(wikidata_id)
            if existing:
                return wikidata_id
        return None

    def _select_candidate(
        self, candidates: list[PublicEntity], entity_type: str
    ) -> PublicEntity | None:
        """Select the best candidate from a list.

        Strategy (from design Section 4):
        1. Single candidate + confidence > 0.8 → ground immediately
        2. Multiple → type compatibility → domain → confidence
        """
        if not candidates:
            return None

        if len(candidates) == 1 and candidates[0].confidence > 0.8:
            return candidates[0]

        if not entity_type:
            # No type context — pick highest confidence, tie-break by QID
            return sorted(candidates, key=lambda c: (-c.confidence, c.qid))[0]

        # Filter by type compatibility
        normalized_type = normalize_type(entity_type)
        type_matches = [
            c for c in candidates
            if normalize_type(c.entity_type) == normalized_type
        ]

        if type_matches:
            return sorted(type_matches, key=lambda c: (-c.confidence, c.qid))[0]

        # No type match — fall back to confidence ranking
        return sorted(candidates, key=lambda c: (-c.confidence, c.qid))[0]

    def _sparql_available(self) -> bool:
        """Check if SPARQL client's circuit breaker allows requests."""
        if not self.sparql:
            return False
        return self.sparql._circuit.allow_request()

    def _create_wikidata_node(self, entity: PublicEntity, graph) -> str:
        """Create a wikidata:{qid} node in the graph."""
        wikidata_id = f"wikidata:{entity.qid}"
        node_properties = {
            "entity": entity.label,
            "qid": entity.qid,
            "description": entity.description[:300] if entity.description else "",
            "entity_type": entity.entity_type,
            "type": "wikidata_entity",
            "node_category": "grounding",
        }
        graph.add_node(item_id=wikidata_id, properties=node_properties, is_global=True)
        logger.info("Created Wikidata node: %s (%s)", wikidata_id, entity.label)
        return wikidata_id

    @staticmethod
    def _ensure_edge(graph, entity_item_id: str | None, wikidata_id: str) -> None:
        """Create GROUNDED_IN edge from entity to Wikidata node."""
        if entity_item_id:
            graph.add_edge(
                entity_item_id, wikidata_id,
                edge_type="GROUNDED_IN", properties={}, is_global=True,
            )
        else:
            logger.warning(
                "No entity item_id for Wikidata node %s, skipping edge", wikidata_id
            )

    def _log_decision(
        self,
        surface_form: str,
        candidates_found: int,
        selected: PublicEntity | None,
        source: str,
        reason: str,
    ) -> None:
        decision = GroundingDecision(
            surface_form=surface_form,
            candidates_found=candidates_found,
            selected_qid=selected.qid if selected else None,
            selected_label=selected.label if selected else None,
            confidence=selected.confidence if selected else 0.0,
            source=source,
            disambiguation_reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.decisions.append(decision)
        logger.debug("Grounding decision: %s → %s (%s)", surface_form, source, reason)
