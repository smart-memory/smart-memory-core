"""FalkorDB-backed PatternStore — thin adapter over OntologyGraph."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph

logger = logging.getLogger(__name__)


class FalkorDBPatternStore:
    """PatternStore backed by FalkorDB EntityPattern nodes via OntologyGraph.

    Delegates all persistence to OntologyGraph methods — no new Cypher here.
    The count >= 2 quality threshold is applied inside OntologyGraph.get_entity_patterns()
    Cypher; this store does not re-filter.
    """

    def __init__(self, ontology_graph: OntologyGraph) -> None:
        self._ontology = ontology_graph

    def load(self, workspace_id: str) -> list[tuple[str, str]]:
        """Load quality-filtered patterns from FalkorDB.

        OntologyGraph.get_entity_patterns() returns List[Dict] with keys
        name/label/confidence/source and applies count >= 2 in Cypher.
        Returns (name, label) tuples for PatternManager.
        """
        try:
            raw = self._ontology.get_entity_patterns(workspace_id)
            return [(p["name"], p["label"]) for p in raw if p.get("name") and p.get("label")]
        except Exception as exc:
            logger.warning("FalkorDBPatternStore.load failed for workspace '%s': %s", workspace_id, exc)
            return []

    def save(
        self,
        name: str,
        label: str,
        confidence: float,
        count: int,
        workspace_id: str,
        source: str = "unknown",
    ) -> None:
        """Persist pattern via OntologyGraph.add_entity_pattern().

        source is forwarded to p.source in FalkorDB for provenance/stats queries.
        OntologyGraph.add_entity_pattern() preserves original source on MATCH
        (ON MATCH SET does not include p.source), consistent with JSONLPatternStore.
        """
        try:
            self._ontology.add_entity_pattern(
                name=name,
                label=label,
                confidence=confidence,
                workspace_id=workspace_id,
                source=source,
                initial_count=count,
            )
        except Exception as exc:
            logger.debug("FalkorDBPatternStore.save failed for '%s': %s", name, exc)

    def delete(self, name: str, label: str, workspace_id: str) -> None:
        """Remove pattern via OntologyGraph.delete_entity_pattern()."""
        try:
            self._ontology.delete_entity_pattern(name=name, label=label, workspace_id=workspace_id)
        except Exception as exc:
            logger.debug("FalkorDBPatternStore.delete failed for '%s': %s", name, exc)
