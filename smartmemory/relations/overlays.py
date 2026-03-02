"""Workspace overlay loading for relation normalizer and type-pair validator.

Loads promoted relation types from the ontology graph and builds the workspace-scoped
alias/type-pair overlay dicts that RelationNormalizer and TypePairValidator accept.

Shared between SmartMemory.__init__ (Tier 1 path) and extraction_worker (Tier 2 per-job).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smartmemory.relations.normalizer import _normalize_key

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph

logger = logging.getLogger(__name__)


def load_workspace_overlays(
    ontology_graph: OntologyGraph,
) -> tuple[dict[str, str], dict[tuple[str, str], list[str]]]:
    """Load promoted relation types and build workspace overlay dicts.

    Args:
        ontology_graph: Workspace-scoped OntologyGraph instance.

    Returns:
        (workspace_aliases, workspace_type_pairs) ready to pass to
        RelationNormalizer(workspace_aliases=...) and
        TypePairValidator(workspace_type_pairs=...).
    """
    workspace_aliases: dict[str, str] = {}
    workspace_type_pairs: dict[tuple[str, str], list[str]] = {}
    try:
        promoted = ontology_graph.get_relation_types(status="confirmed")
        for rt in promoted:
            canonical = rt["name"]
            workspace_aliases[canonical] = canonical
            for alias in rt.get("aliases") or []:
                key = _normalize_key(alias)
                if key:
                    workspace_aliases[key] = canonical
            for pair in rt.get("type_pairs") or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    src, tgt = pair
                    existing = set(workspace_type_pairs.get((src, tgt), []))
                    existing.add(canonical)
                    workspace_type_pairs[(src, tgt)] = list(existing)
    except Exception:
        logger.debug("Failed to load workspace relation overlays; using seed types only", exc_info=True)

    return workspace_aliases, workspace_type_pairs
