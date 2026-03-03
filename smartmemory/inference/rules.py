"""Built-in inference rules for graph enrichment.

Each rule defines an edge type to create and the logic to find matches.
Backend-agnostic matchers use the GraphAlgos protocol and SmartGraphBackend
ABC methods — no raw Cypher.  The ``pattern_cypher`` field is kept for
backward compatibility with custom rules that run on FalkorDB directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Type alias for match functions.  Each receives a backend (SmartGraphBackend)
# and returns rows compatible with InferenceEngine._apply_rule():
#   [(source_id, target_id, conf1_or_None, conf2_or_None), ...]
MatchFn = Callable[[Any], List[Tuple[str, str, Any, Any]]]


@dataclass
class InferenceRule:
    """Definition of a single inference rule."""

    name: str = ""
    description: str = ""
    pattern_cypher: str = ""
    edge_type: str = ""
    confidence: float = 0.7
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "edge_type": self.edge_type,
            "confidence": self.confidence,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRule":
        known_keys = {"name", "description", "pattern_cypher", "edge_type", "confidence", "enabled"}
        return cls(**{k: data[k] for k in known_keys if k in data})


# ── Backend-agnostic match functions ──────────────────────────────────────


def _match_causal_transitivity(backend: Any) -> List[Tuple[str, str, Any, Any]]:
    """A-[CAUSES]->B-[CAUSES]->C where no direct A-[CAUSES]->C exists.

    Uses ``algos.pattern_match_2hop`` to find 2-hop CAUSES chains, then
    filters out pairs that already have a direct CAUSES edge.  Collects
    confidence values from the intermediate edges for confidence decay.
    """
    # Build a fast lookup of existing CAUSES edges and their confidence values.
    all_edges = backend.get_all_edges()
    causes_conf: Dict[Tuple[str, str], Any] = {}
    for e in all_edges:
        if e.get("edge_type") == "CAUSES":
            src, tgt = e["source_id"], e["target_id"]
            # Confidence can be a top-level key or nested in properties.
            props = e.get("properties", {})
            if isinstance(props, str):
                import json

                props = json.loads(props)
            conf = props.get("confidence") if props.get("confidence") is not None else e.get("confidence")
            causes_conf[(src, tgt)] = conf

    triples = backend.algos.pattern_match_2hop("CAUSES", "CAUSES")

    results: List[Tuple[str, str, Any, Any]] = []
    for a, b, c in triples:
        if (a, c) in causes_conf:
            continue  # Direct edge already exists
        conf1 = causes_conf.get((a, b))
        conf2 = causes_conf.get((b, c))
        results.append((a, c, conf1, conf2))
    return results


def _match_contradiction_symmetry(backend: Any) -> List[Tuple[str, str, Any, Any]]:
    """A-[CONTRADICTS]->B where B-[CONTRADICTS]->A is missing.

    Finds asymmetric contradiction edges so the engine can create the
    reverse direction.
    """
    all_edges = backend.get_all_edges()
    contradicts_pairs: set[Tuple[str, str]] = set()
    for e in all_edges:
        if e.get("edge_type") == "CONTRADICTS":
            contradicts_pairs.add((e["source_id"], e["target_id"]))

    results: List[Tuple[str, str, Any, Any]] = []
    for src, tgt in contradicts_pairs:
        if (tgt, src) not in contradicts_pairs:
            results.append((tgt, src, None, None))
    return results


def _match_topic_inheritance(backend: Any) -> List[Tuple[str, str, Any, Any]]:
    """decision-[DERIVED_FROM]->semantic where semantic-[INFLUENCES]->decision is missing.

    Scans DERIVED_FROM edges, filters by node memory_type, and checks for
    the existence of the reverse INFLUENCES edge.
    """
    # Build node-type lookup once.
    all_nodes = backend.get_all_nodes()
    node_types: Dict[str, str | None] = {n.get("item_id", ""): n.get("memory_type") for n in all_nodes}

    all_edges = backend.get_all_edges()
    influences_pairs: set[Tuple[str, str]] = set()
    derived_from_edges: List[Tuple[str, str]] = []
    for e in all_edges:
        etype = e.get("edge_type")
        if etype == "INFLUENCES":
            influences_pairs.add((e["source_id"], e["target_id"]))
        elif etype == "DERIVED_FROM":
            derived_from_edges.append((e["source_id"], e["target_id"]))

    results: List[Tuple[str, str, Any, Any]] = []
    for d_id, s_id in derived_from_edges:
        if node_types.get(d_id) != "decision" or node_types.get(s_id) != "semantic":
            continue
        if (s_id, d_id) in influences_pairs:
            continue
        results.append((s_id, d_id, None, None))
    return results


# Mapping from rule name → backend-agnostic match function.
# InferenceEngine checks this before falling back to pattern_cypher.
RULE_MATCHERS: Dict[str, MatchFn] = {
    "causal_transitivity": _match_causal_transitivity,
    "contradiction_symmetry": _match_contradiction_symmetry,
    "topic_inheritance": _match_topic_inheritance,
}


def get_default_rules() -> list[InferenceRule]:
    """Return built-in inference rules.

    Three core rules:
    1. Causal transitivity - A CAUSES B and B CAUSES C implies A CAUSES C.
    2. Contradiction symmetry - A CONTRADICTS B implies B CONTRADICTS A.
    3. Topic inheritance - decision DERIVED_FROM semantic creates INFLUENCES.

    All three have backend-agnostic matchers registered in RULE_MATCHERS.
    The ``pattern_cypher`` field is kept as documentation and as a Cypher
    fallback for environments where GraphAlgos is unavailable.
    """
    return [
        InferenceRule(
            name="causal_transitivity",
            description="If A CAUSES B and B CAUSES C, infer A CAUSES C with decayed confidence",
            pattern_cypher=(
                "MATCH (a)-[r1:CAUSES]->(b)-[r2:CAUSES]->(c) "
                "WHERE a.item_id <> c.item_id "
                "AND NOT (a)-[:CAUSES]->(c) "
                "RETURN a.item_id AS source_id, c.item_id AS target_id, "
                "r1.confidence AS conf1, r2.confidence AS conf2"
            ),
            edge_type="CAUSES",
            confidence=0.7,
        ),
        InferenceRule(
            name="contradiction_symmetry",
            description="If A CONTRADICTS B, ensure B CONTRADICTS A",
            pattern_cypher=(
                "MATCH (a)-[:CONTRADICTS]->(b) "
                "WHERE NOT (b)-[:CONTRADICTS]->(a) "
                "RETURN b.item_id AS source_id, a.item_id AS target_id"
            ),
            edge_type="CONTRADICTS",
            confidence=1.0,
        ),
        InferenceRule(
            name="topic_inheritance",
            description="If decision DERIVED_FROM semantic, create INFLUENCES edge",
            pattern_cypher=(
                "MATCH (d {memory_type: 'decision'})-[:DERIVED_FROM]->"
                "(s {memory_type: 'semantic'}) "
                "WHERE NOT (s)-[:INFLUENCES]->(d) "
                "RETURN s.item_id AS source_id, d.item_id AS target_id"
            ),
            edge_type="INFLUENCES",
            confidence=0.6,
        ),
    ]
