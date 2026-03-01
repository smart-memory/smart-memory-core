"""Canonical relation type vocabulary for SmartMemory knowledge graphs.

Defines ~39 canonical relation types organized by semantic category,
with alias mappings and type-pair constraints aligned to SEED_TYPES.

This module is the build-time source of truth for relation normalization.
At runtime, the OntologyGraph seeds these as RelationType nodes (mirroring
how SEED_TYPES seeds EntityType nodes).

Pattern: smartmemory/grounding/type_map.py — module-level constants with
derived lookups computed at import time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RelationTypeDef:
    """Definition of a canonical relation type.

    Attributes:
        canonical_name: Lowercase snake_case canonical name (e.g. "works_at").
        category: Semantic category (e.g. "professional", "spatial").
        aliases: Tuple of alternative names that map to this canonical type.
        type_pairs: Tuple of (source_type, target_type) pairs. Use ("*", "*") for wildcard.
        bidirectional: True if the relation is symmetric (e.g. "collaborates_with").
        description: Human-readable description.
    """

    canonical_name: str
    category: str
    aliases: tuple[str, ...]
    type_pairs: tuple[tuple[str, str], ...]
    bidirectional: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Canonical relation types — 39 types, 8 categories
# ---------------------------------------------------------------------------

CANONICAL_RELATION_TYPES: dict[str, RelationTypeDef] = {}


def _register(*defs: RelationTypeDef) -> None:
    for d in defs:
        CANONICAL_RELATION_TYPES[d.canonical_name] = d


# --- Professional/Organizational (8) ---
_register(
    RelationTypeDef(
        canonical_name="works_at",
        category="professional",
        aliases=("employed_by", "employee_of", "works_for", "staff_at"),
        type_pairs=(("person", "organization"),),
        description="Employment relationship",
    ),
    RelationTypeDef(
        canonical_name="founded",
        category="professional",
        aliases=("co_founded", "established", "started"),
        type_pairs=(("person", "organization"),),
        description="Founding relationship",
    ),
    RelationTypeDef(
        canonical_name="manages",
        category="professional",
        aliases=("leads", "oversees", "heads", "directs"),
        type_pairs=(("person", "person"), ("person", "organization")),
        description="Management/leadership relationship",
    ),
    RelationTypeDef(
        canonical_name="reports_to",
        category="professional",
        aliases=("supervised_by", "under"),
        type_pairs=(("person", "person"),),
        description="Reporting hierarchy",
    ),
    RelationTypeDef(
        canonical_name="member_of",
        category="professional",
        aliases=("belongs_to", "affiliated_with"),
        type_pairs=(("person", "organization"),),
        description="Membership/affiliation",
    ),
    RelationTypeDef(
        canonical_name="collaborates_with",
        category="professional",
        aliases=("partners_with", "works_with"),
        type_pairs=(("person", "person"), ("organization", "organization")),
        bidirectional=True,
        description="Collaboration relationship",
    ),
    RelationTypeDef(
        canonical_name="ceo_of",
        category="professional",
        aliases=("president_of", "chairman_of", "chief_of"),
        type_pairs=(("person", "organization"),),
        description="Executive leadership role",
    ),
    RelationTypeDef(
        canonical_name="acquired",
        category="professional",
        aliases=("bought", "merged_with", "took_over"),
        type_pairs=(("organization", "organization"),),
        description="Corporate acquisition/merger",
    ),
)

# --- Spatial (4) ---
_register(
    RelationTypeDef(
        canonical_name="located_in",
        category="spatial",
        aliases=("based_in", "headquartered_in", "situated_in"),
        type_pairs=(("*", "location"),),
        description="Location relationship",
    ),
    RelationTypeDef(
        canonical_name="contains",
        category="spatial",
        aliases=("includes", "encompasses", "has"),
        type_pairs=(("location", "location"), ("organization", "organization")),
        description="Containment relationship",
    ),
    RelationTypeDef(
        canonical_name="lives_in",
        category="spatial",
        aliases=("resides_in", "lives_at", "home_in"),
        type_pairs=(("person", "location"),),
        description="Residential location",
    ),
    RelationTypeDef(
        canonical_name="near",
        category="spatial",
        aliases=("adjacent_to", "close_to", "next_to"),
        type_pairs=(("location", "location"),),
        bidirectional=True,
        description="Proximity relationship",
    ),
)

# --- Creation/Production (5) ---
_register(
    RelationTypeDef(
        canonical_name="created",
        category="creation",
        aliases=("built", "developed", "designed", "invented", "authored", "wrote"),
        type_pairs=(("person", "*"), ("organization", "*")),
        description="Creation/authorship",
    ),
    RelationTypeDef(
        canonical_name="released",
        category="creation",
        aliases=("published", "launched", "shipped", "deployed"),
        type_pairs=(("organization", "technology"), ("organization", "document")),
        description="Publication/release",
    ),
    RelationTypeDef(
        canonical_name="produces",
        category="creation",
        aliases=("manufactures", "generates", "outputs"),
        type_pairs=(("organization", "*"), ("process", "*")),
        description="Production relationship",
    ),
    RelationTypeDef(
        canonical_name="derived_from",
        category="creation",
        aliases=("based_on", "inspired_by", "adapted_from", "forked_from"),
        type_pairs=(("*", "*"),),
        description="Derivation/inspiration",
    ),
    RelationTypeDef(
        canonical_name="contributed_to",
        category="creation",
        aliases=("participated_in", "involved_in"),
        type_pairs=(("person", "*"), ("organization", "*")),
        description="Contribution/participation",
    ),
)

# --- Usage/Application (4) ---
_register(
    RelationTypeDef(
        canonical_name="uses",
        category="usage",
        aliases=("utilizes", "employs", "leverages", "runs_on"),
        type_pairs=(("*", "technology"), ("*", "tool")),
        description="Usage of tool/technology",
    ),
    RelationTypeDef(
        canonical_name="framework_for",
        category="usage",
        aliases=("library_for", "plugin_for", "extension_of", "built_on"),
        type_pairs=(("technology", "technology"),),
        description="Framework/library relationship",
    ),
    RelationTypeDef(
        canonical_name="implements",
        category="usage",
        aliases=("realizes", "conforms_to", "follows"),
        type_pairs=(("technology", "concept"),),
        description="Implementation of concept/standard",
    ),
    RelationTypeDef(
        canonical_name="supports",
        category="usage",
        aliases=("compatible_with", "integrates_with"),
        type_pairs=(("technology", "technology"), ("tool", "*")),
        description="Compatibility/integration",
    ),
)

# --- Hierarchical/Compositional (4) ---
_register(
    RelationTypeDef(
        canonical_name="part_of",
        category="hierarchical",
        aliases=("component_of", "subset_of", "included_in"),
        type_pairs=(("*", "*"),),
        description="Part-whole relationship",
    ),
    RelationTypeDef(
        canonical_name="type_of",
        category="hierarchical",
        aliases=("subclass_of", "kind_of", "instance_of", "category_of"),
        type_pairs=(("*", "*"),),
        description="Type/classification hierarchy",
    ),
    RelationTypeDef(
        canonical_name="parent_of",
        category="hierarchical",
        aliases=("superclass_of",),
        type_pairs=(("person", "person"), ("concept", "concept")),
        description="Parent/superclass relationship",
    ),
    RelationTypeDef(
        canonical_name="specializes",
        category="hierarchical",
        aliases=("refines", "extends", "narrows"),
        type_pairs=(("concept", "concept"), ("technology", "technology")),
        description="Specialization/refinement",
    ),
)

# --- Temporal/Causal (5) ---
_register(
    RelationTypeDef(
        canonical_name="occurred_in",
        category="temporal",
        aliases=("happened_in", "took_place_in"),
        type_pairs=(("event", "location"), ("event", "concept")),
        description="Event location/context",
    ),
    RelationTypeDef(
        canonical_name="started_in",
        category="temporal",
        aliases=("began_in", "launched_in", "founded_in"),
        type_pairs=(("*", "concept"),),
        description="Temporal start",
    ),
    RelationTypeDef(
        canonical_name="causes",
        category="temporal",
        aliases=("leads_to", "results_in", "triggers"),
        type_pairs=(("*", "*"),),
        description="Causal relationship",
    ),
    RelationTypeDef(
        canonical_name="enables",
        category="temporal",
        aliases=("allows", "permits", "makes_possible"),
        type_pairs=(("*", "*"),),
        description="Enablement relationship",
    ),
    RelationTypeDef(
        canonical_name="requires",
        category="temporal",
        aliases=("depends_on", "needs", "prerequisite_for"),
        type_pairs=(("*", "*"),),
        description="Dependency/requirement",
    ),
)

# --- Knowledge/Information (5) ---
_register(
    RelationTypeDef(
        canonical_name="knows_about",
        category="knowledge",
        aliases=("expert_in", "specializes_in", "skilled_in"),
        type_pairs=(("person", "*"),),
        description="Knowledge/expertise",
    ),
    RelationTypeDef(
        canonical_name="teaches",
        category="knowledge",
        aliases=("instructs", "trains"),
        type_pairs=(("person", "concept"), ("person", "skill")),
        description="Teaching/instruction",
    ),
    RelationTypeDef(
        canonical_name="references",
        category="knowledge",
        aliases=("cites", "mentions", "links_to"),
        type_pairs=(("document", "*"), ("*", "document")),
        description="Citation/reference",
    ),
    RelationTypeDef(
        canonical_name="describes",
        category="knowledge",
        aliases=("documents", "covers", "explains"),
        type_pairs=(("document", "*"),),
        description="Documentation/description",
    ),
    RelationTypeDef(
        canonical_name="awarded",
        category="knowledge",
        aliases=("won", "received", "nominated_for"),
        type_pairs=(("person", "*"), ("organization", "*")),
        description="Award/recognition",
    ),
)

# --- Logical/Semantic (4) ---
_register(
    RelationTypeDef(
        canonical_name="contradicts",
        category="logical",
        aliases=("conflicts_with", "opposes", "negates"),
        type_pairs=(("*", "*"),),
        bidirectional=True,
        description="Contradiction/conflict",
    ),
    RelationTypeDef(
        canonical_name="similar_to",
        category="logical",
        aliases=("resembles", "analogous_to", "like"),
        type_pairs=(("*", "*"),),
        bidirectional=True,
        description="Similarity/analogy",
    ),
    RelationTypeDef(
        canonical_name="related_to",
        category="logical",
        aliases=("associated_with", "connected_to"),
        type_pairs=(("*", "*"),),
        bidirectional=True,
        description="Catch-all fallback relation",
    ),
    RelationTypeDef(
        canonical_name="supersedes",
        category="logical",
        aliases=("replaces", "deprecates", "succeeds"),
        type_pairs=(("*", "*"),),
        description="Replacement/succession",
    ),
)


# ---------------------------------------------------------------------------
# Derived lookup tables — computed once at import time
# ---------------------------------------------------------------------------


def _build_alias_index() -> dict[str, str]:
    """Map every alias (and canonical name) to its canonical type. Lowercase snake_case keys."""
    index: dict[str, str] = {}
    for typedef in CANONICAL_RELATION_TYPES.values():
        index[typedef.canonical_name] = typedef.canonical_name
        for alias in typedef.aliases:
            index[alias] = typedef.canonical_name
    return index


def _build_type_pair_priors() -> dict[tuple[str, str], list[str]]:
    """Map (source_type, target_type) → list of valid canonical relation names.

    Keys are lowercase to match normalize_type() from type_map.py.
    Wildcard pairs ("*", "*") are expanded: they add the relation to ALL
    concrete pairs already in the table, but are also stored under ("*", "*").
    """
    priors: dict[tuple[str, str], list[str]] = defaultdict(list)

    # First pass: collect concrete pairs
    for typedef in CANONICAL_RELATION_TYPES.values():
        for src, tgt in typedef.type_pairs:
            key = (src.lower(), tgt.lower())
            if typedef.canonical_name not in priors[key]:
                priors[key].append(typedef.canonical_name)

    return dict(priors)


ALIAS_INDEX: dict[str, str] = _build_alias_index()
TYPE_PAIR_PRIORS: dict[tuple[str, str], list[str]] = _build_type_pair_priors()
