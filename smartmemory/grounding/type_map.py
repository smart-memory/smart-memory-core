"""Wikidata P31 (instance-of) QID → SmartMemory SEED_TYPES mapping.

This is the canonical bridge between Wikidata's type system and SmartMemory's
ontology. Values MUST be from SEED_TYPES in ontology_graph.py (title case).

Use normalize_type() for all type comparisons — EntityRuler emits lowercase
types while SEED_TYPES uses title case.
"""

# Maps Wikidata P31 QIDs to SmartMemory entity types.
# Values must be valid SEED_TYPES entries (title case).
WIKIDATA_TYPE_MAP: dict[str, str] = {
    # Technology
    "Q9143": "Technology",      # programming language
    "Q271680": "Technology",    # software framework
    "Q188860": "Technology",    # software library
    "Q9135": "Technology",      # operating system
    "Q235557": "Technology",    # file format
    "Q7397": "Technology",      # software
    "Q21127166": "Technology",  # Java library
    # Organization
    "Q4830453": "Organization",  # business enterprise
    "Q3918": "Organization",     # university
    "Q7278": "Organization",     # political party
    "Q163740": "Organization",   # non-profit organization
    # Person
    "Q5": "Person",  # human
    # Concept
    "Q336": "Concept",    # science
    "Q11862829": "Concept",  # academic discipline
    "Q1047113": "Concept",  # specialty (medical, etc.)
    # Event
    "Q1656682": "Event",  # event
    # Location
    "Q515": "Location",  # city
    "Q6256": "Location",  # country
    "Q3624078": "Location",  # sovereign state
}


def normalize_type(t: str) -> str:
    """Normalize entity type for case-insensitive comparison.

    EntityRuler emits lowercase ("person"), SEED_TYPES uses title case ("Person").
    This function is the single canonical comparison point.
    """
    return t.lower()


def type_to_p31_qids(entity_type: str) -> list[str]:
    """Reverse lookup: SmartMemory entity type → list of Wikidata P31 QIDs.

    Used by the grounding loop to build type-filtered SPARQL queries.
    Multiple P31 QIDs may map to the same SmartMemory type (e.g., "Technology"
    covers programming languages, frameworks, libraries, etc.).

    Returns empty list if the type has no known P31 mapping.
    """
    normalized = normalize_type(entity_type)
    return [qid for qid, stype in WIKIDATA_TYPE_MAP.items() if normalize_type(stype) == normalized]
