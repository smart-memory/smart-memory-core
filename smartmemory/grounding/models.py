"""Public knowledge entity models for Wikidata grounding.

These dataclasses represent canonical entities from Wikidata and the
decisions made during the grounding process.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PublicEntity:
    """A canonical entity from the public knowledge store (Wikidata).

    Attributes:
        qid: Wikidata QID (e.g. "Q28865" for Python).
        label: Primary display label.
        aliases: Alternative surface forms for this entity.
        description: Short description from Wikidata.
        entity_type: SmartMemory type from WIKIDATA_TYPE_MAP (e.g. "Technology").
        instance_of: Wikidata P31 QIDs this entity is an instance of.
        domain: Knowledge domain (e.g. "software").
        confidence: Confidence score for this entity record (0.0-1.0).
    """

    qid: str
    label: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    entity_type: str = ""
    instance_of: list[str] = field(default_factory=list)
    domain: str = ""
    confidence: float = 1.0


@dataclass
class GroundingDecision:
    """Records the outcome of a single entity grounding attempt.

    Every entity processed by the grounding loop gets a GroundingDecision,
    whether it was successfully grounded or not.

    Attributes:
        surface_form: The entity mention as it appeared in text.
        candidates_found: Number of candidates returned by alias lookup.
        selected_qid: QID of the chosen entity, or None if ungrounded.
        selected_label: Label of the chosen entity, or None if ungrounded.
        confidence: Confidence of the grounding decision.
        source: How the entity was grounded ("alias_lookup", "sparql_fallback", "ungrounded").
        disambiguation_reason: Why this candidate was chosen
            ("single_candidate", "type_match", "domain_match", "ambiguous").
        timestamp: ISO 8601 timestamp of the decision.
    """

    surface_form: str
    candidates_found: int
    selected_qid: str | None
    selected_label: str | None
    confidence: float
    source: str
    disambiguation_reason: str
    timestamp: str
