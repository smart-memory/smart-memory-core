"""EntityRuler stage — rule-based entity extraction using spaCy NER."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# Module-level cache for spaCy model
_nlp_cache: dict = {"nlp": None}

# Label mapping reused from SpacyExtractor._map_spacy_label_to_type()
_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "DATE": "temporal",
    "TIME": "temporal",
    "EVENT": "event",
    "PRODUCT": "product",
    "WORK_OF_ART": "work_of_art",
    "FAC": "location",
    "NORP": "nationality",
    "LAW": "concept",
    "LANGUAGE": "language",
    "MONEY": "concept",
    "QUANTITY": "concept",
    "PERCENT": "concept",
    "ORDINAL": "concept",
    "CARDINAL": "concept",
}


def _map_label(label: str) -> str:
    """Map a spaCy NER label to our entity type system."""
    return _LABEL_MAP.get(label.upper(), "concept")


def _get_nlp(model_name: str = "en_core_web_sm"):
    """Lazy-load and cache a spaCy model."""
    if _nlp_cache["nlp"] is not None:
        return _nlp_cache["nlp"]
    try:
        import spacy

        nlp = spacy.load(model_name)
    except Exception:
        nlp = None
    _nlp_cache["nlp"] = nlp
    return nlp


def _ngram_scan(text: str, patterns: Dict[str, str]) -> List[tuple[str, str]]:
    """Scan text for n-gram matches against pattern dictionary.

    Returns list of (matched_text, entity_type) pairs.
    """
    matches: List[tuple[str, str]] = []
    words = text.split()
    seen_spans: set[str] = set()

    # Try up to 4-grams (covers most entity names)
    for n in range(4, 0, -1):
        for i in range(len(words) - n + 1):
            span = " ".join(words[i : i + n])
            span_lower = span.lower()
            if span_lower in seen_spans:
                continue
            if span_lower in patterns:
                matches.append((span, patterns[span_lower]))
                seen_spans.add(span_lower)
                # Mark component words so shorter n-grams don't re-match
                for j in range(n):
                    if i + j < len(words):
                        seen_spans.add(words[i + j].lower())

    return matches


class EntityRulerStage:
    """Extract entities using spaCy NER with rule-based patterns."""

    def __init__(self, nlp=None, pattern_manager=None, public_knowledge_store=None):
        """Args:
        nlp: Optional pre-loaded spaCy Language (for testing).
        pattern_manager: Optional PatternManager for learned pattern dictionary scan.
        public_knowledge_store: Optional PublicKnowledgeStore for Wikidata entity patterns.
        """
        self._nlp = nlp
        self._pattern_manager = pattern_manager
        self._public_knowledge_store = public_knowledge_store
        self._public_patterns: dict[str, str] = {}
        self._public_version: int = -1

    @property
    def name(self) -> str:
        return "entity_ruler"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        ruler_cfg = config.extraction.entity_ruler
        if not ruler_cfg.enabled:
            return state

        # Build input text from simplified sentences or resolved/raw text
        if state.simplified_sentences:
            text = " ".join(state.simplified_sentences)
        else:
            text = state.resolved_text or state.text

        if not text or not text.strip():
            return state

        nlp = self._nlp or _get_nlp(ruler_cfg.spacy_model)
        if nlp is None:
            logger.warning("spaCy not available, skipping entity ruler")
            return state

        try:
            doc = nlp(text)
            entities: List[Dict[str, Any]] = []
            seen: set = set()

            for ent in doc.ents:
                name = ent.text.strip()
                entity_type = _map_label(ent.label_)
                key = (name.lower(), entity_type)

                if key in seen:
                    continue
                seen.add(key)

                # spaCy NER does not produce per-entity confidence scores.
                # We assign a fixed confidence based on the model quality.
                confidence = 0.9

                if confidence < ruler_cfg.min_confidence:
                    continue

                entities.append(
                    {
                        "name": name,
                        "entity_type": entity_type,
                        "confidence": confidence,
                        "source": "entity_ruler",
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

            # Lazy version check — refresh public patterns if store version changed
            if self._public_knowledge_store:
                try:
                    current_version = self._public_knowledge_store.version
                    if current_version > self._public_version:
                        self._public_patterns = self._public_knowledge_store.get_ruler_patterns()
                        self._public_version = current_version
                except Exception as e:
                    logger.debug("Failed to refresh public knowledge patterns: %s", e)

            # Three-layer precedence: workspace > tenant (future) > public
            merged_patterns: dict[str, str] = {}
            if self._public_patterns:
                merged_patterns.update(self._public_patterns)  # base layer
            if self._pattern_manager:
                learned = self._pattern_manager.get_patterns()
                if learned:
                    for key, ws_type in learned.items():
                        if key in merged_patterns and merged_patterns[key] != ws_type:
                            logger.debug(
                                "Pattern collision '%s': public=%s, workspace=%s (workspace wins)",
                                key, merged_patterns[key], ws_type,
                            )
                    merged_patterns.update(learned)  # workspace wins

            if merged_patterns:
                for span_text, entity_type in _ngram_scan(text, merged_patterns):
                    key = (span_text.lower(), entity_type)
                    if key not in seen:
                        seen.add(key)
                        entities.append(
                            {
                                "name": span_text,
                                "entity_type": entity_type,
                                "confidence": 0.85,
                                "source": "entity_ruler_learned",
                            }
                        )

            return replace(state, ruler_entities=entities)
        except Exception as e:
            logger.warning("Entity ruler failed: %s", e)
            return state

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, ruler_entities=[])
