"""Relation schema, normalization, and validation for SmartMemory knowledge graphs."""

from smartmemory.relations.schema import (
    ALIAS_INDEX,
    CANONICAL_RELATION_TYPES,
    TYPE_PAIR_PRIORS,
    RelationTypeDef,
)
from smartmemory.relations.normalizer import RelationNormalizer
from smartmemory.relations.validator import TypePairValidator
from smartmemory.relations.scorer import PlausibilityScorer

__all__ = [
    "ALIAS_INDEX",
    "CANONICAL_RELATION_TYPES",
    "TYPE_PAIR_PRIORS",
    "RelationTypeDef",
    "RelationNormalizer",
    "TypePairValidator",
    "PlausibilityScorer",
]
