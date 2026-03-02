"""Type-pair validator for relation triples.

Validates (entity_type, canonical_relation, entity_type) triples against
the TYPE_PAIR_PRIORS table from the relation schema.

Score ladder (design.md line 318):
1. Exact pair match in TYPE_PAIR_PRIORS for this relation → (True, 1.0)
2. Relation has wildcard ("*", "*") type-pairs → (True, 0.8)
3. Pair in TYPE_PAIR_PRIORS for different relation → (True/False, 0.5)
4. Pair not in table → (True/False, 0.0)
"""

from __future__ import annotations

from smartmemory.relations.schema import CANONICAL_RELATION_TYPES, TYPE_PAIR_PRIORS


class TypePairValidator:
    """Validate entity type-pairs against the canonical relation schema."""

    def __init__(
        self,
        mode: str = "permissive",
        workspace_type_pairs: dict[tuple[str, str], list[str]] | None = None,
    ):
        """
        Args:
            mode: "strict" (reject unknown pairs) or "permissive" (warn + allow)
            workspace_type_pairs: Optional workspace-scoped type-pair overrides.
                Merged on top of global TYPE_PAIR_PRIORS with set-dedup.
        """
        if mode not in ("strict", "permissive"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'strict' or 'permissive'.")
        self._mode = mode
        self._type_pair_priors = dict(TYPE_PAIR_PRIORS)  # instance copy — never mutate module-level
        if workspace_type_pairs:
            for key, canonicals in workspace_type_pairs.items():
                existing = set(self._type_pair_priors.get(key, []))
                existing.update(canonicals)
                self._type_pair_priors[key] = list(existing)

    def validate(
        self,
        source_type: str,
        canonical_type: str,
        target_type: str,
    ) -> tuple[bool, float]:
        """Validate a triple's type-pair.

        Returns:
            (is_valid, type_pair_score)
            - is_valid: True if the triple passes validation
            - type_pair_score: 1.0 (exact), 0.8 (wildcard), 0.5 (valid pair wrong relation), 0.0 (unknown)
        """
        src = source_type.lower()
        tgt = target_type.lower()

        # Step 1: Check if this exact (src, tgt) pair is valid for this canonical_type
        pair_key = (src, tgt)
        valid_types = self._type_pair_priors.get(pair_key, [])
        if canonical_type in valid_types:
            return (True, 1.0)

        # Also check wildcard source: ("*", tgt)
        wildcard_src = self._type_pair_priors.get(("*", tgt), [])
        if canonical_type in wildcard_src:
            return (True, 1.0)

        # And wildcard target: (src, "*")
        wildcard_tgt = self._type_pair_priors.get((src, "*"), [])
        if canonical_type in wildcard_tgt:
            return (True, 1.0)

        # Step 2: Check if the relation has wildcard ("*", "*") type-pairs
        typedef = CANONICAL_RELATION_TYPES.get(canonical_type)
        if typedef:
            for s, t in typedef.type_pairs:
                if s == "*" and t == "*":
                    return (True, 0.8)

        # Step 3: Check if the specific pair exists for a DIFFERENT relation
        # Only match concrete (src, tgt) entries — NOT the global ("*", "*") bucket,
        # which is populated by wildcard relations and already handled by Step 2.
        if valid_types or wildcard_src or wildcard_tgt:
            # The pair is known, just not for this relation
            return (self._mode == "permissive", 0.5)

        # Step 4: Pair not in table at all
        return (self._mode == "permissive", 0.0)
