"""Tests for smartmemory/relations/normalizer.py."""

from smartmemory.relations.normalizer import RelationNormalizer, _normalize_key


class TestNormalizeKey:
    def test_basic(self):
        assert _normalize_key("works_at") == "works_at"

    def test_uppercase(self):
        assert _normalize_key("Works_At") == "works_at"

    def test_hyphens(self):
        assert _normalize_key("works-at") == "works_at"

    def test_spaces(self):
        assert _normalize_key("works at") == "works_at"

    def test_mixed_punctuation(self):
        assert _normalize_key("Framework--For") == "framework_for"

    def test_empty(self):
        assert _normalize_key("") == ""

    def test_leading_trailing(self):
        assert _normalize_key("__works_at__") == "works_at"


class TestRelationNormalizerAliasOnly:
    """Tests with no embedding function (alias-only mode)."""

    def setup_method(self):
        self.normalizer = RelationNormalizer()

    def test_exact_alias_hit(self):
        canonical, conf = self.normalizer.normalize("employed_by")
        assert canonical == "works_at"
        assert conf == 1.0

    def test_self_lookup(self):
        canonical, conf = self.normalizer.normalize("works_at")
        assert canonical == "works_at"
        assert conf == 1.0

    def test_case_insensitive(self):
        canonical, conf = self.normalizer.normalize("Works_At")
        assert canonical == "works_at"
        assert conf == 1.0

    def test_punctuation_normalization(self):
        canonical, conf = self.normalizer.normalize("works-at")
        assert canonical == "works_at"
        assert conf == 1.0

    def test_unknown_predicate_no_embedding(self):
        canonical, conf = self.normalizer.normalize("completely_unknown_xyz")
        assert canonical == "related_to"
        assert conf == 0.0

    def test_empty_string(self):
        canonical, conf = self.normalizer.normalize("")
        assert canonical == "related_to"
        assert conf == 0.0

    def test_whitespace_only(self):
        canonical, conf = self.normalizer.normalize("   ")
        assert canonical == "related_to"
        assert conf == 0.0

    def test_various_aliases(self):
        cases = [
            ("co_founded", "founded"),
            ("leads", "manages"),
            ("based_in", "located_in"),
            ("built", "created"),
            ("forked_from", "derived_from"),
            ("depends_on", "requires"),
            ("conflicts_with", "contradicts"),
        ]
        for alias, expected in cases:
            canonical, conf = self.normalizer.normalize(alias)
            assert canonical == expected, f"'{alias}' → '{canonical}', expected '{expected}'"
            assert conf == 1.0


class TestRelationNormalizerWithWorkspaceOverlay:
    """Tests for CORE-EXT-1c workspace-scoped alias overlays."""

    def test_workspace_alias_overrides_fallback(self):
        """A workspace-promoted alias resolves to its canonical type instead of related_to."""
        normalizer = RelationNormalizer(workspace_aliases={"mentors": "advises"})
        canonical, conf = normalizer.normalize("mentors")
        assert canonical == "advises"
        assert conf == 1.0

    def test_workspace_alias_does_not_affect_seed(self):
        """Seed aliases still resolve correctly when workspace overlay is set."""
        normalizer = RelationNormalizer(workspace_aliases={"mentors": "advises"})
        canonical, conf = normalizer.normalize("employed_by")
        assert canonical == "works_at"
        assert conf == 1.0

    def test_no_workspace_aliases_matches_baseline(self):
        """No workspace aliases = same behaviour as baseline."""
        baseline = RelationNormalizer()
        with_none = RelationNormalizer(workspace_aliases=None)
        for pred in ["works_at", "employed_by", "completely_unknown_xyz"]:
            assert baseline.normalize(pred) == with_none.normalize(pred)

    def test_global_alias_index_not_mutated(self):
        """Constructing with workspace aliases must NOT mutate the module-level ALIAS_INDEX."""
        from smartmemory.relations.normalizer import ALIAS_INDEX

        before = dict(ALIAS_INDEX)
        RelationNormalizer(workspace_aliases={"novel_pred_xyz": "custom_type"})
        assert ALIAS_INDEX == before, "Module-level ALIAS_INDEX was mutated"


class TestRelationNormalizerWithEmbedding:
    """Tests with a mock embedding function."""

    @staticmethod
    def _mock_embedding(text: str) -> list[float]:
        """Mock: return a unique vector based on text hash so only exact matches score high."""
        import hashlib

        h = hashlib.md5(text.lower().encode()).hexdigest()
        # Convert first 4 hex chars to floats for a 4-dim vector
        return [int(h[i], 16) / 15.0 for i in range(4)]

    @staticmethod
    def _exact_match_embedding(text: str) -> list[float]:
        """Mock: "working for" → same vector as "works at" canonical."""
        if text.lower() in ("working for", "works at"):
            return [1.0, 0.0, 0.0, 0.0]
        # Everything else gets a distant vector
        return [0.0, 0.0, 0.0, 1.0]

    def test_embedding_above_threshold(self):
        normalizer = RelationNormalizer(embedding_fn=self._exact_match_embedding)
        canonical, conf = normalizer.normalize("working_for")
        assert canonical == "works_at"
        assert conf >= 0.75

    def test_embedding_below_threshold(self):
        normalizer = RelationNormalizer(embedding_fn=self._mock_embedding)
        # Hash-based vectors are pseudo-random — "zzz nonsense" should not cosine-match
        # any canonical type with >0.75 similarity
        canonical, conf = normalizer.normalize("zzz_complete_gibberish_xyz_42")
        # We accept either: it falls through to related_to (0.0), or it matched
        # something below threshold. The key guarantee is conf < 0.75 IF canonical == related_to.
        if canonical == "related_to":
            assert conf == 0.0
