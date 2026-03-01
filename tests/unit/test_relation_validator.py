"""Tests for smartmemory/relations/validator.py."""

from smartmemory.relations.validator import TypePairValidator


class TestTypePairValidatorPermissive:
    def setup_method(self):
        self.validator = TypePairValidator(mode="permissive")

    def test_valid_pair_correct_relation(self):
        is_valid, score = self.validator.validate("person", "works_at", "organization")
        assert is_valid is True
        assert score == 1.0

    def test_valid_pair_wrong_relation(self):
        # Person→Organization is valid, but "located_in" is not the right relation for it
        is_valid, score = self.validator.validate("person", "located_in", "organization")
        # located_in has ("*", "location"), not ("person", "organization")
        # Person→Organization IS in priors for other relations, so score=0.5
        assert is_valid is True
        assert score == 0.5

    def test_unknown_pair_permissive(self):
        is_valid, score = self.validator.validate("metric", "works_at", "skill")
        assert is_valid is True
        assert score == 0.0  # pair not in TYPE_PAIR_PRIORS for any relation

    def test_wildcard_relation(self):
        is_valid, score = self.validator.validate("person", "related_to", "technology")
        assert is_valid is True
        assert score == 0.8

    def test_case_insensitive(self):
        is_valid, score = self.validator.validate("Person", "works_at", "Organization")
        assert is_valid is True
        assert score == 1.0

    def test_related_to_always_accepted(self):
        # related_to has ("*", "*") so it's always valid with 0.8
        is_valid, score = self.validator.validate("action", "related_to", "metric")
        assert is_valid is True
        assert score == 0.8

    def test_wildcard_source_type_pair(self):
        # located_in has ("*", "location") — any source type + Location target
        is_valid, score = self.validator.validate("organization", "located_in", "location")
        assert is_valid is True
        assert score == 1.0

    def test_wildcard_target_type_pair(self):
        # created has ("person", "*") — Person source + any target
        is_valid, score = self.validator.validate("person", "created", "technology")
        assert is_valid is True
        assert score == 1.0


class TestTypePairValidatorStrict:
    def setup_method(self):
        self.validator = TypePairValidator(mode="strict")

    def test_valid_pair_correct_relation(self):
        is_valid, score = self.validator.validate("person", "works_at", "organization")
        assert is_valid is True
        assert score == 1.0

    def test_valid_pair_wrong_relation_strict(self):
        is_valid, score = self.validator.validate("person", "located_in", "organization")
        assert is_valid is False
        assert score == 0.5

    def test_unknown_pair_strict(self):
        is_valid, score = self.validator.validate("metric", "works_at", "skill")
        assert is_valid is False
        assert score == 0.0  # pair not in TYPE_PAIR_PRIORS for any relation

    def test_wildcard_relation_strict(self):
        # related_to has ("*", "*") — should still be valid even in strict mode
        is_valid, score = self.validator.validate("metric", "related_to", "skill")
        assert is_valid is True
        assert score == 0.8

    def test_invalid_mode_raises(self):
        try:
            TypePairValidator(mode="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
