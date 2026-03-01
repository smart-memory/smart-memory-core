"""Tests for smartmemory/relations/schema.py — vocabulary integrity."""

import re

from smartmemory.relations.schema import (
    ALIAS_INDEX,
    CANONICAL_RELATION_TYPES,
    TYPE_PAIR_PRIORS,
    RelationTypeDef,
)


class TestRelationTypeDef:
    def test_frozen_dataclass(self):
        td = RelationTypeDef(
            canonical_name="test",
            category="test_cat",
            aliases=("a", "b"),
            type_pairs=(("person", "organization"),),
        )
        assert td.canonical_name == "test"
        assert td.bidirectional is False
        assert td.description == ""

    def test_field_access(self):
        td = CANONICAL_RELATION_TYPES["works_at"]
        assert td.canonical_name == "works_at"
        assert td.category == "professional"
        assert "employed_by" in td.aliases
        assert ("person", "organization") in td.type_pairs


class TestCanonicalRelationTypes:
    def test_count(self):
        assert len(CANONICAL_RELATION_TYPES) == 39

    def test_all_names_lowercase_snake_case(self):
        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for name in CANONICAL_RELATION_TYPES:
            assert pattern.match(name), f"'{name}' is not lowercase snake_case"

    def test_no_empty_aliases(self):
        for name, typedef in CANONICAL_RELATION_TYPES.items():
            assert len(typedef.aliases) > 0, f"'{name}' has no aliases"

    def test_no_empty_type_pairs(self):
        for name, typedef in CANONICAL_RELATION_TYPES.items():
            assert len(typedef.type_pairs) > 0, f"'{name}' has no type_pairs"

    def test_categories_present(self):
        categories = {td.category for td in CANONICAL_RELATION_TYPES.values()}
        expected = {"professional", "spatial", "creation", "usage", "hierarchical", "temporal", "knowledge", "logical"}
        assert categories == expected


class TestAliasIndex:
    def test_every_canonical_is_self_alias(self):
        for name in CANONICAL_RELATION_TYPES:
            assert name in ALIAS_INDEX
            assert ALIAS_INDEX[name] == name

    def test_every_alias_maps_to_valid_canonical(self):
        for alias, canonical in ALIAS_INDEX.items():
            assert canonical in CANONICAL_RELATION_TYPES, f"Alias '{alias}' maps to unknown '{canonical}'"

    def test_no_duplicate_aliases_across_types(self):
        seen: dict[str, str] = {}
        for typedef in CANONICAL_RELATION_TYPES.values():
            for alias in typedef.aliases:
                if alias in seen:
                    assert False, (
                        f"Alias '{alias}' used by both '{seen[alias]}' and '{typedef.canonical_name}'"
                    )
                seen[alias] = typedef.canonical_name

    def test_all_keys_lowercase(self):
        for key in ALIAS_INDEX:
            assert key == key.lower(), f"Alias key '{key}' is not lowercase"


class TestTypePairPriors:
    def test_keys_are_lowercase(self):
        for src, tgt in TYPE_PAIR_PRIORS:
            assert src == src.lower(), f"Source type '{src}' not lowercase"
            assert tgt == tgt.lower(), f"Target type '{tgt}' not lowercase"

    def test_every_canonical_type_in_at_least_one_entry(self):
        all_types_in_priors = set()
        for types in TYPE_PAIR_PRIORS.values():
            all_types_in_priors.update(types)
        for name in CANONICAL_RELATION_TYPES:
            assert name in all_types_in_priors, f"'{name}' not in any TYPE_PAIR_PRIORS entry"

    def test_wildcard_types_have_star_star(self):
        wildcard_relations = {"related_to", "part_of", "type_of", "causes", "enables", "requires",
                              "derived_from", "contradicts", "similar_to", "supersedes"}
        for name in wildcard_relations:
            typedef = CANONICAL_RELATION_TYPES[name]
            assert ("*", "*") in typedef.type_pairs, f"'{name}' should have ('*', '*') type-pair"

    def test_works_at_pair(self):
        assert ("person", "organization") in TYPE_PAIR_PRIORS
        assert "works_at" in TYPE_PAIR_PRIORS[("person", "organization")]
