"""Unit tests for smartmemory.background.id_resolver — no infrastructure required."""
import pytest
from smartmemory.background.id_resolver import build_sha256_to_stored, filter_valid_relations
from smartmemory.models.memory_item import MemoryItem


def _make_entity(name: str, sha256_id: str, entity_type: str = "concept", confidence: float = 0.9) -> MemoryItem:
    return MemoryItem(
        content=name,
        item_id=sha256_id,
        memory_type="concept",
        metadata={"name": name, "entity_type": entity_type, "confidence": confidence},
    )


class TestBuildSha256ToStored:
    def test_entity_in_both_spaces_is_remapped(self):
        llm = [_make_entity("Python", "sha256abc", "language")]
        ruler_ids = {"python": "graph_node_001"}
        result = build_sha256_to_stored(llm, ruler_ids)
        assert result == {"sha256abc": "graph_node_001"}

    def test_entity_only_in_llm_space_is_not_remapped(self):
        llm = [_make_entity("Django", "sha256xyz")]
        ruler_ids = {"python": "graph_node_001"}
        result = build_sha256_to_stored(llm, ruler_ids)
        assert result == {}

    def test_entity_only_in_ruler_space_is_not_remapped(self):
        llm = []
        ruler_ids = {"python": "graph_node_001"}
        result = build_sha256_to_stored(llm, ruler_ids)
        assert result == {}

    def test_name_matching_is_case_insensitive(self):
        llm = [_make_entity("PYTHON", "sha256upper")]
        ruler_ids = {"python": "graph_node_001"}
        result = build_sha256_to_stored(llm, ruler_ids)
        assert result == {"sha256upper": "graph_node_001"}

    def test_empty_stored_id_is_skipped(self):
        llm = [_make_entity("Python", "sha256abc")]
        ruler_ids = {"python": ""}
        result = build_sha256_to_stored(llm, ruler_ids)
        assert result == {}

    def test_entity_with_no_name_is_skipped(self):
        entity = MemoryItem(content="", item_id="sha256nope", memory_type="concept", metadata={})
        result = build_sha256_to_stored([entity], {"python": "graph_node_001"})
        assert result == {}

    def test_multiple_entities_partial_overlap(self):
        llm = [
            _make_entity("Python", "sha_python"),
            _make_entity("Django", "sha_django"),
            _make_entity("Flask", "sha_flask"),
        ]
        ruler_ids = {"python": "g_python", "flask": "g_flask"}
        result = build_sha256_to_stored(llm, ruler_ids)
        assert result == {"sha_python": "g_python", "sha_flask": "g_flask"}
        assert "sha_django" not in result


class TestFilterValidRelations:
    def test_both_endpoints_valid_returns_remapped_relation(self):
        sha256_to_stored = {"sha_a": "node_a", "sha_b": "node_b"}
        net_new_ids: set[str] = set()
        rels = [{"source_id": "sha_a", "target_id": "sha_b", "relation_type": "USES"}]
        result = filter_valid_relations(rels, sha256_to_stored, net_new_ids)
        assert len(result) == 1
        assert result[0] == {"source_id": "node_a", "target_id": "node_b", "relation_type": "USES"}

    def test_source_missing_from_valid_ids_excluded(self):
        sha256_to_stored = {"sha_b": "node_b"}
        net_new_ids: set[str] = set()
        rels = [{"source_id": "sha_a", "target_id": "sha_b", "relation_type": "USES"}]
        result = filter_valid_relations(rels, sha256_to_stored, net_new_ids)
        assert result == []

    def test_target_missing_from_valid_ids_excluded(self):
        sha256_to_stored = {"sha_a": "node_a"}
        net_new_ids: set[str] = set()
        rels = [{"source_id": "sha_a", "target_id": "sha_missing", "relation_type": "USES"}]
        result = filter_valid_relations(rels, sha256_to_stored, net_new_ids)
        assert result == []

    def test_sha256_is_always_truthy_membership_check_used(self):
        # SHA-256 strings are always truthy — verify we check set membership, not truthiness
        sha256_to_stored = {}
        net_new_ids: set[str] = set()
        # Both sha256 ids are truthy strings but not in valid_ids
        rels = [{"source_id": "aaaa1111bbbb2222", "target_id": "cccc3333dddd4444", "relation_type": "X"}]
        result = filter_valid_relations(rels, sha256_to_stored, net_new_ids)
        assert result == []

    def test_net_new_ids_count_as_valid_endpoints(self):
        sha256_to_stored = {"sha_a": "node_a"}
        net_new_ids = {"net_new_b"}
        # source remaps to node_a (via sha256_to_stored), target is a net-new id
        rels = [{"source_id": "sha_a", "target_id": "net_new_b", "relation_type": "LINKS"}]
        result = filter_valid_relations(rels, sha256_to_stored, net_new_ids)
        assert len(result) == 1
        assert result[0]["source_id"] == "node_a"
        assert result[0]["target_id"] == "net_new_b"

    def test_empty_inputs_return_empty(self):
        assert filter_valid_relations([], {}, set()) == []

    def test_multiple_relations_partial_validity(self):
        sha256_to_stored = {"sha_a": "node_a", "sha_b": "node_b"}
        net_new_ids: set[str] = set()
        rels = [
            {"source_id": "sha_a", "target_id": "sha_b", "relation_type": "USES"},
            {"source_id": "sha_a", "target_id": "sha_unknown", "relation_type": "KNOWS"},
            {"source_id": "sha_b", "target_id": "sha_a", "relation_type": "USED_BY"},
        ]
        result = filter_valid_relations(rels, sha256_to_stored, net_new_ids)
        assert len(result) == 2
        types = {r["relation_type"] for r in result}
        assert types == {"USES", "USED_BY"}
