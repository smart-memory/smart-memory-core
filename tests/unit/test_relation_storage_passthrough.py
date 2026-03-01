"""Tests that ONTO-PUB-3 relation metadata survives the storage path.

Tests StoreStage._process_relations() and StoragePipeline.process_extracted_relations()
forward canonical_type, raw_predicate, normalization_confidence, plausibility_score.
"""

from unittest.mock import MagicMock, patch


class TestStoreStagePassthrough:
    """Test that store.py:_process_relations() forwards all dict keys."""

    def test_forwards_canonical_type(self):
        from smartmemory.pipeline.stages.store import StoreStage

        stage = StoreStage(memory=MagicMock())

        # Build entities with item_id (MemoryItem-like)
        entity_a = MagicMock()
        entity_a.item_id = "hash_a"
        entity_a.metadata = {"name": "Alice"}

        entity_b = MagicMock()
        entity_b.item_id = "hash_b"
        entity_b.metadata = {"name": "Acme"}

        entities = [entity_a, entity_b]
        entity_ids = {"Alice": "graph_a", "Acme": "graph_b"}

        relations = [{
            "source_id": "hash_a",
            "target_id": "hash_b",
            "relation_type": "works_at",
            "canonical_type": "works_at",
            "raw_predicate": "Employed By",
            "normalization_confidence": 1.0,
            "plausibility_score": 1.0,
        }]

        state = MagicMock()
        state.entity_ids = entity_ids
        state._context = {}

        # Patch StoragePipeline to capture what gets passed
        captured_relations = []

        def fake_process(ctx, item_id, rels):
            captured_relations.extend(rels)

        with patch("smartmemory.memory.ingestion.storage.StoragePipeline") as MockSP, \
             patch("smartmemory.memory.ingestion.observer.IngestionObserver"):
            mock_sp_instance = MagicMock()
            mock_sp_instance.process_extracted_relations = fake_process
            MockSP.return_value = mock_sp_instance

            stage._process_relations(state, "item_1", entities, relations, entity_ids)

        assert len(captured_relations) == 1
        r = captured_relations[0]
        assert r["source_id"] == "graph_a"
        assert r["target_id"] == "graph_b"
        assert r["canonical_type"] == "works_at"
        assert r["raw_predicate"] == "Employed By"
        assert r["normalization_confidence"] == 1.0
        assert r["plausibility_score"] == 1.0

    def test_backward_compat_no_onto_pub3_fields(self):
        """Relations without ONTO-PUB-3 fields still work."""
        from smartmemory.pipeline.stages.store import StoreStage

        stage = StoreStage(memory=MagicMock())

        entity_a = MagicMock()
        entity_a.item_id = "hash_a"
        entity_a.metadata = {"name": "Alice"}

        entity_b = MagicMock()
        entity_b.item_id = "hash_b"
        entity_b.metadata = {"name": "Acme"}

        entities = [entity_a, entity_b]
        entity_ids = {"Alice": "graph_a", "Acme": "graph_b"}

        relations = [{
            "source_id": "hash_a",
            "target_id": "hash_b",
            "relation_type": "WORKS_AT",
        }]

        state = MagicMock()
        state.entity_ids = entity_ids
        state._context = {}

        captured = []

        def fake_process(ctx, item_id, rels):
            captured.extend(rels)

        with patch("smartmemory.memory.ingestion.storage.StoragePipeline") as MockSP, \
             patch("smartmemory.memory.ingestion.observer.IngestionObserver"):
            mock_sp_instance = MagicMock()
            mock_sp_instance.process_extracted_relations = fake_process
            MockSP.return_value = mock_sp_instance

            stage._process_relations(state, "item_1", entities, relations, entity_ids)

        assert len(captured) == 1
        r = captured[0]
        assert r["source_id"] == "graph_a"
        assert r["relation_type"] == "WORKS_AT"
        assert "canonical_type" not in r


class TestStoragePipelinePassthrough:
    """Test that storage.py includes ONTO-PUB-3 metadata in edge properties."""

    def test_includes_metadata_in_edge_properties(self):
        from smartmemory.memory.ingestion.storage import StoragePipeline

        mock_memory = MagicMock()
        mock_graph = MagicMock()
        mock_memory._graph = mock_graph
        mock_graph.add_edge.return_value = "edge_1"

        mock_observer = MagicMock()
        pipeline = StoragePipeline(mock_memory, mock_observer)

        item = MagicMock()
        item.created_at = MagicMock()
        item.created_at.isoformat.return_value = "2026-03-01T00:00:00"

        context = {
            "item": item,
            "entity_ids": {"graph_a": "graph_a", "graph_b": "graph_b"},
        }

        relations = [{
            "source_id": "graph_a",
            "target_id": "graph_b",
            "relation_type": "WORKS_AT",
            "canonical_type": "works_at",
            "raw_predicate": "Employed By",
            "normalization_confidence": 1.0,
            "plausibility_score": 0.95,
        }]

        pipeline.process_extracted_relations(context, "item_1", relations)

        mock_graph.add_edge.assert_called_once()
        call_kwargs = mock_graph.add_edge.call_args
        props = call_kwargs.kwargs.get("properties") or call_kwargs[1].get("properties", {})
        assert props["canonical_type"] == "works_at"
        assert props["raw_predicate"] == "Employed By"
        assert props["normalization_confidence"] == 1.0
        assert props["plausibility_score"] == 0.95
        assert props["created_from_triple"] is True

    def test_backward_compat_no_metadata(self):
        """Relations without ONTO-PUB-3 fields don't add extra properties."""
        from smartmemory.memory.ingestion.storage import StoragePipeline

        mock_memory = MagicMock()
        mock_graph = MagicMock()
        mock_memory._graph = mock_graph
        mock_graph.add_edge.return_value = "edge_1"

        mock_observer = MagicMock()
        pipeline = StoragePipeline(mock_memory, mock_observer)

        item = MagicMock()
        item.created_at = MagicMock()
        item.created_at.isoformat.return_value = "2026-03-01T00:00:00"

        context = {
            "item": item,
            "entity_ids": {"graph_a": "graph_a", "graph_b": "graph_b"},
        }

        relations = [{
            "source_id": "graph_a",
            "target_id": "graph_b",
            "relation_type": "WORKS_AT",
        }]

        pipeline.process_extracted_relations(context, "item_1", relations)

        call_kwargs = mock_graph.add_edge.call_args
        props = call_kwargs.kwargs.get("properties") or call_kwargs[1].get("properties", {})
        assert "canonical_type" not in props
        assert "raw_predicate" not in props
        assert props["created_from_triple"] is True
