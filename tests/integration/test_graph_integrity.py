"""Graph integrity tests — assert the knowledge graph is actually populated.

These tests exist because the graph was silently empty on lite mode for
months while text search masked the failure. They verify the MECHANISM
(entity nodes, edges, traversal) not just the OUTCOME (search returns text).

If any of these fail, the knowledge graph is broken and SmartMemory is
operating as a flat vector store — its core differentiator is gone.
"""

import pytest
from smartmemory.tools.factory import create_lite_memory
from smartmemory.pipeline.config import PipelineConfig


@pytest.fixture()
def memory(tmp_path):
    mem = create_lite_memory(
        data_dir=str(tmp_path),
        pipeline_profile=PipelineConfig.lite(llm_enabled=False),
    )
    yield mem
    try:
        mem._graph.backend.close()
    except Exception:
        pass


@pytest.mark.integration
class TestGraphIntegrity:
    """Verify that ingest creates entity nodes and edges, not just memory nodes."""

    def test_ingest_creates_entity_nodes(self, memory):
        """spaCy should extract at least one named entity as a graph node."""
        memory.ingest("Alice works at Acme Corp in New York", context={"memory_type": "semantic"})
        nodes = memory._graph.backend.get_all_nodes()
        entity_nodes = [n for n in nodes if n.get("memory_type") == "entity"]
        assert len(entity_nodes) >= 1, (
            f"No entity nodes after ingest. Node types: "
            f"{set(n.get('memory_type') for n in nodes)}. "
            f"add_dual_node may be falling back to single-node."
        )

    def test_entity_edges_created(self, memory):
        """Entity nodes must be linked to memory nodes via CONTAINS_ENTITY / MENTIONED_IN."""
        memory.ingest("Alice works at Acme Corp", context={"memory_type": "semantic"})
        edges = memory._graph.backend.get_all_edges()
        edge_types = {e.get("edge_type") for e in edges}
        assert "CONTAINS_ENTITY" in edge_types, (
            f"No CONTAINS_ENTITY edges. Edge types: {edge_types}"
        )
        assert "MENTIONED_IN" in edge_types, (
            f"No MENTIONED_IN edges. Edge types: {edge_types}"
        )

    def test_entity_dedup_across_memories(self, memory):
        """Same entity mentioned in two memories should create ONE entity node with two edges."""
        memory.ingest("Alice leads Project Atlas", context={"memory_type": "episodic"})
        memory.ingest("Alice presented at PyCon 2025", context={"memory_type": "episodic"})

        nodes = memory._graph.backend.get_all_nodes()
        entity_nodes = [n for n in nodes if n.get("memory_type") == "entity"]
        alice_nodes = [n for n in entity_nodes if "alice" in (n.get("content") or "").lower()]

        # Should be 1 Alice node, not 2
        assert len(alice_nodes) == 1, (
            f"Expected 1 Alice entity node, got {len(alice_nodes)}. "
            f"Canonical key dedup may be broken."
        )

        # That one Alice node should have 2 MENTIONED_IN edges
        alice_id = alice_nodes[0]["item_id"]
        edges = memory._graph.backend.get_edges_for_node(alice_id)
        mentioned_edges = [e for e in edges if e.get("edge_type") == "MENTIONED_IN"]
        assert len(mentioned_edges) == 2, (
            f"Alice entity should link to 2 memories, got {len(mentioned_edges)}"
        )

    def test_entity_nodes_excluded_from_search(self, memory):
        """Entity nodes must never appear in user-facing search results."""
        memory.ingest("Bob is the CTO of Acme Corp", context={"memory_type": "semantic"})
        results = memory.search("Bob", top_k=10)
        for r in results:
            assert getattr(r, "memory_type", "") != "entity", (
                f"Entity node leaked into search results: {r.content[:40]}"
            )

    def test_graph_search_finds_linked_memories(self, memory):
        """Search for an entity name should find memories linked via graph edges."""
        memory.ingest("Alice leads Project Atlas at Acme Corp", context={"memory_type": "episodic"})
        memory.ingest("Django is a Python web framework", context={"memory_type": "semantic"})

        # "Acme" should find Alice's memory (via entity), not Django's
        results = memory.search("Acme Corp", top_k=2)
        assert len(results) >= 1, "Graph search for 'Acme Corp' returned nothing"
        assert any("Acme" in (r.content or "") for r in results), (
            f"Graph search didn't find Acme memory: {[r.content[:40] for r in results]}"
        )

    def test_no_silent_dual_node_fallback(self, memory):
        """add_dual_node must exist on the backend — no silent single-node fallback."""
        backend = memory._graph.backend
        assert hasattr(backend, "add_dual_node"), (
            f"{type(backend).__name__} does not implement add_dual_node. "
            f"Entity nodes will be silently dropped."
        )
        assert callable(getattr(backend, "add_dual_node")), (
            "add_dual_node exists but is not callable"
        )
