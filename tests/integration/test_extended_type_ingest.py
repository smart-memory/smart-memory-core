"""
Targeted test: verify extended memory types (decision, opinion, observation, reasoning)
survive the full ingest → store → retrieve cycle.

Traces: MCP ingest request → smart_memory.ingest() → pipeline → node_types → graph → list/get

Run:
    PYTHONPATH=. pytest tests/integration/test_extended_type_ingest.py -v -s
"""
import pytest
from smartmemory import SmartMemory
from smartmemory.models.memory_item import MEMORY_TYPES


@pytest.fixture
def memory():
    """Create a SmartMemory instance with real infrastructure."""
    sm = SmartMemory()
    yield sm


@pytest.mark.integration
class TestExtendedTypeIngest:
    """Verify extended memory types survive the full pipeline."""

    @pytest.mark.parametrize("memory_type", ["decision", "opinion", "observation", "reasoning"])
    def test_ingest_preserves_extended_type(self, memory, memory_type):
        """Ingest with context memory_type and verify it's stored correctly."""
        content = f"Test {memory_type} memory: this is a {memory_type} about FalkorDB."

        item_id = memory.ingest(
            item=content,
            context={"memory_type": memory_type},
        )

        assert item_id is not None, "ingest() returned None"
        # Handle dict return from ingest
        if isinstance(item_id, dict):
            item_id = item_id.get("item_id") or item_id.get("memory_node_id")

        # Retrieve and check type
        item = memory.get(item_id)
        assert item is not None, f"get({item_id}) returned None"

        actual_type = getattr(item, "memory_type", None)
        if isinstance(item, dict):
            actual_type = item.get("memory_type")

        assert actual_type == memory_type, (
            f"Expected memory_type='{memory_type}', got '{actual_type}'. "
            f"Extended type was coerced during pipeline."
        )

    def test_ingest_core_type_still_works(self, memory):
        """Sanity check: core type 'episodic' still works through ingest."""
        item_id = memory.ingest(
            item="Had coffee with Alice at the park this morning.",
            context={"memory_type": "episodic"},
        )
        assert item_id is not None
        if isinstance(item_id, dict):
            item_id = item_id.get("item_id") or item_id.get("memory_node_id")

        item = memory.get(item_id)
        assert item is not None
        actual_type = getattr(item, "memory_type", None)
        if isinstance(item, dict):
            actual_type = item.get("memory_type")
        assert actual_type == "episodic"

    def test_extended_types_in_memory_types_set(self):
        """All extended types must be in the MEMORY_TYPES set."""
        for t in ["decision", "opinion", "observation", "reasoning"]:
            assert t in MEMORY_TYPES, f"'{t}' not in MEMORY_TYPES"

    def test_search_finds_extended_type_items(self, memory):
        """Items with extended types must appear in search results (viewer data source)."""
        item_id = memory.ingest(
            item="We decided to use Redis for caching because of low latency.",
            context={"memory_type": "decision"},
        )
        if isinstance(item_id, dict):
            item_id = item_id.get("item_id") or item_id.get("memory_node_id")

        # Search — this is what the viewer ultimately relies on
        results = memory.search("Redis caching", top_k=10)

        found = False
        result_list = results if isinstance(results, list) else results.get("results", [])
        for r in result_list:
            r_id = r.get("item_id") if isinstance(r, dict) else getattr(r, "item_id", None)
            if r_id == item_id:
                found = True
                r_type = r.get("memory_type") if isinstance(r, dict) else getattr(r, "memory_type", None)
                assert r_type == "decision", f"Search result has type '{r_type}', expected 'decision'"
                break

        assert found, f"Item {item_id} not found in search() — viewer would not see it"
