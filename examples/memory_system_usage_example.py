"""
Example: Using the Agentic Memory System

This script demonstrates basic usage of the unified SemanticMemoryGraph interface:
- Adding items to the memory graph
- Searching the memory graph
- Removing items from the memory graph
- Using the Zettel interface

This is NOT a formal test. For automated testing, see the test suite.
"""
import logging
from datetime import datetime, UTC

from smartmemory.models.memory_item import MemoryItem
from smartmemory.smart_memory import SmartMemory

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_memory_operations(memory: SmartMemory):
    """Demonstrates ingesting, searching, and removing memories."""
    logger.info("\n--- Ingesting Example ---")
    memory_item_1 = MemoryItem(
        content="The user attended the AI conference on Tuesday.",
        valid_start_time=datetime.now(UTC),
        metadata={'source': 'user_input', 'importance': 0.8}
    )
    # Use ingest() for full pipeline (extract → store → link → enrich → evolve)
    item_id = memory.ingest(memory_item_1)
    memory_item_1.item_id = item_id  # Store the returned ID
    logger.info(f"Ingested item to memory graph: {item_id}")

    logger.info("\n--- Search Example ---")
    query = "AI conference"
    search_results = memory.search(query)

    logger.info(f"Search results for query: '{query}'")
    for item in search_results:
        print(f"  - {item}")

    logger.info("\n--- Remove Example ---")
    memory.delete(memory_item_1.item_id)
    logger.info(f"Removed item from memory graph: {memory_item_1.item_id}")


def main():
    # Initialize the unified memory store
    memory = SmartMemory()

    # Run example operations
    run_memory_operations(memory)

    # Memory summary
    try:
        summary = memory.summary()
        logger.info("Memory summary: %s", summary)
    except Exception as e:
        logger.warning("Could not get memory summary: %s", e)


if __name__ == "__main__":
    logger.info("Starting memory system usage example...")
    main()
    logger.info("Memory system usage example finished.")
