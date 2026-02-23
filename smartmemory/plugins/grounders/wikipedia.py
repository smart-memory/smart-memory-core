"""
Wikipedia Grounder Plugin

Grounds memory items to Wikipedia articles for provenance and validation.
"""

import logging

from smartmemory.plugins.base import GrounderPlugin, PluginMetadata

logger = logging.getLogger(__name__)


class WikipediaGrounder(GrounderPlugin):
    """
    Grounds entities to Wikipedia articles for provenance.

    This grounder links memory items to Wikipedia articles based on
    recognized entities, providing external validation and context.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="wikipedia_grounder",
            version="1.0.0",
            author="SmartMemory Team",
            description="Grounds entities to Wikipedia articles for provenance",
            plugin_type="grounder",
            dependencies=["wikipedia>=1.4.0"],
            min_smartmemory_version="0.1.0",
            tags=["grounding", "wikipedia", "provenance", "validation"],
            requires_network=True,
        )

    def ground(self, item, entities, graph) -> list:
        """
        Ground entities to Wikipedia articles.

        Args:
            item: The memory item (for context)
            entities: List of entity MemoryItems to ground
            graph: The graph backend for creating edges

        Returns:
            list: List of provenance candidate node IDs (Wikipedia nodes created)
        """
        self._graph_hits = 0  # Count of entities resolved from graph (no API call)

        if not entities:
            return []

        provenance_candidates = []

        try:
            entity_map: dict = {}  # Map entity name -> MemoryItem or dict
            needs_lookup: list[str] = []  # Entity names not yet grounded

            for entity in entities:
                name = None
                if isinstance(entity, dict):
                    name = entity.get("name") or (entity.get("metadata") or {}).get("name")
                elif hasattr(entity, "metadata") and entity.metadata:
                    name = entity.metadata.get("name")
                if name:
                    entity_map[name] = entity

            if not entity_map:
                return []

            # Check graph first — skip API call for already-grounded entities
            for entity_name in entity_map:
                wiki_id = f"wikipedia:{entity_name.replace(' ', '_').lower()}"
                existing_node = graph.get_node(wiki_id)
                if existing_node:
                    logger.debug("Reusing existing Wikipedia node: %s", wiki_id)
                    provenance_candidates.append(wiki_id)
                    self._ensure_edge(graph, entity_map[entity_name], wiki_id)
                    self._graph_hits += 1
                else:
                    needs_lookup.append(entity_name)

            if not needs_lookup:
                return provenance_candidates

            # Only call Wikipedia API for entities not already in the graph
            from smartmemory.integration.wikipedia_client import WikipediaClient

            wiki_client = WikipediaClient()
            for entity_name in needs_lookup:
                article = wiki_client.get_article(entity_name)
                if not article.get("exists"):
                    continue

                wiki_id = f"wikipedia:{entity_name.replace(' ', '_').lower()}"
                node_properties = {
                    "entity": entity_name,
                    "summary": article.get("summary", "")[:300],
                    "categories": article.get("categories", []),
                    "url": article.get("url"),
                    "type": "wikipedia_article",
                    "node_category": "grounding",
                }
                graph.add_node(item_id=wiki_id, properties=node_properties, is_global=True)
                logger.info("Created Wikipedia node: %s", wiki_id)

                self._ensure_edge(graph, entity_map.get(entity_name), wiki_id)
                provenance_candidates.append(wiki_id)

        except Exception as e:
            logger.error("Error grounding to Wikipedia: %s", e, exc_info=True)
        return provenance_candidates

    @staticmethod
    def _ensure_edge(graph, entity_item, wiki_id: str) -> None:
        """Create GROUNDED_IN edge from entity to Wikipedia node if possible."""
        eid = None
        if isinstance(entity_item, dict):
            eid = entity_item.get("item_id")
        elif entity_item and hasattr(entity_item, "item_id"):
            eid = entity_item.item_id
        if eid:
            graph.add_edge(eid, wiki_id, edge_type="GROUNDED_IN", properties={}, is_global=True)
        else:
            logger.warning("No entity item_id for Wikipedia node %s, skipping edge", wiki_id)
