from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.manager import get_plugin_manager


class Grounding:
    """
    Handles grounding/provenance logic using the plugin system.

    Grounders are loaded from the PluginRegistry, which discovers them automatically
    from built-in plugins and external plugins via entry points.
    """

    def __init__(self, graph):
        """Initialize grounding component with graph backend."""
        self.graph = graph

        # Get plugin manager and registry
        plugin_manager = get_plugin_manager()
        self.plugin_registry = plugin_manager.registry

        # Get default grounder (wikipedia_grounder)
        self.default_grounder_class = self.plugin_registry.get_grounder("wikipedia_grounder")
        if self.default_grounder_class:
            self.default_grounder = self.default_grounder_class()
        else:
            self.default_grounder = None

    def ground(self, context):
        """
        Ground a memory item using the plugin system.

        Uses registered grounder plugins to link memory items to external
        knowledge sources for provenance and validation.
        """
        item = context.get("item") if isinstance(context, dict) else None
        if not item or not hasattr(item, "item_id"):
            return
        item_id = item.item_id
        with trace_span("pipeline.ground.wikipedia_grounder", {"memory_id": item_id}):
            source_url = context.get("source_url")
            validation = context.get("validation")

            # Get entities from node
            node = self.graph.get_node(item_id)
            if not node:
                return

            # Extract entities from MemoryItem or dict
            if hasattr(node, "metadata"):
                # MemoryItem object
                entities = node.metadata.get("semantic_entities", [])
            elif isinstance(node, dict):
                # Dict
                entities = node.get("semantic_entities") or (node.get("metadata") or {}).get("semantic_entities", [])
            else:
                entities = []
            if not entities:
                return

            # Use default grounder (Wikipedia) if available
            if self.default_grounder:
                try:
                    provenance_candidates = self.default_grounder.ground(item, entities, self.graph)

                    # Update node with provenance information
                    if provenance_candidates:
                        # Extract provenance URL from the Wikipedia nodes already in the graph
                        provenance_url = None
                        for prov_id in provenance_candidates:
                            wiki_node = self.graph.get_node(prov_id)
                            if not wiki_node:
                                continue
                            url = wiki_node.get("url") if isinstance(wiki_node, dict) else (getattr(wiki_node, "metadata", None) or {}).get("url")
                            if url:
                                provenance_url = url
                                break

                        node["provenance"] = {
                            "type": "wikipedia",
                            "entities": entities,
                            "source_url": provenance_url or source_url,
                            "validation": validation,
                        }
                        self.graph.add_node(item_id=item_id, properties=node)
                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(f"Error grounding item {item_id}: {e}")

            # Fallback: use provided provenance_candidates if grounder not available
            elif context.get("provenance_candidates"):
                provenance_candidates = context.get("provenance_candidates")
                for prov_id in provenance_candidates:
                    self.graph.add_edge(item_id, prov_id, edge_type="GROUNDED_IN", properties={}, is_global=True)
