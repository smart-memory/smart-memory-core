import importlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Union, Any, Dict, List, Optional

from smartmemory.graph.models.node_types import NodeTypeProcessor, EntityNodeType
from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.base import BaseHandler
from smartmemory.utils import get_config

if TYPE_CHECKING:
    from smartmemory.evolution.queue import EvolutionQueue

logger = logging.getLogger(__name__)


class CRUD(BaseHandler):
    """
    CRUD operations with dual-node architecture support.
    Handles both memory nodes (for system processing) and entity nodes (for domain modeling).
    """

    def __init__(self, graph, evolution_queue: Optional["EvolutionQueue"] = None, smart_memory=None):
        """Initialize CRUD with graph backend and node type processor.

        Args:
            graph: SmartGraph instance.
            evolution_queue: Optional EvolutionQueue for CORE-EVO-LIVE-1
                mutation event emission.
            smart_memory: Optional SmartMemory instance for lazy worker start.
        """
        self._graph = graph
        self.node_processor = NodeTypeProcessor(graph)
        self._evolution_queue = evolution_queue
        self._smart_memory = smart_memory  # For lazy worker start

    def normalize_item(self, item: Union[MemoryItem, dict, Any]) -> MemoryItem:
        """
        Convert various input types to MemoryItem.
        Centralizes conversion logic to eliminate mixed abstractions.
        """
        # Already a MemoryItem
        if isinstance(item, MemoryItem):
            return item

        # Domain models with to_memory_item method
        if hasattr(item, 'to_memory_item'):
            return item.to_memory_item()

        # Dictionary input
        if isinstance(item, dict):
            return MemoryItem(**item)

        # String or other input - convert to content
        return MemoryItem(content=str(item))

    def denormalize_item(self, item: MemoryItem) -> Any:
        """
        Convert MemoryItem back to domain models if type is specified.
        Dynamically scans smartmemory.models for matching class.
        """
        if item is not None and isinstance(getattr(item, 'memory_type', None), str):
            cls_name = item.memory_type
            try:
                model_pkg = importlib.import_module('smartmemory.models')
                for name in dir(model_pkg):
                    attr = getattr(model_pkg, name)
                    if inspect.isclass(attr) and name == cls_name and hasattr(attr, 'from_memory_item'):
                        return attr.from_memory_item(item)
            except ImportError:
                pass
        return item

    def add(self, item: Union[MemoryItem, dict, Any], **kwargs) -> Dict[str, Any]:
        """Add item using dual-node architecture with selective embedding generation.

        Behavior:
        - Always create via dual-node path. If no ontology_extraction is provided,
          we still create a memory node (entities=[]).
        - Generate embeddings for memory nodes (semantic, episodic, procedural)
        - Skip embeddings for entity/relation nodes (not searchable content)

        Returns:
            Dict with 'memory_node_id' (str) and 'entity_node_ids' (List[str]).
        """
        normalized_item = self.normalize_item(item)
        item_id = normalized_item.item_id or kwargs.get("key")

        ontology_extraction = kwargs.get('ontology_extraction')

        # Always create via dual-node path; entities only if ontology_extraction provided
        dual_spec = self.node_processor.extract_dual_node_spec_from_memory_item(
            normalized_item,
            ontology_extraction
        )
        result = self.node_processor.create_dual_node_structure(dual_spec)

        # Generate embedding for memory nodes (makes them searchable)
        memory_node_id = result['memory_node_id']
        if self._should_generate_embedding(normalized_item):
            try:
                self._generate_and_store_embedding(normalized_item, memory_node_id)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {memory_node_id}: {e}")
                # Non-fatal: item is still in graph, searchable via text fallback

        # Emit evolution event for the new memory node
        self._emit_mutation("add", memory_node_id, normalized_item.memory_type or "semantic")

        # Always return the full creation result (including entity_node_ids)
        # so that flow.py can map extraction-time entity IDs to graph node IDs
        return result
    
    def _should_generate_embedding(self, item: MemoryItem) -> bool:
        """Determine if item needs embedding based on node type and memory type."""
        # Check node category (memory vs entity vs relation)
        node_category = item.metadata.get('node_category') if item.metadata else None
        
        # Only generate for memory nodes
        if node_category and node_category != 'memory':
            return False
        
        # Check memory type
        memory_type = getattr(item, 'memory_type', 'semantic')
        
        # Always generate for searchable memory types
        if memory_type in ['semantic', 'episodic', 'procedural']:
            return True
        
        # Never generate for temporary/metadata types
        if memory_type in ['working', 'metadata', 'relation']:
            return False
        
        # Check configuration for other types
        try:
            embedding_cfg = get_config('embeddings') or {}
            return embedding_cfg.get(f'enable_{memory_type}', True)
        except Exception:
            return True  # Default to generating embeddings
    
    def _generate_and_store_embedding(self, item: MemoryItem, item_id: str):
        """Generate embedding and store in vector store."""
        from smartmemory.plugins.embedding import create_embeddings
        from smartmemory.stores.vector.vector_store import VectorStore
        
        # Generate embedding if not already present
        if not hasattr(item, 'embedding') or item.embedding is None:
            item.embedding = create_embeddings(str(item.content))
        
        # Store in vector store if embedding was generated
        if item.embedding is not None:
            vector_store = VectorStore()
            vector_store.upsert(
                item_id=str(item_id),
                embedding=item.embedding,
                metadata=item.metadata or {},
                node_ids=[str(item_id)],
                is_global=False  # Respect workspace scoping
            )
            logger.debug(f"Generated and stored embedding for {item_id}")

    # Legacy single-node path removed

    def get(self, item_id: str, **kwargs) -> Any:
        """Get item and convert back to domain models if applicable."""
        node = self._graph.get_node(item_id)
        if node:
            # Handle case where graph backend returns MemoryItem directly
            if isinstance(node, MemoryItem):
                return node

            # Handle case where graph backend returns dict
            if isinstance(node, dict):
                if 'history' in node and isinstance(node['history'], str):
                    try:
                        node['history'] = json.loads(node['history'])
                    except Exception:
                        node['history'] = []
                return self.denormalize_item(node)

            # Handle other cases - try to convert to MemoryItem
            try:
                return self.denormalize_item(node)
            except Exception:
                import warnings
                warnings.warn(f"CRUD.get: Unable to process node for key {item_id}, type {type(node)}")
                return None
        return None

    def update(self, item: Union[MemoryItem, dict, Any], **kwargs) -> str:
        """Update item with dual-node architecture support.
        
        Returns:
            item_id of the updated item
        """
        normalized_item = self.normalize_item(item)
        key = normalized_item.item_id

        # Get existing node properties
        existing_node = self._graph.get_node(key)
        if not existing_node:
            raise ValueError(f"Node {key} not found in graph.")

        # Check if this is a memory node (part of dual-node architecture)
        # Handle both MemoryItem objects and dict returns from backend
        if isinstance(existing_node, MemoryItem):
            existing_dict = existing_node.to_dict()
        else:
            # Legacy dict format
            existing_dict = dict(existing_node)
        
        node_category = existing_dict.get('node_category')

        # Memory nodes are the supported path; update properties
        self._update_memory_node(normalized_item, existing_dict)

        # Emit evolution event for update
        memory_type = existing_dict.get("memory_type", "semantic")
        changed_props = {}
        if normalized_item.content is not None:
            changed_props["content"] = normalized_item.content
        if normalized_item.metadata:
            changed_props.update(normalized_item.metadata)
        self._emit_mutation("update", key, memory_type, properties=changed_props)

        # Return the item_id to indicate success
        return key

    def _update_memory_node(self, normalized_item: MemoryItem, existing_properties: Dict[str, Any]):
        """Update a memory node while preserving entity relationships."""
        key = normalized_item.item_id

        # Start with existing properties to preserve them
        properties = dict(existing_properties)

        # Update content if provided
        if normalized_item.content is not None:
            properties["content"] = normalized_item.content

        # Merge metadata intelligently
        new_metadata = normalized_item.metadata or {}
        for key_meta, value in new_metadata.items():
            # Don't overwrite system properties
            if key_meta not in ['node_category', 'memory_type']:
                properties[key_meta] = value

        # Handle history serialization if needed
        if 'history' in properties and isinstance(properties['history'], list):
            properties['history'] = json.dumps(properties['history'])

        # Extract memory type, preserving existing if not specified
        memory_type = properties.get('memory_type', 'semantic')

        # Update the memory node (entity nodes remain unchanged)
        self._graph.add_node(item_id=key, properties=properties, memory_type=memory_type)

    def delete(self, item_id: str, **kwargs) -> bool:
        """Delete memory item and cascade to vector store and Vec_* nodes."""
        import logging
        logger = logging.getLogger(__name__)

        # CORE-EVO-LIVE-1: capture pre-delete snapshot for evolution events
        pre_delete_props = None
        pre_delete_neighbors = None
        pre_delete_memory_type = "semantic"
        if self._evolution_queue is not None:
            try:
                node = self._graph.get_node(item_id)
                if node:
                    if isinstance(node, MemoryItem):
                        pre_delete_props = node.to_dict() if hasattr(node, "to_dict") else {"content": node.content}
                        pre_delete_memory_type = node.memory_type or "semantic"
                    elif isinstance(node, dict):
                        pre_delete_props = dict(node)
                        pre_delete_memory_type = node.get("memory_type", "semantic")
                    # Get edges before deletion
                    backend = getattr(self._graph, "backend", None)
                    if backend and hasattr(backend, "get_edges_for_node"):
                        raw_edges = backend.get_edges_for_node(item_id)
                        pre_delete_neighbors = [
                            {
                                "id": e["target_id"] if e["source_id"] == item_id else e["source_id"],
                                "edge_type": e.get("edge_type", ""),
                                "direction": "outgoing" if e["source_id"] == item_id else "incoming",
                            }
                            for e in raw_edges
                        ]
            except Exception as exc:
                logger.debug("CRUD.delete: failed to capture pre-delete snapshot: %s", exc)

        # 1. Delete from graph (DETACH DELETE removes edges automatically)
        self._graph.remove_node(item_id)

        # Emit evolution event with pre-delete snapshot
        self._emit_mutation(
            "delete", item_id, pre_delete_memory_type,
            properties=pre_delete_props, neighbors=pre_delete_neighbors,
        )
        
        # 2. Delete from vector store
        try:
            from smartmemory.stores.vector.vector_store import VectorStore
            vector_store = VectorStore()
            vector_store.delete(item_id)
            logger.debug(f"Deleted vector embedding for {item_id}")
        except Exception as e:
            logger.warning(f"Failed to delete vector for {item_id}: {e}")
        
        # 3. Delete Vec_* graph nodes that reference this item_id
        try:
            # Query for Vec_* nodes with this item_id in metadata
            if hasattr(self._graph, 'backend'):
                backend = self._graph.backend
                # FalkorDB query to find and delete Vec_* nodes
                query = "MATCH (v) WHERE v.id = $item_id OR v.item_id = $item_id DELETE v"
                if hasattr(backend, '_query'):
                    backend._query(query, {'item_id': item_id})
                    logger.debug(f"Deleted Vec_* nodes for {item_id}")
        except Exception as e:
            logger.warning(f"Failed to delete Vec_* nodes for {item_id}: {e}")
        
        return True

    def add_tags(self, item_id: str, tags: list) -> bool:
        item = self._graph.get_node(item_id)
        if item is None:
            return False
        tag_set = set(item.get('tags', []))
        tag_set.update(tags)
        item['tags'] = list(tag_set)
        self._graph.add_node(item_id=item_id, properties=item)
        return True

    def search_memory_nodes(self, memory_type: str = None, **filters) -> List[Dict[str, Any]]:
        """Search memory nodes (dual-node architecture only). Supports core and extended types."""
        if memory_type:
            return self.node_processor.query_memory_nodes(memory_type.lower(), **filters)
        return self.node_processor.query_memory_nodes(**filters)

    # Legacy search removed

    def search_entity_nodes(self, entity_type: str = None, **filters) -> List[Dict[str, Any]]:
        """Search entity nodes specifically (dual-node architecture)."""
        if entity_type:
            try:
                ent_type = EntityNodeType(entity_type.lower())
                return self.node_processor.query_entity_nodes(ent_type, **filters)
            except ValueError:
                return []
        else:
            return self.node_processor.query_entity_nodes(**filters)

    def update_memory_node(self, item_id: str, properties: Dict[str, Any], write_mode: str | None = None) -> Dict[str, Any]:
        """Update a memory node's properties with merge or replace semantics.

        - write_mode is determined by argument or config ingestion.enrichment.write_mode.
          Defaults to 'merge'.
        - Preserves required system fields like memory_type and node_category when replacing.
        - Returns the updated properties dict.
        """
        # Resolve write mode from config if not provided
        if write_mode is None:
            try:
                ingestion_cfg = get_config('ingestion') or {}
                enrichment_cfg = ingestion_cfg.get('enrichment') or {} if isinstance(ingestion_cfg, dict) else {}
                write_mode = enrichment_cfg.get('write_mode', 'merge')
            except Exception:
                write_mode = 'merge'

        existing = self._graph.get_node(item_id)
        if not existing:
            raise ValueError(f"Node {item_id} not found in graph.")

        # Normalize existing node to a flat dict
        if isinstance(existing, MemoryItem):
            existing_dict: Dict[str, Any] = existing.to_dict()
            metadata = existing_dict.pop('metadata', {})
            if isinstance(metadata, dict):
                # Merge metadata into top-level to align with backend dict shape
                for k, v in metadata.items():
                    if k not in existing_dict:
                        existing_dict[k] = v
        else:
            # Ensure dict copy
            existing_dict = dict(existing)
        existing_memory_type = existing_dict.get('memory_type', 'semantic')
        existing_node_category = existing_dict.get('node_category', 'memory')

        if (write_mode or 'merge').lower() == 'replace':
            new_props: Dict[str, Any] = dict(properties or {})
            # Preserve system fields
            new_props.setdefault('memory_type', existing_memory_type)
            new_props.setdefault('node_category', existing_node_category)
            # Mark replace mode for backend
            new_props['_write_mode'] = 'replace'
        else:
            # Merge semantics
            new_props = dict(existing_dict)
            for k, v in (properties or {}).items():
                new_props[k] = v

        # Serialize history list if needed (graph backends may expect string)
        if isinstance(new_props.get('history'), list):
            try:
                new_props['history'] = json.dumps(new_props['history'])
            except Exception:
                pass

        # Write back via graph add_node (upsert semantics)
        self._graph.add_node(item_id=item_id, properties=new_props, memory_type=new_props.get('memory_type', existing_memory_type))
        return new_props

    # ── CORE-EVO-LIVE-1: mutation event emission ────────────────────────────

    def _emit_mutation(
        self,
        operation: str,
        item_id: str,
        memory_type: str,
        properties: Optional[Dict[str, Any]] = None,
        neighbors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Emit a MutationEvent to the evolution queue (if wired)."""
        if self._evolution_queue is None:
            return
        try:
            from smartmemory.evolution.events import MutationEvent

            # Lazy-start the evolution worker on first mutation
            if self._smart_memory is not None:
                self._smart_memory._ensure_evolution_worker()
                # Don't enqueue if no worker started (evolution disabled) — prevents unbounded backlog
                if self._smart_memory._evolution_worker is None:
                    return

            event = MutationEvent(
                item_id=item_id,
                memory_type=memory_type,
                operation=operation,
                workspace_id="default",  # Lite mode is single-tenant
                properties=properties,
                neighbors=neighbors,
            )
            self._evolution_queue.put(event)
        except Exception as exc:
            logger.debug("CRUD._emit_mutation: failed to emit %s event: %s", operation, exc)

    def search(self, query: Any, **kwargs) -> List[MemoryItem]:
        """Search for items in the store by delegating to graph search."""
        try:
            # Delegate to graph search functionality
            results = self._graph.search(query, **kwargs)

            # Ensure results are MemoryItem objects
            memory_items = []
            for result in results:
                if isinstance(result, MemoryItem):
                    memory_items.append(result)
                else:
                    # Convert graph nodes to MemoryItem if needed
                    memory_items.append(self.normalize_item(result))

            return memory_items
        except Exception:
            # Return empty list on search failure to maintain API contract
            return []
