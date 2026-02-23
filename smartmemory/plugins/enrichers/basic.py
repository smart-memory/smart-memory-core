import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata


@dataclass
class BasicEnricherConfig(MemoryBaseModel):
    enable_entity_tags: bool = True
    enable_summary: bool = True


@dataclass
class BasicEnricherRequest(StageRequest):
    enable_entity_tags: bool = True
    enable_summary: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class BasicEnricher(EnricherPlugin):
    """
    Basic enrichment: tag with entities and generate summary.
    - Tags: Use entities from node_ids if present.
    - Summary: First sentence of item.content.
    Returns enrichment result dict.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="basic_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="Basic enrichment with entity tags and summary generation",
            plugin_type="enricher",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def __init__(self, config: Optional[BasicEnricherConfig] = None):
        self.config = config or BasicEnricherConfig()

    def enrich(self, item, node_ids=None) -> Dict[str, Any]:
        # Fail fast on typed config
        if not isinstance(self.config, BasicEnricherConfig):
            raise TypeError("BasicEnricher requires a typed config (BasicEnricherConfig)")
        memory_id = getattr(item, 'item_id', None)
        entities = (node_ids.get('semantic_entities') or [] if isinstance(node_ids, dict) else [])
        with trace_span("pipeline.enrich.basic_enricher", {"memory_id": memory_id, "entity_count": len(entities)}):
            content = getattr(item, 'content', str(item))
            result: Dict[str, Any] = {'new_items': []}

            # Summary
            if self.config.enable_summary:
                result['summary'] = content.split('.')[0] + '.' if '.' in content else content[:100]

            # Entity tags and temporal annotations if node_ids provided
            if isinstance(node_ids, dict) and self.config.enable_entity_tags:
                result['tags'] = list(entities)
                from datetime import datetime
                temporal = {}
                now = datetime.now()
                for entity in entities:
                    temporal[entity] = {
                        'valid_start': now,
                        'valid_end': None,
                        'transaction_time': now
                    }
                if temporal:
                    result['temporal'] = temporal

        return result
