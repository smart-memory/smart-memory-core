"""
Centralized observability module for ingestion pipeline.

This module consolidates all event emission and observability logic from the ingestion flow,
reducing boilerplate and providing consistent event schemas across the pipeline.
"""

import time
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional

from smartmemory.observability.tracing import trace_span


class IngestionObserver:
    """
    Centralized observability for the ingestion pipeline.

    Handles all event emission, performance tracking, and system monitoring
    with consistent schemas and reduced boilerplate.
    """

    def __init__(self):
        self._session_id = str(uuid.uuid4())
        self._stage_timers = {}

    def emit_event(self, event_type: str, data: Dict[str, Any] = None, component: str = "ingestion"):
        """
        Emit an observer event as a one-shot span.

        Args:
            event_type: Type of event (e.g., 'ingestion_start', 'extraction_complete')
            data: Event data dictionary
            component: Component name for event categorization
        """
        attrs = data.copy() if data else {}
        attrs["session_id"] = self._session_id
        with trace_span(f"observer.{event_type}", attrs):
            pass

    @contextmanager
    def track_stage(self, stage_name: str, item_id: Optional[str] = None, **metadata):
        """
        Track a pipeline stage as a traced span.

        Args:
            stage_name: Name of the pipeline stage
            item_id: Optional item ID for tracking
            **metadata: Additional metadata to include in events

        Usage:
            with observer.track_stage('extraction', item_id='123'):
                # extraction logic here
                pass
        """
        attrs = {}
        if item_id:
            attrs["memory_id"] = item_id
        attrs.update(metadata)
        with trace_span(f"pipeline.{stage_name}", attrs):
            start = time.perf_counter()
            yield
            self._stage_timers[stage_name] = time.perf_counter() - start

    def emit_ingestion_start(self, item_id: str, content_length: int, extractor: str, adapter: str):
        """Emit ingestion start event."""
        self.emit_event(
            "ingestion_start",
            {"memory_id": item_id, "content_length": content_length, "extractor": extractor, "adapter": adapter},
        )

    def emit_extraction_results(
        self, item_id: str, entities_count: int, relations_count: int, extractor: str, duration_ms: float = None
    ):
        """Emit extraction results event."""
        data = {
            "memory_id": item_id,
            "entities_count": entities_count,
            "relations_count": relations_count,
            "extractor": extractor,
        }
        if duration_ms is not None:
            data["duration_ms"] = duration_ms

        self.emit_event("extraction_complete", data)

    def emit_ingestion_complete(
        self,
        item_id: str,
        entities_extracted: int,
        relations_extracted: int,
        total_duration_ms: float,
        extractor: str,
        adapter: str,
    ):
        """Emit comprehensive ingestion completion event."""
        self.emit_event(
            "ingestion_complete",
            {
                "memory_id": item_id,
                "entities_extracted": entities_extracted,
                "relations_extracted": relations_extracted,
                "total_duration_ms": total_duration_ms,
                "extractor": extractor,
                "adapter": adapter,
                "timestamp": time.time(),
            },
        )

    def emit_entity_events(self, entity_name: str, normalized_name: str, node_id: str = None, reused: bool = False):
        """Emit entity creation or reuse events."""
        base_data = {"entity_name": entity_name, "normalized_name": normalized_name}

        if reused:
            self.emit_event("entity_reused", {**base_data, "existing_node_id": node_id})
        else:
            if node_id:
                self.emit_event("entity_created", {**base_data, "node_id": node_id})
            else:
                self.emit_event("entity_creation_start", base_data)

    def emit_edge_events(
        self, subject: str, predicate: str, object_node: str, edge_id: str = None, created: bool = True
    ):
        """Emit edge creation events."""
        base_data = {"subject": subject, "predicate": predicate, "object": object_node}

        if created and edge_id:
            self.emit_event("edge_created", {**base_data, "edge_id": edge_id})
        else:
            self.emit_event("edge_creation_start", base_data)

    def emit_performance_metrics(self, context: Dict[str, Any], total_duration_ms: float):
        """Emit performance metrics for observability."""
        try:
            entities_count = len(context.get("entities", []))
            relations_count = len(context.get("relations", []))

            performance_data = {
                "ingestion_duration_ms": total_duration_ms,
                "entities_per_second": entities_count / (total_duration_ms / 1000) if total_duration_ms > 0 else 0,
                "relations_per_second": relations_count / (total_duration_ms / 1000) if total_duration_ms > 0 else 0,
                "entities_processed": entities_count,
                "relations_processed": relations_count,
                "timestamp": time.time(),
            }

            self.emit_event("performance_metrics", performance_data)

            # Also emit summary event for backward compatibility
            summary_data = {
                "ingestion_duration_ms": total_duration_ms,
                "entities_processed": entities_count,
                "relations_processed": relations_count,
                "timestamp": performance_data["timestamp"],
            }
            self.emit_event("performance_summary", summary_data)

        except Exception:
            # Silent fallback - don't break ingestion for performance monitoring failures
            pass

    def emit_graph_statistics(self, memory):
        """Emit graph statistics for observability."""
        try:
            # Try to get graph statistics from SmartGraph
            if hasattr(memory, "_graph") and memory._graph:
                graph = memory._graph

                try:
                    # Try to query graph for node and edge counts
                    node_count = 0
                    edge_count = 0

                    # Attempt to get counts from graph backend
                    if hasattr(graph, "get_all_nodes"):
                        nodes = graph.get_all_nodes()
                        node_count = len(nodes) if nodes else 0

                    if hasattr(graph, "get_all_edges"):
                        edges = graph.get_all_edges()
                        edge_count = len(edges) if edges else 0

                    graph_data = {
                        "total_nodes": node_count,
                        "total_edges": edge_count,
                        "graph_density": edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0,
                        "timestamp": time.time(),
                    }

                    self.emit_event("graph_statistics", graph_data)
                    # Emit v1 schema event name for backward compatibility
                    self.emit_event("graph_stats_update", graph_data)

                except Exception:
                    # If direct graph queries fail, emit fallback stats
                    fallback_data = {
                        "total_nodes": "unknown",
                        "total_edges": "unknown",
                        "graph_density": 0,
                        "timestamp": time.time(),
                        "note": "Graph statistics unavailable - using fallback",
                    }
                    self.emit_event("graph_statistics", fallback_data)
                    self.emit_event("graph_stats_update", fallback_data)

        except Exception:
            # Silent fallback - don't break ingestion for graph statistics failures
            pass

    def emit_system_health_metrics(self):
        """
        Emit system health metrics (currently disabled).

        System health metrics are disabled because:
        - UI gets health data from /api/system/health endpoint, not events
        - No consumers benefit from per-ingestion health metrics
        - Creates noise in event stream without value

        If system health monitoring is needed, it should be:
        1. Periodic (not per-operation)
        2. From a dedicated health service
        3. Via dedicated endpoints, not event stream
        """
        pass

    def emit_error(self, item_id: str, error: str, error_type: str, stage: str = "ingestion"):
        """Emit error event during ingestion."""
        self.emit_event(
            "ingestion_error",
            {"memory_id": item_id, "error": error, "error_type": error_type, "stage": stage, "timestamp": time.time()},
        )

    def emit_edge_creation_start(self, subject: str, predicate: str, object_node: str):
        """Emit edge creation start event."""
        self.emit_event("edge_creation_start", {"subject": subject, "predicate": predicate, "object": object_node})

    def emit_edge_created(self, subject: str, predicate: str, object_node: str, edge_id: str):
        """Emit edge creation complete event."""
        self.emit_event(
            "edge_created", {"subject": subject, "predicate": predicate, "object": object_node, "edge_id": edge_id}
        )

    def emit_entity_reused(self, entity_name: str, normalized_name: str, existing_node_id: str):
        """Emit entity reuse event."""
        self.emit_event(
            "entity_reused",
            {"entity_name": entity_name, "normalized_name": normalized_name, "existing_node_id": existing_node_id},
        )

    def emit_entity_creation_start(self, entity_name: str, normalized_name: str):
        """Emit entity creation start event."""
        self.emit_event("entity_creation_start", {"entity_name": entity_name, "normalized_name": normalized_name})

    def emit_entity_created(self, entity_name: str, normalized_name: str, node_id: str):
        """Emit entity creation complete event."""
        self.emit_event(
            "entity_created", {"entity_name": entity_name, "normalized_name": normalized_name, "node_id": node_id}
        )
