"""EnrichmentPipeline component for componentized memory ingestion pipeline.
Handles synchronous enrichment plugin execution with configurable plugin selection.
"""
import logging
from typing import Dict, Any, List

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import EnrichmentConfig
from smartmemory.memory.pipeline.state import LinkingState
from smartmemory.memory.pipeline.transactions.change_set import ChangeOp, ChangeSet
from smartmemory.memory.ingestion.utils import sanitize_relation_type
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.utils.pipeline_utils import create_error_result

logger = logging.getLogger(__name__)


class EnrichmentPipeline(PipelineComponent[EnrichmentConfig]):
    """
    Component responsible for running enrichment plugins synchronously.
    Supports configurable plugin selection and execution order.
    """

    def __init__(self, enrichment_instance=None, memory_instance=None):
        if enrichment_instance is None:
            raise ValueError("EnrichmentPipeline requires a valid enrichment_instance")
        if memory_instance is None:
            raise ValueError("EnrichmentPipeline requires a valid memory_instance")
        self.enrichment = enrichment_instance
        self.memory = memory_instance

    def _add_derived_items(self, context: Dict[str, Any], enrichment_result: Dict[str, Any], *, preview_mode: bool = False, proposed_ops: List[ChangeOp] | None = None) -> List[
        str]:
        """Add new items from enrichment to memory graph and link them.

        In preview mode, append ChangeOps instead of mutating stores.
        """
        derived_ids: List[str] = []

        if not enrichment_result or not enrichment_result.get('new_items'):
            return derived_ids

        original_id = context.get('item_id')
        if not original_id:
            return derived_ids

        try:
            for i, new_item in enumerate(enrichment_result['new_items']):
                # Convert to MemoryItem if needed
                if not hasattr(new_item, 'item_id'):
                    if isinstance(new_item, dict):
                        mem_item = MemoryItem(**new_item)
                    else:
                        mem_item = MemoryItem(content=str(new_item))
                else:
                    mem_item = new_item

                if preview_mode:
                    # Propose node add and canonical edge
                    derived_id = getattr(mem_item, 'item_id', None) or f"derived_{i}"
                    derived_ids.append(derived_id)
                    if proposed_ops is not None:
                        try:
                            proposed_ops.append(ChangeOp(op_type='add_node', args={
                                'node_id': derived_id,
                                'properties': {
                                    'type': 'memory_node',
                                    'content': getattr(mem_item, 'content', None),
                                    'memory_type': getattr(mem_item, 'memory_type', None),
                                    'provenance': 'enrichment'
                                }
                            }))
                            proposed_ops.append(ChangeOp(op_type='add_edge', args={
                                'source': derived_id,
                                'target': original_id,
                                'relation_type': sanitize_relation_type('CANONICAL'),
                                'properties': {'provenance': 'enrichment'}
                            }))
                        except Exception:
                            # keep preview robust
                            pass
                    # Skip relation handling for preview
                else:
                    # Mutating path
                    derived_id = self.memory.add(mem_item)
                    derived_ids.append(derived_id)
                    try:
                        self.memory._graph.add_edge(
                            source_id=derived_id,
                            target_id=original_id,
                            edge_type=sanitize_relation_type('CANONICAL'),
                            properties={'provenance': 'enrichment'},
                        )
                    except Exception:
                        pass
                    # Handle relations/triples if present (omitted here)
        except Exception as e:
            logger.warning(f"Failed to add derived items: {e}")

        return derived_ids

    def validate_config(self, config: EnrichmentConfig) -> bool:
        """Validate EnrichmentPipeline configuration using typed config"""
        try:
            names = getattr(config, 'enricher_names', None)
            if names is not None and not isinstance(names, list):
                return False
            order = getattr(config, 'execution_order', 'parallel')
            if order not in ['parallel', 'sequential', 'priority']:
                return False
            edi = getattr(config, 'enable_derived_items', True)
            if not isinstance(edi, bool):
                return False
            return True
        except Exception:
            return False

    def run(self, linking_state: LinkingState, config: EnrichmentConfig) -> ComponentResult:
        """
        Execute EnrichmentPipeline with given linking state and configuration.
        
        Args:
            linking_state: LinkingState from previous stage with linked entities
            config: Enrichment configuration dict with optional 'enricher_names'
        
        Returns:
            ComponentResult with enrichment results and metadata
        """
        try:
            if not linking_state or not linking_state.success:
                return create_error_result('enrichment_pipeline', ValueError('Invalid or failed linking state'))

            # Extract context from linking state
            context = linking_state.data.get('context', {})
            if not context:
                return ComponentResult(
                    success=False,
                    data={'error': 'No context available from linking state'},
                    metadata={'stage': 'enrichment_pipeline'}
                )

            # Add enricher names to context if specified
            enricher_names = getattr(config, 'enricher_names', None)
            if enricher_names:
                context['enricher_names'] = enricher_names

            # Run enrichment
            try:
                # Get per-enricher configs if provided
                enricher_configs = getattr(config, 'enricher_configs', None) or {}
                memory_id = context.get('item_id')
                _enricher_names = enricher_names or ['all']

                with trace_span("pipeline.enrich", {"memory_id": memory_id, "enricher_names": _enricher_names}):
                    if enricher_names:
                        # Run specific enrichers with their configs
                        enrichment_result = {}
                        for enricher_name in enricher_names:
                            partial_result = self.enrichment.enrich(
                                context,
                                enricher_names=[enricher_name],
                                enricher_configs=enricher_configs
                            )
                            enrichment_result.update(partial_result)
                    else:
                        # Run all enrichers by default
                        enrichment_result = self.enrichment.enrich(
                            context,
                            enricher_configs=enricher_configs
                        )

                enrichment_success = True
                enrichment_error = None

            except Exception as e:
                enrichment_success = False
                enrichment_error = str(e)
                enrichment_result = {}
                logger.warning(f"Enrichment failed: {e}")

            # Detect preview mode
            preview_mode = bool(getattr(config, 'preview', False))
            proposed_ops: List[ChangeOp] = []

            # Add derived items from enrichment to memory graph
            derived_ids = []
            if enrichment_success and enrichment_result:
                derived_ids = self._add_derived_items(context, enrichment_result, preview_mode=preview_mode, proposed_ops=proposed_ops)

            # In preview mode, also propose edges from enricher-produced relations/triples
            proposed_relation_edges = 0
            if preview_mode and enrichment_success and enrichment_result:
                def _append_edge(src: Any, rel_type: Any, tgt: Any, props: Dict[str, Any] | None = None):
                    nonlocal proposed_relation_edges
                    try:
                        proposed_ops.append(ChangeOp(op_type='add_edge', args={
                            'source': src,
                            'target': tgt,
                            'relation_type': sanitize_relation_type(rel_type) if isinstance(rel_type, str) else (rel_type or 'RELATED'),
                            'properties': props or {}
                        }))
                        proposed_relation_edges += 1
                    except Exception:
                        # keep preview robust
                        pass

                def _process_relations_payload(payload: Dict[str, Any]):
                    # relations: list of dicts or tuples
                    rels = (payload.get('relations') or []) if isinstance(payload, dict) else []
                    for rel in rels:
                        if isinstance(rel, dict):
                            # Handle dict format with flexible key names
                            src = rel.get('source') or rel.get('source_id') or rel.get('from')
                            tgt = rel.get('target') or rel.get('target_id') or rel.get('to')
                            rtype = rel.get('relation_type') or rel.get('type') or 'RELATED'
                            props = rel.get('properties') or rel.get('metadata') or {}
                            if src and tgt:
                                _append_edge(src, rtype, tgt, props)
                        elif isinstance(rel, (list, tuple)) and len(rel) == 3:
                            # Handle tuple format (source, relation_type, target)
                            src, rtype, tgt = rel
                            if src and tgt:
                                _append_edge(src, rtype, tgt, {})

                # top-level payload
                _process_relations_payload(enrichment_result)
                # nested plugin_* payloads
                if isinstance(enrichment_result, dict):
                    for k, v in enrichment_result.items():
                        if isinstance(v, dict) and str(k).startswith('plugin_'):
                            _process_relations_payload(v)

            # Extract enriched data
            enriched_data = {}
            plugin_results = {}

            if enrichment_result:
                # Separate enriched data from plugin-specific results
                for key, value in enrichment_result.items():
                    if key.startswith('plugin_'):
                        plugin_results[key] = value
                    else:
                        enriched_data[key] = value

            # Build enrichment metadata
            enrichment_metadata = {
                'enrichers_requested': enricher_names or ['all'],
                'enrichment_success': enrichment_success,
                'enrichment_error': enrichment_error,
                'derived_items_created': len(derived_ids),
                'execution_order': getattr(config, 'execution_order', 'parallel'),
                'plugins_executed': len(plugin_results)
            }
            if preview_mode:
                enrichment_metadata['proposed_relation_edges'] = proposed_relation_edges

            # Collect provenance candidates for grounding
            provenance_candidates = context.get('provenance_candidates', [])
            if enrichment_result and 'provenance_candidates' in enrichment_result:
                provenance_candidates.extend(enrichment_result['provenance_candidates'])

            result_data: Dict[str, Any] = {
                'enriched_data': enriched_data,
                'plugin_results': plugin_results,
                'enrichment_metadata': enrichment_metadata,
                'derived_ids': derived_ids,
                'provenance_candidates': provenance_candidates,
                'context': context
            }

            # Attach change_set when preview proposed ops exist
            if preview_mode and proposed_ops:
                cs = ChangeSet.new(stage='enrichment', plugin='EnrichmentPipeline', run_id=getattr(config, 'run_id', None))
                cs.ops.extend(proposed_ops)
                result_data['change_set'] = {
                    'change_set_id': cs.change_set_id,
                    'ops_count': len(cs.ops),
                }
                result_data['change_set_ops'] = [
                    {'op_type': op.op_type, 'args': op.args} for op in proposed_ops
                ]

            return ComponentResult(
                success=True,
                data=result_data,
                metadata={
                    'stage': 'enrichment_pipeline',
                    'plugins_count': len(plugin_results),
                    'derived_items': len(derived_ids)
                }
            )

        except Exception as e:
            return create_error_result('enrichment_pipeline', e)
