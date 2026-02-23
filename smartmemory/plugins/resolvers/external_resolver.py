from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span


class ExternalResolver:
    def __init__(self):
        pass  # enricher_registry was never used

    def _payload(self, node, results):
        try:
            ref_count = 0
            if node and getattr(node, 'metadata', None):
                refs = node.metadata.get('external_refs')
                if isinstance(refs, list):
                    ref_count = len(refs)
                elif node.metadata.get('external_ref'):
                    ref_count = 1
            resolved_count = len(results) if isinstance(results, list) else (0 if results is None else 1)
            return {"references": ref_count, "resolved_count": resolved_count}
        except Exception:
            return {}

    def resolve_external(self, node: MemoryItem):
        """
        Resolve linked resources from other registered memory backends (hybrid memory).
        If node.metadata contains 'external_refs' (list of dicts with 'ref' and 'type'), resolve all.
        For backward compatibility, also supports single 'external_ref' and 'external_type'.
        Returns a list of resolved MemoryItems (empty if none resolved).
        """
        with trace_span("resolver.resolve_external", {}):
            return self._resolve_external_impl(node)

    def _resolve_external_impl(self, node: MemoryItem):
        results = []
        if node and getattr(node, 'metadata', None):
            refs = node.metadata.get('external_refs')
            if isinstance(refs, list):
                for ref_entry in refs:
                    ext = ref_entry.get('ref')
                    ext_type = ref_entry.get('type')
                    if ext and ext_type:
                        backend = None  # No memory type registry; implement if needed
                        if backend and hasattr(backend, "get"):
                            resolved = backend.get(ext)
                            if resolved:
                                results.append(resolved)
            else:
                # Backward compatibility: single external_ref/external_type
                ext = node.metadata.get("external_ref")
                ext_type = node.metadata.get("external_type")
                if ext and ext_type:
                    backend = None  # No memory type registry; implement if needed
                    if backend and hasattr(backend, "get"):
                        resolved = backend.get(ext)
                        if resolved:
                            results.append(resolved)
        return results if results else None
