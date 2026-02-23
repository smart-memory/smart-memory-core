from typing import Any

from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.manager import get_plugin_manager


class Enrichment:
    """
    Handles memory enrichment logic using the plugin system.

    All enrichers are loaded from the PluginRegistry, which discovers them automatically
    from built-in plugins and external plugins via entry points.
    """

    def __init__(self, graph):
        self.graph = graph

        # Get plugin manager and registry
        plugin_manager = get_plugin_manager()
        self.plugin_registry = plugin_manager.registry

        # Store enricher classes for config-aware instantiation
        self._enricher_classes: dict[str, type] = {}
        for enricher_name in self.plugin_registry.list_plugins('enricher'):
            enricher_class = self.plugin_registry.get_enricher(enricher_name)
            if enricher_class:
                self._enricher_classes[enricher_name] = enricher_class

        # Build enricher registry with default (no-config) callables for backward compatibility
        self.enricher_registry: dict[str, Any] = {}
        for enricher_name, enricher_class in self._enricher_classes.items():
            self.enricher_registry[enricher_name] = self._make_enricher_callable(enricher_class)

        # Default pipeline: all enrichers, in registry order
        self._enricher_pipeline = list(self.enricher_registry.keys())


    def _make_enricher_callable(self, cls, config: dict[str, Any] | None = None):
        """Create a callable that instantiates an enricher with optional config."""
        def enricher_fn(item, node_ids=None):
            if config:
                # Try to instantiate with config
                try:
                    instance = cls(config)
                except TypeError:
                    # Enricher doesn't accept config dict, try converting to typed config
                    config_class = self._get_config_class(cls)
                    if config_class:
                        try:
                            typed_config = config_class(**config)
                            instance = cls(typed_config)
                        except Exception:
                            # Fall back to no config
                            instance = cls()
                    else:
                        instance = cls()
            else:
                instance = cls()
            return instance.enrich(item, node_ids)
        return enricher_fn

    def _get_config_class(self, enricher_class):
        """Try to find the config class for an enricher."""
        # Convention: EnricherName -> EnricherNameConfig
        class_name = enricher_class.__name__
        config_class_name = f"{class_name}Config"

        # Check the same module
        module = enricher_class.__module__
        try:
            import importlib
            mod = importlib.import_module(module)
            return getattr(mod, config_class_name, None)
        except Exception:
            return None

    def register_enricher(self, name, enricher_fn):
        """Register a new enricher by name."""
        self.enricher_registry[name] = enricher_fn

    def enrich(self, context, enricher_names=None, enricher_configs: dict[str, dict[str, Any]] | None = None):
        """
        Call all enrichers in the pipeline (in order). If enricher_names is None, use all enrichers from the registry.
        Merges results from all enrichers.

        Args:
            context: Dict with 'item' and 'node_ids' keys.
            enricher_names: Optional list of enricher names to run.
            enricher_configs: Optional dict mapping enricher names to their config dicts.
        """
        pipeline = enricher_names or self._enricher_pipeline
        enricher_configs = enricher_configs or {}
        result = {}

        memory_id = getattr(context.get("item"), "item_id", None) if isinstance(context, dict) else None

        for enricher_name in pipeline:
            # Get enricher callable - use config-aware version if config provided
            config = enricher_configs.get(enricher_name)
            if config and enricher_name in self._enricher_classes:
                enricher = self._make_enricher_callable(self._enricher_classes[enricher_name], config)
            else:
                enricher = self.enricher_registry.get(enricher_name)

            if enricher is None:
                raise ValueError(f"Enricher '{enricher_name}' not registered.")

            with trace_span(f"pipeline.enrich.{enricher_name}", {"memory_id": memory_id, "plugin": enricher_name}):
                enricher_result = enricher(context["item"], context["node_ids"])
            if enricher_result:
                # Merge 'properties' deeply so multiple enrichers can contribute
                if 'properties' in enricher_result:
                    props = enricher_result.get('properties') or {}
                    if props:
                        if 'properties' not in result or not isinstance(result.get('properties'), dict):
                            result['properties'] = {}
                        result['properties'].update(props)
                # Merge 'relations' as a concatenated list
                if 'relations' in enricher_result:
                    rels = enricher_result.get('relations') or []
                    if rels:
                        if 'relations' not in result or not isinstance(result.get('relations'), list):
                            result['relations'] = []
                        result['relations'].extend(rels)
                # Merge any other top-level keys via last-wins (legacy compatibility)
                for k, v in enricher_result.items():
                    if k in ('properties', 'relations'):
                        continue
                    result[k] = v
        return result
