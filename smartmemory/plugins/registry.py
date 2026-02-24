"""
Plugin registry for SmartMemory.

This module provides a central registry for all plugin types, enabling
dynamic plugin registration and retrieval.
"""

from typing import Dict, Type, Callable, List, Optional
import logging
import threading

from .base import PluginMetadata, PluginBase

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for all SmartMemory plugins.
    
    This class maintains separate registries for each plugin type and provides
    thread-safe registration and retrieval operations.
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_enricher("my_enricher", MyEnricher, metadata)
        >>> enricher_class = registry.get_enricher("my_enricher")
    """
    
    def __init__(self):
        """Initialize the plugin registry with empty registries for each type."""
        self._enrichers: Dict[str, Type[PluginBase]] = {}
        self._extractors: Dict[str, Callable] = {}
        self._evolvers: Dict[str, Type[PluginBase]] = {}
        self._embeddings: Dict[str, Type[PluginBase]] = {}
        self._grounders: Dict[str, Type[PluginBase]] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        
        # Thread lock for thread-safe operations
        self._lock = threading.RLock()
        
        logger.debug("PluginRegistry initialized")
    
    def register_enricher(self, name: str, enricher_class: Type[PluginBase], 
                         metadata: PluginMetadata) -> None:
        """
        Register an enricher plugin.
        
        Args:
            name: Unique name for the enricher
            enricher_class: The enricher class
            metadata: Plugin metadata
        
        Raises:
            ValueError: If name is empty or metadata type doesn't match
        """
        if not name:
            raise ValueError("Plugin name cannot be empty")
        
        if metadata.plugin_type != 'enricher':
            raise ValueError(f"Plugin type must be 'enricher', got '{metadata.plugin_type}'")
        
        with self._lock:
            if name in self._enrichers:
                logger.warning(f"Enricher '{name}' already registered, overwriting")
            
            self._enrichers[name] = enricher_class
            self._metadata[name] = metadata
            logger.info(f"Registered enricher: {name} v{metadata.version}")
    
    def register_extractor(self, name: str, extractor_class_or_factory: Callable,
                          metadata: PluginMetadata) -> None:
        """
        Register an extractor plugin.
        
        Args:
            name: Unique name for the extractor
            extractor_class_or_factory: Extractor class (preferred) or factory function (legacy)
            metadata: Plugin metadata
        
        Raises:
            ValueError: If name is empty or metadata type doesn't match
        """
        if not name:
            raise ValueError("Plugin name cannot be empty")
        
        if metadata.plugin_type != 'extractor':
            raise ValueError(f"Plugin type must be 'extractor', got '{metadata.plugin_type}'")
        
        with self._lock:
            if name in self._extractors:
                logger.warning(f"Extractor '{name}' already registered, overwriting")
            
            self._extractors[name] = extractor_class_or_factory
            self._metadata[name] = metadata
            logger.info(f"Registered extractor: {name} v{metadata.version}")
    
    def register_evolver(self, name: str, evolver_class: Type[PluginBase],
                        metadata: PluginMetadata) -> None:
        """
        Register an evolver plugin.
        
        Args:
            name: Unique name for the evolver
            evolver_class: The evolver class
            metadata: Plugin metadata
        
        Raises:
            ValueError: If name is empty or metadata type doesn't match
        """
        if not name:
            raise ValueError("Plugin name cannot be empty")
        
        if metadata.plugin_type != 'evolver':
            raise ValueError(f"Plugin type must be 'evolver', got '{metadata.plugin_type}'")
        
        with self._lock:
            if name in self._evolvers:
                logger.warning(f"Evolver '{name}' already registered, overwriting")
            
            self._evolvers[name] = evolver_class
            self._metadata[name] = metadata
            logger.info(f"Registered evolver: {name} v{metadata.version}")
    
    def register_embedding_provider(self, name: str, provider_class: Type[PluginBase],
                                   metadata: PluginMetadata) -> None:
        """
        Register an embedding provider plugin.
        
        Args:
            name: Unique name for the embedding provider
            provider_class: The embedding provider class
            metadata: Plugin metadata
        
        Raises:
            ValueError: If name is empty or metadata type doesn't match
        """
        if not name:
            raise ValueError("Plugin name cannot be empty")
        
        if metadata.plugin_type != 'embedding':
            raise ValueError(f"Plugin type must be 'embedding', got '{metadata.plugin_type}'")
        
        with self._lock:
            if name in self._embeddings:
                logger.warning(f"Embedding provider '{name}' already registered, overwriting")
            
            self._embeddings[name] = provider_class
            self._metadata[name] = metadata
            logger.info(f"Registered embedding provider: {name} v{metadata.version}")
    
    def register_grounder(self, name: str, grounder_class: Type[PluginBase],
                         metadata: PluginMetadata) -> None:
        """
        Register a grounder plugin.
        
        Args:
            name: Unique name for the grounder
            grounder_class: The grounder class
            metadata: Plugin metadata
        
        Raises:
            ValueError: If name is empty or metadata type doesn't match
        """
        if not name:
            raise ValueError("Plugin name cannot be empty")
        
        if metadata.plugin_type != 'grounder':
            raise ValueError(f"Plugin type must be 'grounder', got '{metadata.plugin_type}'")
        
        with self._lock:
            if name in self._grounders:
                logger.warning(f"Grounder '{name}' already registered, overwriting")
            
            self._grounders[name] = grounder_class
            self._metadata[name] = metadata
            logger.info(f"Registered grounder: {name} v{metadata.version}")
    
    def get_enricher(self, name: str) -> Optional[Type[PluginBase]]:
        """
        Get an enricher plugin by name.
        
        Args:
            name: Name of the enricher
        
        Returns:
            Optional[Type[PluginBase]]: The enricher class or None if not found
        """
        with self._lock:
            return self._enrichers.get(name)
    
    def get_extractor(self, name: str) -> Optional[Callable]:
        """
        Get an extractor plugin by name.
        
        Args:
            name: Name of the extractor
        
        Returns:
            Optional[Callable]: The extractor factory or None if not found
        """
        with self._lock:
            return self._extractors.get(name)
    
    def get_evolver(self, name: str) -> Optional[Type[PluginBase]]:
        """
        Get an evolver plugin by name.
        
        Args:
            name: Name of the evolver
        
        Returns:
            Optional[Type[PluginBase]]: The evolver class or None if not found
        """
        with self._lock:
            return self._evolvers.get(name)
    
    def get_embedding_provider(self, name: str) -> Optional[Type[PluginBase]]:
        """
        Get an embedding provider plugin by name.
        
        Args:
            name: Name of the embedding provider
        
        Returns:
            Optional[Type[PluginBase]]: The embedding provider class or None if not found
        """
        with self._lock:
            return self._embeddings.get(name)
    
    def get_grounder(self, name: str) -> Optional[Type[PluginBase]]:
        """
        Get a grounder plugin by name.
        
        Args:
            name: Name of the grounder
        
        Returns:
            Optional[Type[PluginBase]]: The grounder class or None if not found
        """
        with self._lock:
            return self._grounders.get(name)
    
    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata by name.
        
        Args:
            name: Name of the plugin
        
        Returns:
            Optional[PluginMetadata]: The plugin metadata or None if not found
        """
        with self._lock:
            return self._metadata.get(name)
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """
        List all registered plugins, optionally filtered by type.
        
        Args:
            plugin_type: Optional filter by type ('enricher', 'extractor', 'evolver', 'embedding', 'grounder')
        
        Returns:
            List[str]: Sorted list of plugin names
        """
        with self._lock:
            if plugin_type == 'enricher':
                return sorted(self._enrichers.keys())
            elif plugin_type == 'extractor':
                return sorted(self._extractors.keys())
            elif plugin_type == 'evolver':
                return sorted(self._evolvers.keys())
            elif plugin_type == 'embedding':
                return sorted(self._embeddings.keys())
            elif plugin_type == 'grounder':
                return sorted(self._grounders.keys())
            elif plugin_type is None:
                # Return all plugins
                all_plugins = set()
                all_plugins.update(self._enrichers.keys())
                all_plugins.update(self._extractors.keys())
                all_plugins.update(self._evolvers.keys())
                all_plugins.update(self._embeddings.keys())
                all_plugins.update(self._grounders.keys())
                return sorted(all_plugins)
            else:
                raise ValueError(
                    f"Invalid plugin_type '{plugin_type}'. "
                    f"Must be one of: enricher, extractor, evolver, embedding, grounder, or None"
                )
    
    def has_plugin(self, name: str, plugin_type: Optional[str] = None) -> bool:
        """
        Check if a plugin is registered.
        
        Args:
            name: Name of the plugin
            plugin_type: Optional type to check specifically
        
        Returns:
            bool: True if plugin is registered, False otherwise
        """
        with self._lock:
            if plugin_type == 'enricher':
                return name in self._enrichers
            elif plugin_type == 'extractor':
                return name in self._extractors
            elif plugin_type == 'evolver':
                return name in self._evolvers
            elif plugin_type == 'embedding':
                return name in self._embeddings
            elif plugin_type == 'grounder':
                return name in self._grounders
            elif plugin_type is None:
                return (name in self._enrichers or 
                       name in self._extractors or 
                       name in self._evolvers or 
                       name in self._embeddings or
                       name in self._grounders)
            else:
                raise ValueError(f"Invalid plugin_type '{plugin_type}'")
    
    def unregister(self, name: str, plugin_type: Optional[str] = None) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Name of the plugin to unregister
            plugin_type: Optional type to unregister from specifically
        
        Returns:
            bool: True if plugin was unregistered, False if not found
        """
        with self._lock:
            unregistered = False
            
            if plugin_type is None or plugin_type == 'enricher':
                if name in self._enrichers:
                    del self._enrichers[name]
                    unregistered = True
            
            if plugin_type is None or plugin_type == 'extractor':
                if name in self._extractors:
                    del self._extractors[name]
                    unregistered = True
            
            if plugin_type is None or plugin_type == 'evolver':
                if name in self._evolvers:
                    del self._evolvers[name]
                    unregistered = True
            
            if plugin_type is None or plugin_type == 'embedding':
                if name in self._embeddings:
                    del self._embeddings[name]
                    unregistered = True
            
            if plugin_type is None or plugin_type == 'grounder':
                if name in self._grounders:
                    del self._grounders[name]
                    unregistered = True
            
            if unregistered and name in self._metadata:
                del self._metadata[name]
                logger.info(f"Unregistered plugin: {name}")
            
            return unregistered
    
    def clear(self, plugin_type: Optional[str] = None) -> None:
        """
        Clear all plugins or plugins of a specific type.
        
        Args:
            plugin_type: Optional type to clear ('enricher', 'extractor', 'evolver', 'embedding', 'grounder')
                        If None, clears all plugins
        """
        with self._lock:
            if plugin_type is None:
                self._enrichers.clear()
                self._extractors.clear()
                self._evolvers.clear()
                self._embeddings.clear()
                self._grounders.clear()
                self._metadata.clear()
                logger.info("Cleared all plugins from registry")
            elif plugin_type == 'enricher':
                for name in list(self._enrichers.keys()):
                    del self._enrichers[name]
                    if name in self._metadata:
                        del self._metadata[name]
                logger.info("Cleared all enricher plugins")
            elif plugin_type == 'extractor':
                for name in list(self._extractors.keys()):
                    del self._extractors[name]
                    if name in self._metadata:
                        del self._metadata[name]
                logger.info("Cleared all extractor plugins")
            elif plugin_type == 'evolver':
                for name in list(self._evolvers.keys()):
                    del self._evolvers[name]
                    if name in self._metadata:
                        del self._metadata[name]
                logger.info("Cleared all evolver plugins")
            elif plugin_type == 'embedding':
                for name in list(self._embeddings.keys()):
                    del self._embeddings[name]
                    if name in self._metadata:
                        del self._metadata[name]
                logger.info("Cleared all embedding provider plugins")
            elif plugin_type == 'grounder':
                for name in list(self._grounders.keys()):
                    del self._grounders[name]
                    if name in self._metadata:
                        del self._metadata[name]
                logger.info("Cleared all grounder plugins")
            else:
                raise ValueError(f"Invalid plugin_type '{plugin_type}'")
    
    def get_all_metadata(self, plugin_type: Optional[str] = None) -> Dict[str, PluginMetadata]:
        """
        Get metadata for all plugins or plugins of a specific type.
        
        Args:
            plugin_type: Optional type filter
        
        Returns:
            Dict[str, PluginMetadata]: Dictionary mapping plugin names to metadata
        """
        with self._lock:
            if plugin_type is None:
                return dict(self._metadata)
            
            plugin_names = self.list_plugins(plugin_type)
            return {name: self._metadata[name] for name in plugin_names if name in self._metadata}
    
    def count(self, plugin_type: Optional[str] = None) -> int:
        """
        Count registered plugins.
        
        Args:
            plugin_type: Optional type to count
        
        Returns:
            int: Number of registered plugins
        """
        return len(self.list_plugins(plugin_type))
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        with self._lock:
            return (f"PluginRegistry("
                   f"enrichers={len(self._enrichers)}, "
                   f"extractors={len(self._extractors)}, "
                   f"evolvers={len(self._evolvers)}, "
                   f"embeddings={len(self._embeddings)}, "
                   f"grounders={len(self._grounders)})")


# Global registry instance
_global_registry: Optional[PluginRegistry] = None
_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    This function implements the singleton pattern to ensure only one
    registry exists throughout the application lifecycle.
    
    Returns:
        PluginRegistry: The global plugin registry
    """
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            # Double-check locking pattern
            if _global_registry is None:
                _global_registry = PluginRegistry()
                logger.debug("Created global plugin registry")
    
    return _global_registry


def reset_plugin_registry() -> None:
    """
    Reset the global plugin registry.
    
    This is primarily useful for testing. In production, the registry
    should persist for the lifetime of the application.
    """
    global _global_registry
    
    with _registry_lock:
        if _global_registry is not None:
            _global_registry.clear()
            _global_registry = None
            logger.debug("Reset global plugin registry")
