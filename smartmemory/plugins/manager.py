"""
Plugin Manager for SmartMemory

Handles plugin discovery, loading, and lifecycle management.
"""

import logging
import sys
import importlib.metadata
from pathlib import Path
from typing import Optional, List, Set

from smartmemory.plugins.registry import PluginRegistry, get_plugin_registry
from smartmemory.plugins.base import PluginBase, PluginMetadata, validate_plugin_class, check_version_compatibility

logger = logging.getLogger(__name__)

# SmartMemory version for compatibility checking
SMARTMEMORY_VERSION = "0.1.0"  # TODO: Import from version module


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    The PluginManager is responsible for:
    - Discovering plugins from multiple sources
    - Validating plugin compatibility
    - Loading plugins into the registry
    - Handling loading errors gracefully
    
    Example:
        >>> manager = PluginManager()
        >>> manager.discover_plugins()
        >>> print(f"Loaded {len(manager.loaded_plugins)} plugins")
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the plugin manager.
        
        Args:
            registry: Optional plugin registry to use. If None, uses global registry.
        """
        self.registry = registry or get_plugin_registry()
        self._loaded_plugins: Set[str] = set()
        logger.debug("PluginManager initialized")
    
    @property
    def loaded_plugins(self) -> Set[str]:
        """Get the set of loaded plugin names."""
        return self._loaded_plugins.copy()
    
    def discover_plugins(self, 
                        include_builtin: bool = True,
                        include_entry_points: bool = True,
                        plugin_dirs: Optional[List[Path]] = None) -> None:
        """
        Discover and load all available plugins.
        
        This method orchestrates plugin discovery from multiple sources:
        1. Built-in plugins from smartmemory.plugins
        2. External plugins via setuptools entry points
        3. Plugins from specified directories
        
        Args:
            include_builtin: Whether to load built-in plugins
            include_entry_points: Whether to discover plugins via entry points
            plugin_dirs: Optional list of directories to scan for plugins
        """
        logger.info("Starting plugin discovery...")
        initial_count = len(self._loaded_plugins)
        
        # 1. Load built-in plugins
        if include_builtin:
            self._load_builtin_plugins()
        
        # 2. Discover via entry points
        if include_entry_points:
            self._discover_entry_point_plugins()
        
        # 3. Scan plugin directories
        if plugin_dirs:
            for plugin_dir in plugin_dirs:
                self._scan_plugin_directory(plugin_dir)
        
        new_plugins = len(self._loaded_plugins) - initial_count
        logger.info(f"Plugin discovery complete. Loaded {new_plugins} new plugins "
                   f"(total: {len(self._loaded_plugins)})")
    
    def _load_builtin_plugins(self) -> None:
        """Load built-in plugins from smartmemory.plugins."""
        logger.info("Loading built-in plugins...")
        
        # Load built-in enrichers
        try:
            from smartmemory.plugins.enrichers.basic import BasicEnricher
            from smartmemory.plugins.enrichers.sentiment import SentimentEnricher
            from smartmemory.plugins.enrichers.skills_tools import ExtractSkillsToolsEnricher
            from smartmemory.plugins.enrichers.temporal import TemporalEnricher
            from smartmemory.plugins.enrichers.topic import TopicEnricher

            enrichers = [
                BasicEnricher,
                SentimentEnricher,
                TemporalEnricher,
                ExtractSkillsToolsEnricher,
                TopicEnricher,
            ]
            
            for enricher_class in enrichers:
                try:
                    if not validate_plugin_class(enricher_class):
                        logger.warning(f"Skipping invalid enricher: {enricher_class.__name__}")
                        continue
                    
                    metadata = enricher_class.metadata()
                    self.registry.register_enricher(
                        metadata.name,
                        enricher_class,
                        metadata
                    )
                    self._loaded_plugins.add(metadata.name)
                except Exception as e:
                    logger.error(f"Failed to load built-in enricher {enricher_class.__name__}: {e}")
        except ImportError as e:
            logger.warning(f"Could not import built-in enrichers: {e}")
        
        # Load built-in extractors (all class-based now)
        try:
            from smartmemory.plugins.extractors.llm import LLMExtractor
            from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor
            from smartmemory.plugins.extractors.conversation_aware_llm import ConversationAwareLLMExtractor

            # Primary extractors
            class_based_extractors = [
                LLMExtractor,
                LLMSingleExtractor,
                ConversationAwareLLMExtractor,
            ]
            
            try:
                from smartmemory.plugins.extractors.spacy import SpacyExtractor
                class_based_extractors.append(SpacyExtractor)
            except ImportError:
                pass
            
            for extractor_class in class_based_extractors:
                try:
                    if not validate_plugin_class(extractor_class):
                        logger.warning(f"Skipping invalid extractor: {extractor_class.__name__}")
                        continue
                    
                    metadata = extractor_class.metadata()
                    
                    # Register the class directly (not a factory)
                    self.registry.register_extractor(metadata.name, extractor_class, metadata)
                    self._loaded_plugins.add(metadata.name)
                except Exception as e:
                    logger.error(f"Failed to load built-in extractor {extractor_class.__name__}: {e}")
        except ImportError as e:
            logger.warning(f"Could not import built-in extractors: {e}")
        
        # Load built-in evolvers
        try:
            from smartmemory.plugins.evolvers.enhanced.exponential_decay import ExponentialDecayEvolver
            from smartmemory.plugins.evolvers.enhanced.interference_based_consolidation import (
                InterferenceBasedConsolidationEvolver,
            )
            from smartmemory.plugins.evolvers.enhanced.retrieval_based_strengthening import (
                RetrievalBasedStrengtheningEvolver,
            )
            from smartmemory.plugins.evolvers.episodic_decay import EpisodicDecayEvolver
            from smartmemory.plugins.evolvers.episodic_to_semantic import EpisodicToSemanticEvolver
            from smartmemory.plugins.evolvers.episodic_to_zettel import EpisodicToZettelEvolver
            from smartmemory.plugins.evolvers.semantic_decay import SemanticDecayEvolver
            from smartmemory.plugins.evolvers.working_to_episodic import WorkingToEpisodicEvolver
            from smartmemory.plugins.evolvers.working_to_procedural import WorkingToProceduralEvolver
            from smartmemory.plugins.evolvers.zettel_prune import ZettelPruneEvolver

            evolvers = [
                WorkingToEpisodicEvolver,
                WorkingToProceduralEvolver,
                EpisodicToSemanticEvolver,
                EpisodicDecayEvolver,
                SemanticDecayEvolver,
                EpisodicToZettelEvolver,
                ZettelPruneEvolver,
                ExponentialDecayEvolver,
                InterferenceBasedConsolidationEvolver,
                RetrievalBasedStrengtheningEvolver,
            ]
            
            for evolver_class in evolvers:
                try:
                    if not validate_plugin_class(evolver_class):
                        logger.warning(f"Skipping invalid evolver: {evolver_class.__name__}")
                        continue
                    
                    metadata = evolver_class.metadata()
                    self.registry.register_evolver(
                        metadata.name,
                        evolver_class,
                        metadata
                    )
                    self._loaded_plugins.add(metadata.name)
                except Exception as e:
                    logger.error(f"Failed to load built-in evolver {evolver_class.__name__}: {e}")
        except ImportError as e:
            logger.warning(f"Could not import built-in evolvers: {e}")
        
        # Load built-in grounders
        try:
            from smartmemory.plugins.grounders import WikipediaGrounder
            
            grounders = [WikipediaGrounder]
            
            for grounder_class in grounders:
                try:
                    if not validate_plugin_class(grounder_class):
                        logger.warning(f"Skipping invalid grounder: {grounder_class.__name__}")
                        continue
                    
                    metadata = grounder_class.metadata()
                    self.registry.register_grounder(
                        metadata.name,
                        grounder_class,
                        metadata
                    )
                    self._loaded_plugins.add(metadata.name)
                except Exception as e:
                    logger.error(f"Failed to load built-in grounder {grounder_class.__name__}: {e}")
        except ImportError as e:
            logger.warning(f"Could not import built-in grounders: {e}")
    
    def _discover_entry_point_plugins(self) -> None:
        """Discover plugins via setuptools entry points."""
        logger.info("Discovering plugins via entry points...")
        
        try:
            import importlib.metadata
            
            # Get all entry points for smartmemory plugins
            entry_points = importlib.metadata.entry_points()
            
            # Handle different Python versions (3.9 vs 3.10+)
            if hasattr(entry_points, 'select'):
                # Python 3.10+
                enricher_eps = entry_points.select(group='smartmemory.plugins.enrichers')
                extractor_eps = entry_points.select(group='smartmemory.plugins.extractors')
                evolver_eps = entry_points.select(group='smartmemory.plugins.evolvers')
                embedding_eps = entry_points.select(group='smartmemory.plugins.embeddings')
                grounder_eps = entry_points.select(group='smartmemory.plugins.grounders')
            else:
                # Python 3.9
                enricher_eps = entry_points.get('smartmemory.plugins.enrichers', [])
                extractor_eps = entry_points.get('smartmemory.plugins.extractors', [])
                evolver_eps = entry_points.get('smartmemory.plugins.evolvers', [])
                embedding_eps = entry_points.get('smartmemory.plugins.embeddings', [])
                grounder_eps = entry_points.get('smartmemory.plugins.grounders', [])
            
            # Load enrichers
            for ep in enricher_eps:
                self._load_entry_point_enricher(ep)
            
            # Load extractors
            for ep in extractor_eps:
                self._load_entry_point_extractor(ep)
            
            # Load evolvers
            for ep in evolver_eps:
                self._load_entry_point_evolver(ep)
            
            # Load embedding providers
            for ep in embedding_eps:
                self._load_entry_point_embedding(ep)
            
            # Load grounders
            for ep in grounder_eps:
                self._load_entry_point_grounder(ep)
        
        except ImportError:
            logger.warning("importlib.metadata not available, skipping entry point discovery")
        except Exception as e:
            logger.error(f"Error discovering entry point plugins: {e}")
    
    def _load_entry_point_enricher(self, entry_point) -> None:
        """Load an enricher from an entry point."""
        try:
            enricher_class = entry_point.load()
            
            if not validate_plugin_class(enricher_class):
                logger.warning(f"Entry point {entry_point.name} is not a valid plugin")
                return
            
            metadata = enricher_class.metadata()
            
            # Check version compatibility
            if not check_version_compatibility(
                SMARTMEMORY_VERSION,
                metadata.min_smartmemory_version,
                metadata.max_smartmemory_version
            ):
                logger.warning(
                    f"Plugin {entry_point.name} is not compatible with "
                    f"SmartMemory {SMARTMEMORY_VERSION}"
                )
                return
            
            # Validate environment
            if not enricher_class.validate_environment():
                logger.warning(
                    f"Plugin {entry_point.name} failed environment validation, skipping"
                )
                return
            
            self.registry.register_enricher(entry_point.name, enricher_class, metadata)
            self._loaded_plugins.add(entry_point.name)
            logger.info(f"Loaded external enricher: {entry_point.name}")
        
        except Exception as e:
            logger.error(f"Failed to load enricher plugin {entry_point.name}: {e}")
    
    def _load_entry_point_extractor(self, entry_point) -> None:
        """Load an extractor from an entry point."""
        try:
            extractor_factory = entry_point.load()
            
            # Extractors should provide metadata via a special attribute
            if hasattr(extractor_factory, 'metadata'):
                metadata = extractor_factory.metadata
            else:
                logger.warning(f"Extractor {entry_point.name} missing metadata, using defaults")
                metadata = PluginMetadata(
                    name=entry_point.name,
                    version="unknown",
                    author="unknown",
                    description=f"External extractor: {entry_point.name}",
                    plugin_type="extractor"
                )
            
            # Check version compatibility
            if not check_version_compatibility(
                SMARTMEMORY_VERSION,
                metadata.min_smartmemory_version,
                metadata.max_smartmemory_version
            ):
                logger.warning(
                    f"Plugin {entry_point.name} is not compatible with "
                    f"SmartMemory {SMARTMEMORY_VERSION}"
                )
                return
            
            self.registry.register_extractor(entry_point.name, extractor_factory, metadata)
            self._loaded_plugins.add(entry_point.name)
            logger.info(f"Loaded external extractor: {entry_point.name}")
        
        except Exception as e:
            logger.error(f"Failed to load extractor plugin {entry_point.name}: {e}")
    
    def _load_entry_point_evolver(self, entry_point) -> None:
        """Load an evolver from an entry point."""
        try:
            evolver_class = entry_point.load()
            
            if not validate_plugin_class(evolver_class):
                logger.warning(f"Entry point {entry_point.name} is not a valid plugin")
                return
            
            metadata = evolver_class.metadata()
            
            # Check version compatibility
            if not check_version_compatibility(
                SMARTMEMORY_VERSION,
                metadata.min_smartmemory_version,
                metadata.max_smartmemory_version
            ):
                logger.warning(
                    f"Plugin {entry_point.name} is not compatible with "
                    f"SmartMemory {SMARTMEMORY_VERSION}"
                )
                return
            
            # Validate environment
            if not evolver_class.validate_environment():
                logger.warning(
                    f"Plugin {entry_point.name} failed environment validation, skipping"
                )
                return
            
            self.registry.register_evolver(entry_point.name, evolver_class, metadata)
            self._loaded_plugins.add(entry_point.name)
            logger.info(f"Loaded external evolver: {entry_point.name}")
        
        except Exception as e:
            logger.error(f"Failed to load evolver plugin {entry_point.name}: {e}")
    
    def _load_entry_point_embedding(self, entry_point) -> None:
        """Load an embedding provider from an entry point."""
        try:
            provider_class = entry_point.load()
            
            if not validate_plugin_class(provider_class):
                logger.warning(f"Entry point {entry_point.name} is not a valid plugin")
                return
            
            metadata = provider_class.metadata()
            
            # Check version compatibility
            if not check_version_compatibility(
                SMARTMEMORY_VERSION,
                metadata.min_smartmemory_version,
                metadata.max_smartmemory_version
            ):
                logger.warning(
                    f"Plugin {entry_point.name} is not compatible with "
                    f"SmartMemory {SMARTMEMORY_VERSION}"
                )
                return
            
            # Validate environment
            if not provider_class.validate_environment():
                logger.warning(
                    f"Plugin {entry_point.name} failed environment validation, skipping"
                )
                return
            
            self.registry.register_embedding_provider(entry_point.name, provider_class, metadata)
            self._loaded_plugins.add(entry_point.name)
            logger.info(f"Loaded external embedding provider: {entry_point.name}")
        
        except Exception as e:
            logger.error(f"Failed to load embedding provider plugin {entry_point.name}: {e}")
    
    def _load_entry_point_grounder(self, entry_point) -> None:
        """Load a grounder from an entry point."""
        try:
            grounder_class = entry_point.load()
            
            if not validate_plugin_class(grounder_class):
                logger.warning(f"Entry point {entry_point.name} is not a valid plugin")
                return
            
            metadata = grounder_class.metadata()
            
            # Check version compatibility
            if not check_version_compatibility(
                SMARTMEMORY_VERSION,
                metadata.min_smartmemory_version,
                metadata.max_smartmemory_version
            ):
                logger.warning(
                    f"Plugin {entry_point.name} is not compatible with "
                    f"SmartMemory {SMARTMEMORY_VERSION}"
                )
                return
            
            # Validate environment
            if not grounder_class.validate_environment():
                logger.warning(
                    f"Plugin {entry_point.name} failed environment validation, skipping"
                )
                return
            
            self.registry.register_grounder(entry_point.name, grounder_class, metadata)
            self._loaded_plugins.add(entry_point.name)
            logger.info(f"Loaded external grounder: {entry_point.name}")
        
        except Exception as e:
            logger.error(f"Failed to load grounder plugin {entry_point.name}: {e}")
    
    def _scan_plugin_directory(self, plugin_dir: Path) -> None:
        """
        Scan a directory for plugin modules.
        
        Args:
            plugin_dir: Directory to scan for plugins
        """
        logger.info(f"Scanning plugin directory: {plugin_dir}")
        
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return
        
        # Look for Python files or packages
        for item in plugin_dir.iterdir():
            if item.suffix == '.py' and item.stem != '__init__':
                self._load_plugin_module(item)
            elif item.is_dir() and (item / '__init__.py').exists():
                self._load_plugin_module(item / '__init__.py')
    
    def _load_plugin_module(self, module_path: Path) -> None:
        """
        Load a plugin from a module file.
        
        Args:
            module_path: Path to the plugin module
        """
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location("plugin_module", module_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {module_path}")
                return
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            # Look for plugin classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, PluginBase):
                    if not validate_plugin_class(attr):
                        continue
                    
                    try:
                        metadata = attr.metadata()
                        
                        # Check version compatibility
                        if not check_version_compatibility(
                            SMARTMEMORY_VERSION,
                            metadata.min_smartmemory_version,
                            metadata.max_smartmemory_version
                        ):
                            logger.warning(
                                f"Plugin {metadata.name} is not compatible with "
                                f"SmartMemory {SMARTMEMORY_VERSION}"
                            )
                            continue
                        
                        if metadata.plugin_type == 'enricher':
                            self.registry.register_enricher(metadata.name, attr, metadata)
                        elif metadata.plugin_type == 'evolver':
                            self.registry.register_evolver(metadata.name, attr, metadata)
                        elif metadata.plugin_type == 'embedding':
                            self.registry.register_embedding_provider(metadata.name, attr, metadata)
                        
                        self._loaded_plugins.add(metadata.name)
                        logger.info(f"Loaded plugin from file: {metadata.name}")
                    except Exception as e:
                        logger.error(f"Failed to register plugin {attr_name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load plugin module {module_path}: {e}")
    
    def reload_plugin(self, name: str) -> bool:
        """
        Reload a specific plugin (useful for development).
        
        Args:
            name: Name of the plugin to reload
        
        Returns:
            bool: True if reloaded successfully, False otherwise
        
        Note:
            This is primarily for development. Hot-reloading in production
            should be done carefully.
        """
        # TODO: Implement plugin reloading
        logger.warning("Plugin reloading not yet implemented")
        return False
    
    def unload_plugin(self, name: str) -> bool:
        """
        Unload a specific plugin.
        
        Args:
            name: Name of the plugin to unload
        
        Returns:
            bool: True if unloaded successfully, False otherwise
        """
        if name in self._loaded_plugins:
            self.registry.unregister(name)
            self._loaded_plugins.remove(name)
            logger.info(f"Unloaded plugin: {name}")
            return True
        return False


# Global plugin manager instance
_global_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.
    
    This function implements the singleton pattern. On first call,
    it creates a PluginManager and discovers all plugins.
    
    Returns:
        PluginManager: The global plugin manager
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = PluginManager()
        _global_manager.discover_plugins()
        logger.debug("Created and initialized global plugin manager")
    
    return _global_manager


def reset_plugin_manager() -> None:
    """
    Reset the global plugin manager.
    
    This is primarily useful for testing.
    """
    global _global_manager
    _global_manager = None
    logger.debug("Reset global plugin manager")
