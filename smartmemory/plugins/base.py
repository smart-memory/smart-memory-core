"""
Base classes and interfaces for SmartMemory plugin system.

This module provides the foundation for creating external plugins that can be
dynamically discovered and loaded into SmartMemory.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Optional, List, Any

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """
    Metadata for a SmartMemory plugin.
    
    This information is used for plugin discovery, validation, and compatibility checking.
    """
    name: str
    """Unique identifier for the plugin (lowercase, no spaces)"""
    
    version: str
    """Semantic version string (e.g., '1.0.0')"""
    
    author: str
    """Plugin author name or organization"""
    
    description: str
    """Brief description of what the plugin does"""
    
    plugin_type: str
    """Type of plugin: 'enricher', 'extractor', 'evolver', or 'embedding'"""
    
    dependencies: List[str] = field(default_factory=list)
    """List of Python package dependencies (e.g., ['numpy>=1.20.0', 'requests'])"""
    
    min_smartmemory_version: str = "0.1.0"
    """Minimum compatible SmartMemory version"""
    
    max_smartmemory_version: Optional[str] = None
    """Maximum compatible SmartMemory version (None = no limit)"""
    
    license: str = "MIT"
    """License identifier (SPDX format preferred)"""
    
    homepage: Optional[str] = None
    """Project homepage or repository URL"""
    
    tags: List[str] = field(default_factory=list)
    """Optional tags for categorization and search"""
    
    security_profile: str = "standard"
    """Security profile: 'trusted', 'standard', 'restricted', or 'untrusted' (default: 'standard')"""
    
    requires_network: bool = False
    """Whether this plugin requires network access"""
    
    requires_llm: bool = False
    """Whether this plugin requires LLM API access"""
    
    requires_file_access: bool = False
    """Whether this plugin requires file system access"""
    
    def __post_init__(self):
        """Validate metadata after initialization"""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        
        if not self.version:
            raise ValueError("Plugin version cannot be empty")
        
        if self.plugin_type not in ['enricher', 'extractor', 'evolver', 'embedding', 'grounder']:
            raise ValueError(
                f"Invalid plugin_type '{self.plugin_type}'. "
                f"Must be one of: enricher, extractor, evolver, embedding, grounder"
            )
        
        # Normalize name (lowercase, replace spaces with underscores)
        self.name = self.name.lower().replace(' ', '_').replace('-', '_')


class PluginBase(ABC):
    """
    Abstract base class for all SmartMemory plugins.
    
    All plugins must inherit from this class and implement the required methods.
    This ensures a consistent interface for plugin discovery and loading.
    
    Example:
        >>> class MyEnricher(PluginBase):
        ...     @classmethod
        ...     def metadata(cls) -> PluginMetadata:
        ...         return PluginMetadata(
        ...             name="my_enricher",
        ...             version="1.0.0",
        ...             author="Your Name",
        ...             description="My custom enricher",
        ...             plugin_type="enricher"
        ...         )
        ...     
        ...     def enrich(self, item, node_ids=None):
        ...         return {"custom_field": "value"}
    """
    
    @classmethod
    @abstractmethod
    def metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata.
        
        This method must be implemented by all plugins to provide information
        about the plugin for discovery and compatibility checking.
        
        Returns:
            PluginMetadata: Plugin metadata instance
        """
        pass
    
    @classmethod
    def config_schema(cls) -> Optional[Type]:
        """
        Return the configuration schema class for this plugin.
        
        If the plugin accepts configuration, return the dataclass or Pydantic model
        that defines the configuration schema. Return None if no configuration is needed.
        
        Returns:
            Optional[Type]: Configuration schema class or None
        """
        return None
    
    @classmethod
    def validate_environment(cls) -> bool:
        """
        Validate that the plugin can run in the current environment.
        
        This method checks for required dependencies, system resources, or other
        prerequisites. It's called during plugin discovery to determine if the
        plugin can be loaded.
        
        Returns:
            bool: True if the environment is valid, False otherwise
        """
        return True
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """
        Get the list of Python package dependencies for this plugin.
        
        Returns:
            List[str]: List of package requirements (e.g., ['numpy>=1.20.0'])
        """
        metadata = cls.metadata()
        return metadata.dependencies or []


class EnricherPlugin(PluginBase):
    """
    Base class for enricher plugins.
    
    Enrichers add metadata, properties, or relationships to memory items
    during the enrichment stage of the ingestion pipeline.
    
    Example:
        >>> class SentimentEnricher(EnricherPlugin):
        ...     @classmethod
        ...     def metadata(cls) -> PluginMetadata:
        ...         return PluginMetadata(
        ...             name="sentiment",
        ...             version="1.0.0",
        ...             author="SmartMemory Team",
        ...             description="Adds sentiment analysis",
        ...             plugin_type="enricher"
        ...         )
        ...     
        ...     def enrich(self, item, node_ids=None):
        ...         # Analyze sentiment
        ...         sentiment = self._analyze_sentiment(item.content)
        ...         return {"properties": {"sentiment": sentiment}}
    """
    
    @abstractmethod
    def enrich(self, item, node_ids=None) -> dict:
        """
        Enrich a memory item with additional information.
        
        Args:
            item: The memory item to enrich
            node_ids: Optional dict of node IDs from extraction stage
        
        Returns:
            dict: Enrichment results with keys like 'properties', 'relations', 'tags'
        """
        pass


class EvolverPlugin(PluginBase):
    """
    Base class for evolver plugins.

    Evolvers transform memory over time, moving items between memory types,
    pruning old memories, or consolidating related memories.

    CORE-EVO-LIVE-1: Subclasses may declare ``TRIGGERS`` (a set of
    ``(memory_type, operation)`` tuples) and implement ``evolve_incremental()``
    for event-driven incremental evolution. The default ``TRIGGERS`` is empty
    (batch-only evolver), and the default ``evolve_incremental()`` raises
    ``NotImplementedError``.
    """

    # CORE-EVO-LIVE-1: Mutation types this evolver cares about.
    # Set of (memory_type, operation) tuples, e.g. {("working", "add")}.
    # Empty set = batch-only evolver (runs during idle catch-up).
    TRIGGERS: set = set()

    def __init__(self, config: Optional[Any] = None):
        self.config = config or {}

    @abstractmethod
    def evolve(self, memory, logger=None):
        """Apply batch evolution logic to the memory system."""
        pass

    def evolve_incremental(self, ctx: Any) -> list:
        """Apply incremental evolution in response to a single mutation event.

        Args:
            ctx: EvolutionContext with event, graph, backend, workspace_id.

        Returns:
            List of EvolutionAction objects to be flushed by WriteBatcher.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement evolve_incremental(). "
            "Declare TRIGGERS and override this method for event-driven evolution."
        )


class ExtractorPlugin(PluginBase):
    """
    Base class for extractor plugins.
    
    Extractors identify entities and relationships from text content.
    
    Example:
        >>> class MyExtractor(ExtractorPlugin):
        ...     @classmethod
        ...     def metadata(cls):
        ...         return PluginMetadata(
        ...             name="my_extractor",
        ...             version="1.0.0",
        ...             author="Your Name",
        ...             description="My custom extractor",
        ...             plugin_type="extractor"
        ...         )
        ...     
        ...     def extract(self, text):
        ...         # Extract entities and relations
        ...         entities = [{"name": "John", "type": "person"}]
        ...         relations = [("John", "works_at", "Google")]
        ...         return {"entities": entities, "relations": relations}
    """
    
    @abstractmethod
    def extract(self, text: str) -> dict:
        """
        Extract entities and relationships from text.
        
        Args:
            text: The text to extract from
        
        Returns:
            dict: Dictionary with 'entities' and 'relations' keys
        """
        pass


class GrounderPlugin(PluginBase):
    """
    Base class for grounder plugins.
    
    Grounders link memory items to external knowledge sources
    (e.g., Wikipedia, databases, APIs) for provenance and validation.
    
    Example:
        >>> class WikipediaGrounder(GrounderPlugin):
        ...     @classmethod
        ...     def metadata(cls):
        ...         return PluginMetadata(
        ...             name="wikipedia_grounder",
        ...             version="1.0.0",
        ...             author="SmartMemory Team",
        ...             description="Grounds entities to Wikipedia",
        ...             plugin_type="grounder"
        ...         )
        ...     
        ...     def ground(self, item, entities, graph):
        ...         # Ground entities to Wikipedia
        ...         provenance_candidates = []
        ...         for entity in entities:
        ...             wiki_id = self._lookup_wikipedia(entity)
        ...             if wiki_id:
        ...                 provenance_candidates.append(wiki_id)
        ...         return provenance_candidates
    """
    
    @abstractmethod
    def ground(self, item, entities: list, graph) -> list:
        """
        Ground memory item to external knowledge sources.
        
        Args:
            item: The memory item to ground
            entities: List of entities to ground
            graph: The graph backend for creating edges
        
        Returns:
            list: List of provenance candidate node IDs
        """
        pass


class EmbeddingProviderPlugin(PluginBase):
    """
    Base class for embedding provider plugins.
    
    Embedding providers generate vector embeddings from text for similarity search.
    
    Example:
        >>> class MyEmbeddingProvider(EmbeddingProviderPlugin):
        ...     @classmethod
        ...     def metadata(cls) -> PluginMetadata:
        ...         return PluginMetadata(
        ...             name="my_embeddings",
        ...             version="1.0.0",
        ...             author="Your Name",
        ...             description="Custom embedding provider",
        ...             plugin_type="embedding"
        ...         )
        ...     
        ...     def embed(self, text: str) -> np.ndarray:
        ...         # Generate embedding
        ...         return np.array([...])
    """
    
    @abstractmethod
    def embed(self, text: str):
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
        
        Returns:
            numpy.ndarray: The embedding vector
        """
        pass


def validate_plugin_class(plugin_class: Type[PluginBase]) -> bool:
    """
    Validate that a class is a proper plugin.
    
    Args:
        plugin_class: The class to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Must be a subclass of PluginBase
        if not issubclass(plugin_class, PluginBase):
            logger.warning(f"{plugin_class.__name__} is not a subclass of PluginBase")
            return False
        
        # Must not be an abstract base class itself
        if plugin_class in [PluginBase, EnricherPlugin, EvolverPlugin, 
                           ExtractorPlugin, EmbeddingProviderPlugin]:
            return False
        
        # Must implement metadata() method
        try:
            metadata = plugin_class.metadata()
            if not isinstance(metadata, PluginMetadata):
                logger.warning(f"{plugin_class.__name__}.metadata() must return PluginMetadata")
                return False
        except NotImplementedError:
            logger.warning(f"{plugin_class.__name__} does not implement metadata()")
            return False
        
        # Must pass environment validation
        if not plugin_class.validate_environment():
            logger.info(f"{plugin_class.__name__} failed environment validation")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating plugin class {plugin_class}: {e}")
        return False


def check_version_compatibility(plugin_version: str, min_version: str, 
                                max_version: Optional[str] = None) -> bool:
    """
    Check if a plugin version is compatible with SmartMemory version requirements.
    
    Args:
        plugin_version: The SmartMemory version to check
        min_version: Minimum required version
        max_version: Maximum allowed version (None = no limit)
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        from packaging import version
        
        plugin_ver = version.parse(plugin_version)
        min_ver = version.parse(min_version)
        
        if plugin_ver < min_ver:
            return False
        
        if max_version is not None:
            max_ver = version.parse(max_version)
            if plugin_ver > max_ver:
                return False
        
        return True
    
    except Exception as e:
        logger.warning(f"Error checking version compatibility: {e}")
        # If we can't check, assume compatible
        return True
