"""
Factory-based memory creation utility.

Provides high-level functions for creating memory instances using the unified factory pattern.
This module serves as the main entry point for factory-based memory creation.
"""

import logging
from typing import Optional, Dict, Any

from smartmemory.configuration import MemoryConfig
from smartmemory.memory.base import MemoryBase
from smartmemory.memory.memory_factory import MemoryFactory, MemoryType
from smartmemory.stores.factory import StoreFactory, StoreType

logger = logging.getLogger(__name__)


def create_memory_system(config: Optional[Any] = None) -> Dict[str, MemoryBase]:
    """
    Create a complete memory system with all memory types using factory pattern.
    
    Args:
        config: Optional memory configuration. If None, loads default config.
        
    Returns:
        Dictionary mapping memory type names to memory instances
        
    Example:
        >>> memory_system = create_memory_system()
        >>> semantic = memory_system['semantic']
        >>> episodic = memory_system['episodic']
    """
    if config is None:
        try:
            config = MemoryConfig()
            logger.info("Loaded default memory configuration")
        except (FileNotFoundError, OSError) as e:
            logger.debug(f"Could not load default config: {e}")
            config = None
        except Exception as e:
            logger.warning(f"Could not load default config: {e}")
            config = None

    # Create all memory types using factory
    memory_instances = MemoryFactory.create_all_memory_types(config)

    # Convert to string keys for easier access
    memory_system = {}
    for memory_type, instance in memory_instances.items():
        memory_system[memory_type.value] = instance

    logger.info(f"Created complete memory system with {len(memory_system)} memory types")
    return memory_system


def create_smart_memory(config: Optional[Any] = None) -> MemoryBase:
    """
    Create a SmartMemory instance using factory pattern.
    
    Args:
        config: Optional memory configuration
        
    Returns:
        SmartMemory instance
    """
    if config is None:
        try:
            config = MemoryConfig()
        except (FileNotFoundError, OSError) as e:
            logger.debug(f"Could not load config for SmartMemory: {e}")
            config = None
        except Exception as e:
            logger.warning(f"Could not load config for SmartMemory: {e}")
            config = None

    try:
        # Import SmartMemory and register it if not already registered
        from smartmemory.smart_memory import SmartMemory

        # Check if SmartMemory is registered, if not register it
        smart_memory_type = None
        for mem_type in MemoryFactory.get_registered_types():
            if mem_type.value == "smart":
                smart_memory_type = mem_type
                break

        if smart_memory_type is None:
            # Create a new memory type for SmartMemory
            from enum import Enum
            class ExtendedMemoryType(Enum):
                SMART = "smart"

            # Register SmartMemory
            MemoryFactory.register_memory_type(ExtendedMemoryType.SMART, SmartMemory)
            smart_memory = MemoryFactory.create_memory(ExtendedMemoryType.SMART, config)
        else:
            smart_memory = MemoryFactory.create_memory(smart_memory_type, config)

        logger.info("Created SmartMemory instance using factory pattern")
        return smart_memory

    except Exception as e:
        logger.error(f"Failed to create SmartMemory via factory: {e}")
        # Fallback to direct instantiation
        from smartmemory.smart_memory import SmartMemory
        return SmartMemory(config=config)


def create_memory_by_type(memory_type: str, config: Optional[Any] = None) -> MemoryBase:
    """
    Create a specific memory type by name using factory pattern.
    
    Args:
        memory_type: Name of memory type ('semantic', 'episodic', 'procedural', 'working', 'zettel')
        config: Optional memory configuration
        
    Returns:
        Memory instance of the specified type
        
    Raises:
        ValueError: If memory type is not supported
    """
    if config is None:
        try:
            config = MemoryConfig()
        except (FileNotFoundError, OSError) as e:
            logger.debug(f"Could not load config: {e}")
            config = None
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            config = None

    # Map string names to MemoryType enum values
    type_mapping = {
        'semantic': MemoryType.SEMANTIC,
        'episodic': MemoryType.EPISODIC,
        'procedural': MemoryType.PROCEDURAL,
        'working': MemoryType.WORKING,
        'zettel': MemoryType.ZETTEL
    }

    if memory_type.lower() not in type_mapping:
        available_types = list(type_mapping.keys())
        raise ValueError(f"Unsupported memory type: {memory_type}. Available: {available_types}")

    enum_type = type_mapping[memory_type.lower()]
    memory_instance = MemoryFactory.create_memory(enum_type, config)

    logger.info(f"Created {memory_type} memory instance using factory pattern")
    return memory_instance


def create_store_system(config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create a complete store system with all store types using factory pattern.
    
    Args:
        config: Optional memory configuration
        
    Returns:
        Dictionary mapping store type names to store instances
    """
    if config is None:
        try:
            config = MemoryConfig()
        except (FileNotFoundError, OSError) as e:
            logger.debug(f"Could not load config: {e}")
            config = None
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            config = None

    # Create all graph stores using factory
    graph_stores = StoreFactory.create_all_graph_stores(config)

    # Also create vector and main graph store
    try:
        vector_store = StoreFactory.create_vector_store(config)
        graph_stores[StoreType.VECTOR] = vector_store
    except Exception as e:
        logger.warning(f"Could not create vector store: {e}")

    try:
        main_graph = StoreFactory.create_graph_store(config)
        graph_stores[StoreType.GRAPH] = main_graph
    except Exception as e:
        logger.warning(f"Could not create main graph store: {e}")

    # Convert to string keys
    store_system = {}
    for store_type, instance in graph_stores.items():
        store_system[store_type.value] = instance

    logger.info(f"Created complete store system with {len(store_system)} store types")
    return store_system


def validate_factory_system() -> Dict[str, bool]:
    """
    Validate that the factory system is properly configured and functional.
    
    Returns:
        Dictionary with validation results for different stages
    """
    results = {}

    # Validate memory factory registration
    try:
        registered_memory_types = MemoryFactory.get_registered_types()
        results['memory_factory_registered'] = len(registered_memory_types) >= 5
        results['memory_types_count'] = len(registered_memory_types)
    except Exception as e:
        logger.error(f"Memory factory validation failed: {e}")
        results['memory_factory_registered'] = False
        results['memory_types_count'] = 0

    # Validate store factory registration
    try:
        registered_store_types = StoreFactory.get_registered_types()
        results['store_factory_registered'] = len(registered_store_types) >= 6
        results['store_types_count'] = len(registered_store_types)
    except Exception as e:
        logger.error(f"Store factory validation failed: {e}")
        results['store_factory_registered'] = False
        results['store_types_count'] = 0

    # Test memory creation
    try:
        semantic = create_memory_by_type('semantic')
        results['memory_creation_works'] = semantic is not None
    except Exception as e:
        logger.error(f"Memory creation test failed: {e}")
        results['memory_creation_works'] = False

    # Test system creation
    try:
        memory_system = create_memory_system()
        results['system_creation_works'] = len(memory_system) > 0
        results['system_memory_count'] = len(memory_system)
    except Exception as e:
        logger.error(f"System creation test failed: {e}")
        results['system_creation_works'] = False
        results['system_memory_count'] = 0

    # Overall validation
    results['overall_valid'] = all([
        results.get('memory_factory_registered', False),
        results.get('store_factory_registered', False),
        results.get('memory_creation_works', False),
        results.get('system_creation_works', False)
    ])

    return results


# Convenience functions for common use cases
def get_semantic_memory(config: Optional[Any] = None) -> MemoryBase:
    """Get a semantic memory instance using factory pattern."""
    return create_memory_by_type('semantic', config)


def get_episodic_memory(config: Optional[Any] = None) -> MemoryBase:
    """Get an episodic memory instance using factory pattern."""
    return create_memory_by_type('episodic', config)


def get_procedural_memory(config: Optional[Any] = None) -> MemoryBase:
    """Get a procedural memory instance using factory pattern."""
    return create_memory_by_type('procedural', config)


def get_working_memory(config: Optional[Any] = None) -> MemoryBase:
    """Get a working memory instance using factory pattern."""
    return create_memory_by_type('working', config)


def get_zettel_memory(config: Optional[Any] = None) -> MemoryBase:
    """Get a zettel memory instance using factory pattern."""
    return create_memory_by_type('zettel', config)
