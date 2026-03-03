"""
Example: Creating a Custom Evolver Plugin

This example demonstrates how to create a custom evolver plugin that
transforms memories based on custom rules and conditions.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta, UTC
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


class ImportantMemoryEvolver(EvolverPlugin):
    """
    Custom evolver that promotes frequently accessed memories to 'important' status.
    
    This evolver tracks memory access patterns and promotes memories that
    are accessed frequently to a higher importance level.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery and registration."""
        return PluginMetadata(
            name="important_memory_evolver",
            version="1.0.0",
            author="Your Name",
            description="Promotes frequently accessed memories to important status",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["importance", "promotion", "access-patterns"]
        )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evolver with configuration.
        
        Args:
            config: Configuration dictionary with:
                - access_threshold: Minimum access count for promotion (default: 5)
                - time_window_days: Time window for counting accesses (default: 7)
        """
        super().__init__(config)
        self.access_threshold = config.get('access_threshold', 5) if config else 5
        self.time_window_days = config.get('time_window_days', 7) if config else 7
    
    def evolve(self, memory, logger=None) -> None:
        """
        Evolve memories by promoting important ones.
        
        Args:
            memory: The memory system to evolve
            logger: Optional logger for tracking evolution
        """
        if logger:
            logger.info(f"Running {self.metadata().name} evolver")
        
        # Get all memories (you'd typically filter by type or other criteria)
        try:
            # This is a simplified example - actual implementation would query the graph
            # For demonstration, we'll show the logic
            
            cutoff_date = datetime.now(UTC) - timedelta(days=self.time_window_days)
            
            # In a real implementation, you would:
            # 1. Query memories from the graph
            # 2. Check access counts in metadata
            # 3. Promote memories that meet the threshold
            
            if logger:
                logger.info(f"Checking memories accessed since {cutoff_date}")
                logger.info(f"Promotion threshold: {self.access_threshold} accesses")
            
            # Example logic (pseudo-code):
            # for item in memory.get_all():
            #     access_count = item.metadata.get('access_count', 0)
            #     if access_count >= self.access_threshold:
            #         item.metadata['importance'] = 'high'
            #         item.metadata['promoted_at'] = datetime.now(UTC).isoformat()
            #         memory.update(item)
            
            if logger:
                logger.info("Important memory evolution complete")
                
        except Exception as e:
            if logger:
                logger.error(f"Error in {self.metadata().name}: {e}")


class StaleMemoryEvolver(EvolverPlugin):
    """
    Custom evolver that archives or removes stale memories.
    
    Moves memories that haven't been accessed in a long time to an archive
    or marks them for deletion.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="stale_memory_evolver",
            version="1.0.0",
            author="Your Name",
            description="Archives or removes stale memories",
            plugin_type="evolver",
            tags=["cleanup", "archival", "maintenance"]
        )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration with:
                - stale_days: Days without access before considering stale (default: 90)
                - action: 'archive' or 'delete' (default: 'archive')
        """
        super().__init__(config)
        self.stale_days = config.get('stale_days', 90) if config else 90
        self.action = config.get('action', 'archive') if config else 'archive'
    
    def evolve(self, memory, logger=None) -> None:
        """Archive or delete stale memories."""
        if logger:
            logger.info(f"Running {self.metadata().name} evolver")
        
        cutoff_date = datetime.now(UTC) - timedelta(days=self.stale_days)
        
        try:
            # Example logic for stale memory handling
            if logger:
                logger.info(f"Checking for memories not accessed since {cutoff_date}")
                logger.info(f"Action: {self.action}")
            
            # In real implementation:
            # for item in memory.get_all():
            #     last_access = item.metadata.get('last_accessed')
            #     if last_access and last_access < cutoff_date:
            #         if self.action == 'archive':
            #             item.metadata['archived'] = True
            #             item.metadata['archived_at'] = datetime.now(UTC).isoformat()
            #             memory.update(item)
            #         elif self.action == 'delete':
            #             memory.delete(item.item_id)
            
            if logger:
                logger.info("Stale memory evolution complete")
                
        except Exception as e:
            if logger:
                logger.error(f"Error in {self.metadata().name}: {e}")


# Example usage
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create evolvers with custom config
    important_evolver = ImportantMemoryEvolver(config={
        'access_threshold': 3,
        'time_window_days': 14
    })
    
    stale_evolver = StaleMemoryEvolver(config={
        'stale_days': 60,
        'action': 'archive'
    })
    
    # Display metadata
    print("Important Memory Evolver:")
    print(f"  Name: {important_evolver.metadata().name}")
    print(f"  Description: {important_evolver.metadata().description}")
    print(f"  Config: threshold={important_evolver.access_threshold}, window={important_evolver.time_window_days} days")
    print()
    
    print("Stale Memory Evolver:")
    print(f"  Name: {stale_evolver.metadata().name}")
    print(f"  Description: {stale_evolver.metadata().description}")
    print(f"  Config: stale_days={stale_evolver.stale_days}, action={stale_evolver.action}")
    print()
    
    # In a real application, you would:
    # from smartmemory.smart_memory import SmartMemory
    # memory = SmartMemory()
    # important_evolver.evolve(memory, logger)
    # stale_evolver.evolve(memory, logger)
    
    print("Note: Run these evolvers with a SmartMemory instance to see them in action!")
