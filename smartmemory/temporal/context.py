"""
Time travel context manager for SmartMemory.

Provides a convenient way to execute queries in a temporal context.
"""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class TemporalContext:
    """
    Context manager for time-travel queries.
    
    All queries executed within this context will use the specified
    time point as their reference.
    
    Example:
        with memory.time_travel("2024-09-01"):
            # All queries in this block use Sept 1st as reference
            results = memory.search("Python")
            user = memory.get("user123")
    """
    
    def __init__(self, memory_system, time_point: str):
        """
        Initialize temporal context.
        
        Args:
            memory_system: SmartMemory instance
            time_point: Time to travel to (ISO format)
        """
        self.memory = memory_system
        self.time_point = time_point
        self.original_time = None
    
    def __enter__(self):
        """Enter temporal context."""
        # Save current time context
        self.original_time = getattr(self.memory, '_temporal_context', None)
        
        # Set new time context
        self.memory._temporal_context = self.time_point
        
        logger.debug(f"Entering temporal context: {self.time_point}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit temporal context."""
        # Restore original time context
        self.memory._temporal_context = self.original_time
        
        logger.debug(f"Exiting temporal context, restored to: {self.original_time}")
        
        # Don't suppress exceptions
        return False


@contextmanager
def time_travel(memory_system, to: str):
    """
    Context manager for time-travel queries.
    
    Args:
        memory_system: SmartMemory instance
        to: Time point to travel to (ISO format)
        
    Yields:
        TemporalContext instance
        
    Example:
        with time_travel(memory, "2024-09-01"):
            # All queries use Sept 1st as reference
            results = memory.search("Python")
            
        # Back to current time
        results = memory.search("Python")
    """
    ctx = TemporalContext(memory_system, to)
    with ctx:
        yield ctx
