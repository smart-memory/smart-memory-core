"""
SmartMemory: Multi-layered AI memory system with graph databases, vector stores, and
intelligent processing pipelines for context-aware AI applications. Provides semantic,
episodic, procedural, and working memory types with advanced relationship modeling and
storage and retrieval for AI applications.
"""

from smartmemory.__version__ import __version__, __version_info__

__author__ = "SmartMemory Team"

__all__ = [
    "SmartMemory",
    "MemoryItem",
    "__version__",
    "__version_info__",
]


def __getattr__(name: str):
    """Lazy-load heavy top-level exports so submodule imports don't pull the
    full ML stack (torch, numpy, etc.) unless SmartMemory itself is used."""
    if name == "SmartMemory":
        from smartmemory.smart_memory import SmartMemory
        return SmartMemory
    if name == "MemoryItem":
        from smartmemory.models.memory_item import MemoryItem
        return MemoryItem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
