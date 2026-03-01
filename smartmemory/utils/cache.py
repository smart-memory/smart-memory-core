"""
Redis Cache Manager for Agentic Memory

Provides distributed caching for high-performance memory operations.
Focuses on the highest-impact bottlenecks: embeddings, search results,
entity extraction, and similarity calculations.
"""

import hashlib
import json
import logging
import os
import pickle
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Union

from smartmemory.utils import get_config

logger = logging.getLogger(__name__)


class RedisCache:
    """
    High-performance Redis cache for agentic memory operations.

    Provides caching for:
    - Embedding calculations (highest impact)
    - Search results
    - Entity extraction results
    - Similarity scores
    - Graph query results
    """

    def __init__(self, redis_url: Optional[str] = None, prefix: str = "smartmemory"):
        """Initialize Redis cache with connection and configuration."""
        try:
            import redis as _redis
        except ImportError:
            raise ImportError(
                "redis is required for caching. Install it with: pip install smartmemory-core[server]"
            ) from None

        self.prefix = prefix

        # Get Redis configuration - fail fast if missing
        config = get_config()
        redis_config = config.cache.redis

        # Connect to Redis
        if redis_url:
            self.redis = _redis.from_url(redis_url)
        else:
            host = redis_config.host
            port = redis_config.port
            db = redis_config.get("db", 0)
            password = redis_config.get("password", None)

            self.redis = _redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            "embedding": config.cache.get("embedding_ttl", 86400),  # 24 hours
            "search": config.cache.get("search_ttl", 900),  # 15 minutes
            "entity_extraction": config.cache.get("extraction_ttl", 3600),  # 1 hour
            "similarity": config.cache.get("similarity_ttl", 1800),  # 30 minutes
            "graph_query": config.cache.get("query_ttl", 600),  # 10 minutes
        }

        # Performance tracking
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}

        # Test connection
        try:
            self.redis.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _make_key(self, cache_type: str, identifier: str) -> str:
        """Create a namespaced cache key."""
        return f"{self.prefix}:{cache_type}:{identifier}"

    def _hash_content(self, content: Union[str, Dict, List]) -> str:
        """Create a consistent hash for content."""
        if isinstance(content, (dict, list)):
            content = json.dumps(content, sort_keys=True)
        elif not isinstance(content, str):
            content = str(content)

        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def get(self, cache_type: str, identifier: str) -> Optional[Any]:
        """Get cached value by type and identifier."""
        try:
            key = self._make_key(cache_type, identifier)
            value = self.redis.get(key)

            if value is not None:
                self.stats["hits"] += 1
                return pickle.loads(value)
            else:
                self.stats["misses"] += 1
                return None

        except Exception as e:
            self.stats["errors"] += 1
            logger.warning(f"Redis cache get error: {e}")
            return None

    def set(self, cache_type: str, identifier: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value with optional TTL override."""
        try:
            key = self._make_key(cache_type, identifier)
            ttl = ttl or self.ttl_settings.get(cache_type, 3600)

            serialized = pickle.dumps(value)
            result = self.redis.setex(key, ttl, serialized)

            if result:
                self.stats["sets"] += 1
                return True
            return False

        except Exception as e:
            self.stats["errors"] += 1
            logger.warning(f"Redis cache set error: {e}")
            return False

    def delete(self, cache_type: str, identifier: str) -> bool:
        """Delete cached value."""
        try:
            key = self._make_key(cache_type, identifier)
            result = self.redis.delete(key)
            return result > 0
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning(f"Redis cache delete error: {e}")
            return False

    def clear_type(self, cache_type: str) -> int:
        """Clear all cached values of a specific type."""
        try:
            pattern = self._make_key(cache_type, "*")
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning(f"Redis cache clear error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0.0

        try:
            info = self.redis.info("memory")
            memory_usage = info.get("used_memory_human", "unknown")
        except Exception:
            memory_usage = "unknown"

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "errors": self.stats["errors"],
            "hit_rate": hit_rate,
            "memory_usage": memory_usage,
            "ttl_settings": self.ttl_settings,
        }

    # High-level caching methods for specific use cases

    def get_embedding(self, content: str) -> Optional[List[float]]:
        """Get cached embedding for content."""
        content_hash = self._hash_content(content)
        return self.get("embedding", content_hash)

    def set_embedding(self, content: str, embedding: List[float]) -> bool:
        """Cache embedding for content."""
        content_hash = self._hash_content(content)
        return self.set("embedding", content_hash, embedding)

    def get_search_results(self, query: str, top_k: int, memory_type: Optional[str] = None) -> Optional[List]:
        """Get cached search results."""
        search_key = f"{query}:{top_k}:{memory_type or 'all'}"
        search_hash = self._hash_content(search_key)
        return self.get("search", search_hash)

    def set_search_results(self, query: str, top_k: int, results: List, memory_type: Optional[str] = None) -> bool:
        """Cache search results."""
        search_key = f"{query}:{top_k}:{memory_type or 'all'}"
        search_hash = self._hash_content(search_key)
        return self.set("search", search_hash, results)

    def get_entity_extraction(self, content: str) -> Optional[Dict]:
        """Get cached entity extraction results."""
        content_hash = self._hash_content(content)
        return self.get("entity_extraction", content_hash)

    def set_entity_extraction(self, content: str, extraction_result: Dict) -> bool:
        """Cache entity extraction results."""
        content_hash = self._hash_content(content)
        return self.set("entity_extraction", content_hash, extraction_result)

    def get_similarity(self, item1_id: str, item2_id: str) -> Optional[float]:
        """Get cached similarity score."""
        # Ensure consistent ordering for cache key
        ids = sorted([item1_id, item2_id])
        similarity_key = f"{ids[0]}:{ids[1]}"
        return self.get("similarity", similarity_key)

    def set_similarity(self, item1_id: str, item2_id: str, score: float) -> bool:
        """Cache similarity score."""
        # Ensure consistent ordering for cache key
        ids = sorted([item1_id, item2_id])
        similarity_key = f"{ids[0]}:{ids[1]}"
        return self.set("similarity", similarity_key, score)


class NoOpCache:
    """No-op cache implementation for zero-infra environments (e.g. SmartMemory Lite).

    Uses __getattr__ to silently absorb any method call and return None.
    """

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


# Global cache instance
_global_cache: Optional[RedisCache] = None

# Optional override: when set, get_cache() returns this object regardless of env vars.
# Used by SmartMemory(cache=...) constructor parameter to inject a custom backend.
_CACHE_OVERRIDE: Optional[Any] = None

# CORE-DI-1: per-call override via ContextVar (set by SmartMemory._di_context())
_cache_ctx: ContextVar[Optional[Any]] = ContextVar("_cache_ctx", default=None)


def set_cache_override(cache: Optional[Any]) -> None:
    """Override the cache returned by get_cache().

    Pass a cache instance (e.g. NoOpCache()) to force all get_cache() calls to
    return it. Pass None to clear the override and restore env-var-based resolution.
    """
    global _CACHE_OVERRIDE, _global_cache
    _CACHE_OVERRIDE = cache
    # Clear the global cache so it will be re-evaluated on the next get_cache() call
    # after the override is removed.
    _global_cache = None


def get_cache() -> RedisCache:
    """Get the global Redis cache instance."""
    # CORE-DI-1: per-instance override takes priority over all process globals
    ctx = _cache_ctx.get()
    if ctx is not None:
        return ctx  # type: ignore[return-value]
    global _global_cache
    if _CACHE_OVERRIDE is not None:
        return _CACHE_OVERRIDE  # type: ignore[return-value]
    if os.environ.get("SMARTMEMORY_CACHE_DISABLED", "").lower() in ("true", "1", "yes"):
        return NoOpCache()  # type: ignore[return-value]
    if _global_cache is None:
        _global_cache = RedisCache()
    return _global_cache


def clear_cache() -> None:
    """Clear the global cache instance (for testing)."""
    global _global_cache
    _global_cache = None
