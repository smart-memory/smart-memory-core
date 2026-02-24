"""Tests for the set_cache_override() mechanism in smartmemory.utils.cache."""
import os


def test_set_cache_override_returns_override():
    """get_cache() returns the override object when one is set."""
    from smartmemory.utils.cache import get_cache, set_cache_override, NoOpCache

    override = NoOpCache()
    try:
        set_cache_override(override)
        result = get_cache()
        assert result is override, "get_cache() must return the installed override"
    finally:
        set_cache_override(None)


def test_set_cache_override_none_restores_default():
    """Clearing the override restores env-var-based cache resolution."""
    from smartmemory.utils.cache import get_cache, set_cache_override, NoOpCache

    override = NoOpCache()
    set_cache_override(override)
    set_cache_override(None)

    # With override cleared, the env-var path takes over.
    # Force the disabled path so we don't need a live Redis.
    old = os.environ.get("SMARTMEMORY_CACHE_DISABLED")
    os.environ["SMARTMEMORY_CACHE_DISABLED"] = "true"
    try:
        result = get_cache()
        assert isinstance(result, NoOpCache), (
            "With override cleared and CACHE_DISABLED=true, get_cache() must return NoOpCache"
        )
    finally:
        if old is None:
            os.environ.pop("SMARTMEMORY_CACHE_DISABLED", None)
        else:
            os.environ["SMARTMEMORY_CACHE_DISABLED"] = old


def test_set_cache_override_clears_global_cache():
    """set_cache_override() resets _global_cache so stale instances are not returned."""
    import smartmemory.utils.cache as _cache_mod
    from smartmemory.utils.cache import set_cache_override, NoOpCache

    # Prime _global_cache with a sentinel so we can detect staleness
    sentinel = object()
    _cache_mod._global_cache = sentinel  # type: ignore[assignment]

    try:
        # Installing an override must clear _global_cache
        set_cache_override(NoOpCache())
        assert _cache_mod._global_cache is None, (
            "set_cache_override() must reset _global_cache to None"
        )

        # Clearing the override must also reset _global_cache
        _cache_mod._global_cache = sentinel  # type: ignore[assignment]
        set_cache_override(None)
        assert _cache_mod._global_cache is None, (
            "set_cache_override(None) must reset _global_cache to None"
        )
    finally:
        set_cache_override(None)
        _cache_mod._global_cache = None
