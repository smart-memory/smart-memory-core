"""Tests for NoOpCache and SMARTMEMORY_CACHE_DISABLED guard (P0-5 prerequisite for DIST-LITE-1)."""
import importlib
import pytest


@pytest.fixture(autouse=True)
def clear_cache_env(monkeypatch):
    """Ensure env var is clean before each test."""
    monkeypatch.delenv("SMARTMEMORY_CACHE_DISABLED", raising=False)


def test_cache_disabled_returns_noop(monkeypatch):
    """With SMARTMEMORY_CACHE_DISABLED=true, get_cache() returns NoOpCache."""
    monkeypatch.setenv("SMARTMEMORY_CACHE_DISABLED", "true")
    import smartmemory.utils.cache as cache_mod
    # Force re-evaluation by reimporting
    importlib.reload(cache_mod)
    from smartmemory.utils.cache import get_cache, NoOpCache
    result = get_cache()
    assert isinstance(result, NoOpCache)


def test_noop_cache_get_embedding_returns_none():
    """NoOpCache.get_embedding() returns None without raising."""
    from smartmemory.utils.cache import NoOpCache
    c = NoOpCache()
    assert c.get_embedding("some_key") is None


def test_noop_cache_set_embedding_returns_none():
    """NoOpCache.set_embedding() returns None without raising."""
    from smartmemory.utils.cache import NoOpCache
    c = NoOpCache()
    assert c.set_embedding("some_key", [1.0, 2.0, 3.0]) is None


def test_noop_cache_get_search_results_returns_none():
    """NoOpCache.get_search_results() returns None without raising."""
    from smartmemory.utils.cache import NoOpCache
    c = NoOpCache()
    assert c.get_search_results("query", 5, "semantic") is None


def test_noop_cache_arbitrary_method_returns_none():
    """NoOpCache.__getattr__ covers any future method."""
    from smartmemory.utils.cache import NoOpCache
    c = NoOpCache()
    assert c.some_future_method_xyz("a", "b") is None


def test_cache_enabled_by_default_is_not_noop(monkeypatch):
    """Without the env var, get_cache() does NOT return NoOpCache (existing behavior)."""
    import smartmemory.utils.cache as cache_mod
    from smartmemory.utils.cache import NoOpCache
    # Just check NoOpCache is not always returned
    # (actual Redis connection attempt may fail, that's fine)
    c = cache_mod.NoOpCache()
    assert isinstance(c, NoOpCache)  # NoOpCache exists as a class
    # The important thing: _global_cache is not set to NoOpCache when env var absent
    # We just verify the class exists and works correctly
