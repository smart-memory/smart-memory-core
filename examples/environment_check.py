"""
Environment check for SmartMemory.

Performs non-destructive checks:
- Loads MemoryConfig and prints resolved path/namespace
- Instantiates SmartMemory
- Runs a tiny graph search
- Instantiates VectorStore and performs a small search
- Attempts cache access (read-only)

All checks are best-effort and print clear diagnostics.
"""

import sys


def main() -> int:
    print("SmartMemory Environment Check")
    print("=" * 50)

    # 1) Config
    try:
        from smartmemory.configuration import MemoryConfig
        cfg = MemoryConfig()
        print(f"✅ Config loaded: {cfg.resolved_config_path}")
        print(f"   Active namespace: {cfg.active_namespace}")
    except Exception as e:
        print(f"❌ Config load failed: {e}")
        return 1

    # 2) SmartMemory and Graph search
    try:
        from smartmemory.smart_memory import SmartMemory
        memory = SmartMemory()
        # Use canonical search; if backends are down, errors should be caught below
        results = memory.search("test", top_k=1)
        print(f"✅ SmartMemory instantiated; search returned {len(results)} result(s)")
    except Exception as e:
        print(f"❌ SmartMemory or graph search failed: {e}")
        # continue; other checks may still be informative

    # 3) Vector store
    try:
        from smartmemory.stores.vector.vector_store import VectorStore
        vs = VectorStore()
        # Attempt a small search with a dummy embedding length of 10
        # If backend is not initialized, this may raise; catch and report
        dummy = [0.0] * 10
        hits = vs.search(dummy, top_k=1)
        print(f"✅ VectorStore reachable; search returned {len(hits)} hit(s)")
    except Exception as e:
        print(f"⚠️ VectorStore check encountered an issue: {e}")

    # 4) Cache (read-only)
    try:
        from smartmemory.utils.cache import get_cache
        cache = get_cache()
        # Non-destructive: check a small, namespaced key pattern count
        # Avoid large scans; only test a direct ping via redis
        try:
            pong = cache.redis.ping()
            print(f"✅ Redis cache reachable: PING -> {pong}")
        except Exception as e:
            print(f"⚠️ Redis cache ping failed: {e}")
    except Exception as e:
        print(f"⚠️ Cache check unavailable: {e}")

    print("\nEnvironment check completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
