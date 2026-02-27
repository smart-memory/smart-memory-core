"""Unit tests for SmartMemory._di_context() — CORE-DI-1.

All six tests are marked @pytest.mark.unit (no infrastructure required).
They verify:
  1. All four ContextVars are set inside _di_context and None after.
  2. Cache ContextVar isolation across two stubs (no Redis).
  3. Vector backend ContextVar isolation (per-instance writes stay separate).
  4. Observability ContextVar correctness for True/False instances.
  5. All ContextVars reset to None even when an exception propagates.
  6. Construction with all three params does NOT write process globals.
"""
import os
import threading
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONSTRUCTOR_PATCHES = [
    "smartmemory.smart_memory.EvolutionOrchestrator",
    "smartmemory.smart_memory.GlobalClustering",
    "smartmemory.smart_memory.VersionTracker",
    "smartmemory.smart_memory.TemporalQueries",
    "smartmemory.smart_memory.ProcedureMatcher",
    "smartmemory.smart_memory.DriftDetector",
]


def _make_mock_graph():
    g = MagicMock()
    g.search = MagicMock()
    g.search.set_smart_memory = MagicMock()
    return g


def _make_sm(stack, **kwargs):
    for target in _CONSTRUCTOR_PATCHES:
        stack.enter_context(patch(target))
    from smartmemory.smart_memory import SmartMemory
    return SmartMemory(graph=_make_mock_graph(), **kwargs)


# ---------------------------------------------------------------------------
# Test 1 — all four ContextVars are set inside and None after
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_di_context_sets_all_four_vars():
    """_di_context() sets all four ContextVars inside the block and resets them after."""
    from smartmemory.observability.events import _current_sink
    from smartmemory.observability.tracing import _observability_ctx
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx
    from smartmemory.utils.cache import NoOpCache, _cache_ctx

    stub_cache = NoOpCache()
    stub_backend = MagicMock()
    stub_sink = MagicMock()

    with ExitStack() as stack:
        mem = _make_sm(
            stack,
            cache=stub_cache,
            vector_backend=stub_backend,
            observability=False,
            event_sink=stub_sink,
        )

    # All vars start at None outside _di_context
    assert _cache_ctx.get() is None
    assert _vector_backend_ctx.get() is None
    assert _observability_ctx.get() is None
    assert _current_sink.get() is None

    with mem._di_context():
        assert _cache_ctx.get() is stub_cache
        assert _vector_backend_ctx.get() is stub_backend
        assert _observability_ctx.get() is False
        assert _current_sink.get() is stub_sink

    # All vars reset after exit
    assert _cache_ctx.get() is None
    assert _vector_backend_ctx.get() is None
    assert _observability_ctx.get() is None
    assert _current_sink.get() is None


# ---------------------------------------------------------------------------
# Test 2 — cache ContextVar isolation (two stub caches, no Redis)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cache_ctx_isolation():
    """Two SmartMemory instances with different stub caches see separate _cache_ctx values."""
    from smartmemory.utils.cache import NoOpCache, _cache_ctx

    stub_a = NoOpCache()
    stub_b = MagicMock()  # second "cache" — just needs to be a distinct object

    with ExitStack() as stack_a:
        mem_a = _make_sm(stack_a, cache=stub_a)
    with ExitStack() as stack_b:
        mem_b = _make_sm(stack_b, cache=stub_b)

    barrier = threading.Barrier(2)
    results = {}

    def _run(name, mem, expected):
        with mem._di_context():
            barrier.wait()  # both threads enter _di_context before either reads
            results[name] = _cache_ctx.get()

    t1 = threading.Thread(target=_run, args=("a", mem_a, stub_a))
    t2 = threading.Thread(target=_run, args=("b", mem_b, stub_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["a"] is stub_a, "Thread A must see stub_a, not stub_b"
    assert results["b"] is stub_b, "Thread B must see stub_b, not stub_a"


# ---------------------------------------------------------------------------
# Test 3 — vector backend ContextVar isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vector_backend_ctx_isolation():
    """Two SmartMemory instances with different backends see separate _vector_backend_ctx values."""
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx

    mock_a = MagicMock(name="backend_a")
    mock_b = MagicMock(name="backend_b")

    with ExitStack() as stack_a:
        mem_a = _make_sm(stack_a, vector_backend=mock_a)
    with ExitStack() as stack_b:
        mem_b = _make_sm(stack_b, vector_backend=mock_b)

    barrier = threading.Barrier(2)
    results = {}

    def _run(name, mem):
        with mem._di_context():
            barrier.wait()
            results[name] = _vector_backend_ctx.get()

    t1 = threading.Thread(target=_run, args=("a", mem_a))
    t2 = threading.Thread(target=_run, args=("b", mem_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["a"] is mock_a
    assert results["b"] is mock_b


# ---------------------------------------------------------------------------
# Test 4 — observability ContextVar isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_observability_ctx_isolation():
    """One instance with observability=False and one with True see correct _observability_ctx values."""
    from smartmemory.observability.tracing import _is_enabled, _observability_ctx

    with ExitStack() as stack_on:
        mem_on = _make_sm(stack_on, observability=True)
    with ExitStack() as stack_off:
        mem_off = _make_sm(stack_off, observability=False)

    # observability=True: _observability_ctx is set to None (explicit isolation; env default wins)
    with mem_on._di_context():
        assert _observability_ctx.get() is None, (
            "observability=True must set _observability_ctx to None so _is_enabled() falls through to env"
        )

    # observability=False: _observability_ctx is False
    with mem_off._di_context():
        assert _observability_ctx.get() is False
        assert _is_enabled() is False

    # Both reset after exit
    assert _observability_ctx.get() is None


# ---------------------------------------------------------------------------
# Test 5 — all ContextVars reset on exception
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_di_context_reset_on_exception():
    """All ContextVars reset to None even when an exception propagates out of _di_context."""
    from smartmemory.observability.tracing import _observability_ctx
    from smartmemory.stores.vector.vector_store import _vector_backend_ctx
    from smartmemory.utils.cache import NoOpCache, _cache_ctx

    stub_cache = NoOpCache()
    stub_backend = MagicMock()

    with ExitStack() as stack:
        mem = _make_sm(stack, cache=stub_cache, vector_backend=stub_backend, observability=False)

    try:
        with mem._di_context():
            raise RuntimeError("synthetic failure inside _di_context")
    except RuntimeError:
        pass

    assert _cache_ctx.get() is None, "_cache_ctx must reset after exception"
    assert _vector_backend_ctx.get() is None, "_vector_backend_ctx must reset after exception"
    assert _observability_ctx.get() is None, "_observability_ctx must reset after exception"


# ---------------------------------------------------------------------------
# Test 6 — construction does NOT mutate any process globals
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_global_mutation_on_construct():
    """SmartMemory(cache=x, vector_backend=y, observability=False) must NOT write process globals."""
    from smartmemory.stores.vector.vector_store import _DEFAULT_BACKEND
    from smartmemory.utils.cache import _CACHE_OVERRIDE, NoOpCache

    # Capture pre-construction state
    pre_backend = _DEFAULT_BACKEND
    pre_cache = _CACHE_OVERRIDE
    pre_obs = os.environ.get("SMARTMEMORY_OBSERVABILITY")

    stub_backend = MagicMock()
    stub_cache = NoOpCache()

    with ExitStack() as stack:
        _make_sm(stack, cache=stub_cache, vector_backend=stub_backend, observability=False)

    # Process globals must be unchanged after construction
    from smartmemory.stores.vector.vector_store import _DEFAULT_BACKEND as post_backend
    from smartmemory.utils.cache import _CACHE_OVERRIDE as post_cache

    assert post_backend is pre_backend, "_DEFAULT_BACKEND must not be mutated by constructor"
    assert post_cache is pre_cache, "_CACHE_OVERRIDE must not be mutated by constructor"
    assert os.environ.get("SMARTMEMORY_OBSERVABILITY") == pre_obs, (
        "SMARTMEMORY_OBSERVABILITY env var must not be mutated by constructor"
    )
