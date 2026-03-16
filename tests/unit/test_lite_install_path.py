"""Unit tests for DIST-LITE-6: lazy-import safety for the zero-infra install path.

Structure:
  TestImportTimeWithoutInfra   — each lazified module can be imported without its infra pkg
  TestHardFailInstantiation    — FalkorDB backends + RedisCache raise ImportError on init
  TestSoftFailInstantiation    — MetricsConsumer + EventSpooler do not raise when Redis absent
  TestEnsureSpacyModel         — auto-download logic in create_lite_memory()
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@contextmanager
def _absent_package(pkg: str, *also_absent: str):
    """Set pkg (and optionally extra sub-packages) to None in sys.modules.

    Python checks sys.modules before the filesystem, so setting a key to None
    causes ``import pkg`` to raise ImportError even when the package is installed
    on disk.  This is the canonical CPython mechanism for blocking an import.
    """
    block = {pkg: None}
    for extra in also_absent:
        block[extra] = None
    with patch.dict(sys.modules, block):
        yield


@contextmanager
def _reload_module(module_name: str, pkg: str, *also_absent: str):
    """Block pkg, force-reimport module_name, restore cache on exit.

    Used for import-time tests only: pops the cached module so Python
    re-executes module-level code under the patched sys.modules.

    Restores both sys.modules[module_name] AND the parent package's attribute
    so that subsequent ``import a.b.c`` and ``importlib.reload(mod)`` see the
    same module object.  Without the attribute sync, importlib.import_module()
    inside _absent_package writes ``smartmemory.utils.cache = new_mod`` on the
    parent while we restore sys.modules to the original; downstream code that
    resolves via the parent attribute gets the wrong object, causing
    importlib.reload() to raise "module not in sys.modules".
    """
    orig = sys.modules.pop(module_name, None)
    # Save parent's current attribute so we can restore it too.
    parts = module_name.rsplit(".", 1)
    parent_name, attr = (parts[0], parts[1]) if len(parts) == 2 else (None, None)
    orig_parent_attr = getattr(sys.modules.get(parent_name), attr, None) if parent_name else None
    try:
        with _absent_package(pkg, *also_absent):
            yield
    finally:
        sys.modules.pop(module_name, None)
        target = orig if orig is not None else orig_parent_attr
        if target is not None:
            sys.modules[module_name] = target
        # Sync parent package attribute to match sys.modules, preventing
        # importlib.reload() failures in tests that run after this context.
        if parent_name and attr:
            parent = sys.modules.get(parent_name)
            if parent is not None and target is not None:
                setattr(parent, attr, target)


# ---------------------------------------------------------------------------
# Import-time tests
# ---------------------------------------------------------------------------

class TestImportTimeWithoutInfra:
    """Each lazified module must be importable even when its infra pkg is absent."""

    def _assert_importable(self, module_name: str, pkg: str, *also: str) -> None:
        with _reload_module(module_name, pkg, *also):
            mod = importlib.import_module(module_name)
            assert mod is not None

    def test_falkordb_backend_importable_without_falkordb(self):
        self._assert_importable("smartmemory.graph.backends.falkordb", "falkordb")

    def test_async_falkordb_backend_importable_without_falkordb(self):
        self._assert_importable(
            "smartmemory.graph.backends.async_falkordb",
            "falkordb",
            "falkordb.asyncio",
        )

    def test_ontology_registry_importable_without_falkordb(self):
        self._assert_importable("smartmemory.ontology.registry", "falkordb")

    def test_falkor_vector_backend_importable_without_falkordb(self):
        self._assert_importable("smartmemory.stores.vector.backends.falkor", "falkordb")

    def test_falkordb_graph_service_importable_without_falkordb(self):
        self._assert_importable("smartmemory.stores.ontology.falkordb", "falkordb")

    def test_metrics_consumer_importable_without_redis(self):
        self._assert_importable("smartmemory.pipeline.metrics_consumer", "redis")

    def test_events_importable_without_redis(self):
        self._assert_importable("smartmemory.observability.events", "redis")

    def test_cache_importable_without_redis(self):
        self._assert_importable("smartmemory.utils.cache", "redis")

    # -- LLM/ML packages (optional since DIST-LITE-7) --

    def test_litellm_client_importable_without_litellm(self):
        self._assert_importable("smartmemory.utils.llm_client.litellm", "litellm")

    def test_openai_client_importable_without_openai(self):
        self._assert_importable("smartmemory.utils.llm_client.openai", "openai")

    def test_dspy_client_importable_without_dspy(self):
        self._assert_importable("smartmemory.utils.llm_client.dspy", "dspy")

    def test_temporal_enricher_importable_without_openai(self):
        self._assert_importable("smartmemory.plugins.enrichers.temporal", "openai")

    def test_clustering_importable_without_sklearn(self):
        self._assert_importable("smartmemory.clustering.embedding", "sklearn", "sklearn.cluster")

    def test_hybrid_retrieval_importable_without_sklearn(self):
        self._assert_importable("smartmemory.utils.hybrid_retrieval", "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise")

    def test_analytics_stage_importable_without_sklearn(self):
        self._assert_importable(
            "smartmemory.memory.pipeline.stages.analytics",
            "sklearn", "sklearn.feature_extraction", "sklearn.feature_extraction.text",
            "sklearn.decomposition",
        )


# ---------------------------------------------------------------------------
# Instantiation behavior — soft fail (LLM/ML enrichers without deps)
# ---------------------------------------------------------------------------

class TestSoftFailLLMML:
    """Enrichers that need LLM/ML packages must degrade gracefully, not crash."""

    def test_temporal_enricher_returns_empty_without_openai(self):
        with _absent_package("openai"):
            from smartmemory.plugins.enrichers.temporal import TemporalEnricher
            enricher = TemporalEnricher()
            assert enricher._openai is None
            result = enricher.enrich("test content")
            assert result == {"temporal": {}}


# ---------------------------------------------------------------------------
# Instantiation behavior — hard fail (FalkorDB backends + RedisCache)
# ---------------------------------------------------------------------------

class TestHardFailInstantiation:
    """Classes that require infra must raise ImportError with install hint."""

    def test_falkordb_backend_raises_with_hint(self):
        with _absent_package("falkordb"):
            from smartmemory.graph.backends.falkordb import FalkorDBBackend
            with pytest.raises(ImportError, match=r"smartmemory-core\[server\]"):
                FalkorDBBackend()

    def test_async_falkordb_backend_raises_on_connect(self):
        import asyncio
        with _absent_package("falkordb", "falkordb.asyncio"):
            from smartmemory.graph.backends.async_falkordb import AsyncFalkorDBBackend
            backend = AsyncFalkorDBBackend()  # __init__ stores config only — must not raise
            with pytest.raises(ImportError, match=r"smartmemory-core\[server\]"):
                asyncio.run(backend.connect())

    def test_ontology_registry_raises_with_hint(self):
        with _absent_package("falkordb"):
            from smartmemory.ontology.registry import OntologyRegistry
            with pytest.raises(ImportError, match=r"smartmemory-core\[server\]"):
                OntologyRegistry()

    def test_falkor_vector_backend_raises_with_hint(self):
        with _absent_package("falkordb"):
            from smartmemory.stores.vector.backends.falkor import FalkorVectorBackend
            with pytest.raises(ImportError, match=r"smartmemory-core\[server\]"):
                FalkorVectorBackend("test_collection", None)

    def test_falkordb_graph_service_raises_with_hint(self):
        with _absent_package("falkordb"):
            from smartmemory.stores.ontology.falkordb import FalkorDBGraphService
            with pytest.raises(ImportError, match=r"smartmemory-core\[server\]"):
                FalkorDBGraphService()

    def test_redis_cache_raises_with_hint(self):
        with _absent_package("redis"):
            from smartmemory.utils.cache import RedisCache
            with pytest.raises(ImportError, match=r"smartmemory-core\[server\]"):
                RedisCache()


# ---------------------------------------------------------------------------
# Instantiation behavior — soft fail (Redis observability classes)
# ---------------------------------------------------------------------------

class TestSoftFailInstantiation:
    """Observability classes must not raise when Redis is absent."""

    def test_metrics_consumer_no_raise_redis_absent(self):
        with _absent_package("redis"):
            from smartmemory.pipeline.metrics_consumer import MetricsConsumer
            consumer = MetricsConsumer()
            assert consumer._redis_available is False

    def test_metrics_consumer_run_returns_zero_when_unavailable(self):
        with _absent_package("redis"):
            from smartmemory.pipeline.metrics_consumer import MetricsConsumer
            consumer = MetricsConsumer()
            assert consumer.run() == 0

    def test_metrics_consumer_get_aggregated_returns_list_when_unavailable(self):
        with _absent_package("redis"):
            from smartmemory.pipeline.metrics_consumer import MetricsConsumer
            consumer = MetricsConsumer()
            result = consumer.get_aggregated("stage")
            assert result == []
            assert isinstance(result, list)

    def test_event_spooler_no_raise_redis_absent(self):
        _mock_cfg = MagicMock()
        _mock_cfg.cache.redis.host = "localhost"
        _mock_cfg.cache.redis.port = 9012
        _mock_cfg.get.return_value = {}

        with _absent_package("redis"):
            with patch("smartmemory.observability.events.get_config", return_value=_mock_cfg):
                from smartmemory.observability.events import EventSpooler
                spooler = EventSpooler()
                assert spooler._connected is False

    def test_event_spooler_emit_no_raise_when_disconnected(self):
        _mock_cfg = MagicMock()
        _mock_cfg.cache.redis.host = "localhost"
        _mock_cfg.cache.redis.port = 9012
        _mock_cfg.get.return_value = {}

        with _absent_package("redis"):
            with patch("smartmemory.observability.events.get_config", return_value=_mock_cfg):
                from smartmemory.observability.events import EventSpooler
                spooler = EventSpooler()
                # Must not raise — emit is a no-op when disconnected
                spooler.emit_event("test_event", "test", "op")


# ---------------------------------------------------------------------------
# spaCy auto-download tests
# ---------------------------------------------------------------------------

class TestEnsureSpacyModel:

    def test_skips_download_if_model_already_installed(self):
        with patch("spacy.util.is_package", return_value=True) as mock_check:
            with patch("spacy.cli.download") as mock_dl:
                from smartmemory.tools.factory import _ensure_spacy_model
                _ensure_spacy_model()
                mock_check.assert_called_once_with("en_core_web_sm")
                mock_dl.assert_not_called()

    def test_downloads_when_model_missing(self):
        with patch("spacy.util.is_package", return_value=False):
            with patch("spacy.cli.download") as mock_dl:
                from smartmemory.tools.factory import _ensure_spacy_model
                _ensure_spacy_model()
                mock_dl.assert_called_once_with("en_core_web_sm")

    def test_falls_back_to_print_when_rich_absent(self):
        """Verify the rich-absent fallback path completes without raising."""
        orig_factory = sys.modules.pop("smartmemory.tools.factory", None)
        try:
            with patch.dict(sys.modules, {"rich": None, "rich.console": None}):
                with patch("spacy.util.is_package", return_value=False):
                    with patch("spacy.cli.download") as mock_dl:
                        from smartmemory.tools.factory import _ensure_spacy_model
                        _ensure_spacy_model()  # must not raise despite rich being absent
                        mock_dl.assert_called_once_with("en_core_web_sm")
        finally:
            sys.modules.pop("smartmemory.tools.factory", None)
            if orig_factory is not None:
                sys.modules["smartmemory.tools.factory"] = orig_factory
