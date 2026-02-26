"""Unit tests for CORE-SYS2-1c: flag wiring through _build_pipeline_config() and sync-only contract."""

from unittest.mock import MagicMock


def _make_smart_memory(**kwargs):
    """Build a minimal SmartMemory-like object with _build_pipeline_config accessible."""
    # We can't easily construct SmartMemory (needs infra), so test the function directly
    # by calling it as an unbound method with a mock self.
    mock_self = MagicMock()
    mock_self._pipeline_profile = kwargs.get("pipeline_profile", None)
    mock_self.scope_provider.get_scope.side_effect = Exception("no scope")

    return mock_self


class TestFlagWiring:
    def test_extract_reasoning_sets_config(self):
        from smartmemory.smart_memory import SmartMemory

        mock_self = _make_smart_memory()
        config = SmartMemory._build_pipeline_config(
            mock_self, extract_reasoning=True,
        )
        assert config.extraction.reasoning_detect.enabled is True

    def test_extract_reasoning_false_by_default(self):
        from smartmemory.smart_memory import SmartMemory

        mock_self = _make_smart_memory()
        config = SmartMemory._build_pipeline_config(mock_self)
        assert config.extraction.reasoning_detect.enabled is False

    def test_extract_decisions_sets_config(self):
        from smartmemory.smart_memory import SmartMemory

        mock_self = _make_smart_memory()
        config = SmartMemory._build_pipeline_config(
            mock_self, extract_decisions=True,
        )
        assert config.extraction.llm_extract.extract_decisions is True

    def test_extract_reasoning_survives_lite_profile(self):
        """Flag stamp must happen AFTER _apply_pipeline_profile() so lite profile doesn't overwrite."""
        from smartmemory.smart_memory import SmartMemory
        from smartmemory.pipeline.config import PipelineConfig

        mock_self = _make_smart_memory(pipeline_profile=PipelineConfig.lite())
        config = SmartMemory._build_pipeline_config(
            mock_self, extract_reasoning=True,
        )
        # Lite profile disables LLM extract but must NOT touch reasoning_detect
        assert config.extraction.reasoning_detect.enabled is True
        assert config.extraction.llm_extract.enabled is False  # lite profile effect

    def test_extract_decisions_survives_lite_profile(self):
        """Pre-existing 1b gap fix: extract_decisions must also survive profile application."""
        from smartmemory.smart_memory import SmartMemory
        from smartmemory.pipeline.config import PipelineConfig

        mock_self = _make_smart_memory(pipeline_profile=PipelineConfig.lite())
        config = SmartMemory._build_pipeline_config(
            mock_self, extract_decisions=True,
        )
        assert config.extraction.llm_extract.extract_decisions is True

    def test_both_flags_together(self):
        from smartmemory.smart_memory import SmartMemory

        mock_self = _make_smart_memory()
        config = SmartMemory._build_pipeline_config(
            mock_self, extract_decisions=True, extract_reasoning=True,
        )
        assert config.extraction.llm_extract.extract_decisions is True
        assert config.extraction.reasoning_detect.enabled is True


class TestSyncOnlyContract:
    """Encode the sync-only extraction flags contract from blueprint §7."""

    def test_tier1_config_has_reasoning_disabled(self):
        """Core async path uses PipelineConfig.tier1() which must NOT enable reasoning."""
        from smartmemory.pipeline.config import PipelineConfig

        tier1 = PipelineConfig.tier1()
        assert tier1.extraction.reasoning_detect.enabled is False

    def test_default_config_has_reasoning_disabled(self):
        """Default config (used by EventBus async) must NOT enable reasoning."""
        from smartmemory.pipeline.config import PipelineConfig

        config = PipelineConfig()
        assert config.extraction.reasoning_detect.enabled is False

    def test_with_reasoning_is_only_explicit_opt_in(self):
        """Only PipelineConfig.with_reasoning() enables it — no other factory does."""
        from smartmemory.pipeline.config import PipelineConfig

        for factory in [PipelineConfig.default, PipelineConfig.lite, PipelineConfig.tier1,
                        PipelineConfig.preview, PipelineConfig.with_decisions]:
            config = factory()
            assert config.extraction.reasoning_detect.enabled is False, (
                f"{factory.__name__}() should not enable reasoning_detect"
            )

        # Only with_reasoning enables it
        assert PipelineConfig.with_reasoning().extraction.reasoning_detect.enabled is True
