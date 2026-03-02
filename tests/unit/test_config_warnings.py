"""Tests for DEGRADE-1f: config warning log levels.

Validates that missing config files log at debug (expected) vs warning (explicit path).
"""

import logging
import os
from unittest.mock import patch

# Import at module level (before autouse conftest fixtures activate)
# so the stores.factory import chain runs with real VectorStore.
import smartmemory.memory.types.factory_memory_creator as fmc
from smartmemory.configuration.manager import ConfigManager


class TestConfigManagerLogLevel:
    """ConfigManager should log debug for default path, warning for explicit path."""

    def test_manager_default_path_logs_debug(self, caplog):
        """ConfigManager() with no config file and no env var logs debug, not warning."""
        old_env = os.environ.pop("SMARTMEMORY_CONFIG", None)
        try:
            with caplog.at_level(logging.DEBUG, logger="smartmemory.configuration.manager"):
                ConfigManager()

            # Should NOT have any WARNING about "Config file not found"
            warnings = [
                r for r in caplog.records if r.levelno >= logging.WARNING and "Config file not found" in r.message
            ]
            assert len(warnings) == 0, f"Expected no warnings for default path, got: {[w.message for w in warnings]}"
        finally:
            if old_env is not None:
                os.environ["SMARTMEMORY_CONFIG"] = old_env

    def test_manager_explicit_path_logs_warning(self, caplog):
        """ConfigManager(config_path='/nonexistent') logs warning for missing file."""
        with caplog.at_level(logging.DEBUG, logger="smartmemory.configuration.manager"):
            ConfigManager(config_path="/tmp/nonexistent_smartmemory_config_1234.json")

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "Config file not found" in r.message]
        assert len(warnings) >= 1, "Expected warning for explicit missing config path"


class TestFactoryMemoryCreatorLogLevel:
    """factory_memory_creator should log debug for FileNotFoundError, warning for other exceptions."""

    def test_factory_file_not_found_logs_debug(self, caplog):
        """FileNotFoundError from MemoryConfig() → debug, not warning."""
        with patch.object(fmc, "MemoryConfig", side_effect=FileNotFoundError("no config")):
            with caplog.at_level(logging.DEBUG, logger="smartmemory.memory.types.factory_memory_creator"):
                try:
                    fmc.create_memory_system()
                except Exception:
                    pass  # We only care about log level, not success

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "Could not load" in r.message]
        assert len(warnings) == 0, (
            f"Expected debug not warning for FileNotFoundError, got: {[w.message for w in warnings]}"
        )

    def test_factory_parse_error_logs_warning(self, caplog):
        """ValueError from MemoryConfig() → warning (real config problem)."""
        with patch.object(fmc, "MemoryConfig", side_effect=ValueError("bad json")):
            with caplog.at_level(logging.DEBUG, logger="smartmemory.memory.types.factory_memory_creator"):
                try:
                    fmc.create_memory_system()
                except Exception:
                    pass

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "Could not load" in r.message]
        assert len(warnings) >= 1, "Expected warning for ValueError (parse error)"
