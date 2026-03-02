# Copyright (C) 2025 SmartMemory
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# For commercial licensing options, please contact: help@smartmemory.ai
# Commercial licenses are available for organizations that wish to use
# this software in proprietary applications without the AGPL restrictions.

"""
Configuration Manager

Consolidates configuration management from:
- config_loader.py (MemoryConfig class with file loading and namespace handling)
"""

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

from .environment import EnvironmentHandler
from .validator import ConfigValidator, ValidatedConfigDict

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager with file loading, validation, and namespace support"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager

        Args:
            config_path: Path to configuration file (optional)
        """
        # Resolve config path (param > env > default) and propagate to process env
        candidate_path = config_path or os.environ.get("SMARTMEMORY_CONFIG", "config.json")
        self._is_explicit_path = config_path is not None or "SMARTMEMORY_CONFIG" in os.environ
        self._config_path = EnvironmentHandler.resolve_config_path(candidate_path)
        EnvironmentHandler.set_config_path_env(self._config_path)

        # Thread safety for config reloading
        self._lock = threading.RLock()
        self._last_mtime = 0.0
        self._config: Dict[str, Any] = {}

        # Load initial configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load and process the configuration file with env expansion and namespace handling"""
        with self._lock:
            config_dict: Dict[str, Any] = {}

            # Compute current mtime first to decide if a physical read is required
            try:
                current_mtime = os.path.getmtime(self._config_path) if os.path.exists(self._config_path) else 0.0
            except Exception:
                current_mtime = 0.0

            # If file exists and hasn't changed, and we already have a config, skip physical load
            if current_mtime and self._last_mtime and current_mtime == self._last_mtime and self._config:
                logger.debug("Config unchanged; skipping reload")
                return

            # Load JSON configuration file
            try:
                if os.path.exists(self._config_path):
                    # Physical read will only happen if we didn't early-return above
                    with open(self._config_path, "r") as f:
                        config_dict = json.load(f)
                    logger.debug(f"Loaded config from: {self._config_path}")
                    self._last_mtime = current_mtime or 0.0
                else:
                    if self._is_explicit_path:
                        logger.warning(f"Config file not found: {self._config_path}. Using empty config.")
                    else:
                        logger.debug(f"Config file not found: {self._config_path}. Using empty config.")
                    self._last_mtime = 0.0
            except Exception as e:
                logger.error(f"Error loading config from {self._config_path}: {e}")

            # Process environment variables and overrides
            processed = EnvironmentHandler.process_config_dict(config_dict)

            # Handle namespace selection and merging
            merged = self._handle_namespaces(processed)

            self._config = merged
            logger.debug(f"Configuration loaded with {len(self._config)} top-level keys")

    def _handle_namespaces(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Handle namespace selection and deep merging

        Args:
            processed: Processed configuration dictionary

        Returns:
            Configuration with namespace overlay applied
        """

        def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively deep-merge override into base and return a new dict."""
            out: Dict[str, Any] = {}
            # Start with base
            for k, v in base.items():
                out[k] = v
            # Apply overrides
            for k, v in (override or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge_dicts(out[k], v)
                else:
                    out[k] = v
            return out

        merged = processed
        namespaces = processed.get("namespaces") if isinstance(processed.get("namespaces"), dict) else None
        active_ns = None

        if namespaces:
            # Determine active namespace: env TEST_NAMESPACE > config.namespaces.active > 'default' if present
            active_ns = EnvironmentHandler.get_namespace() or namespaces.get("active")
            if not active_ns and "default" in namespaces:
                active_ns = "default"

            # Only merge if a matching namespace overlay exists
            if active_ns and isinstance(namespaces.get(active_ns), dict):
                ns_overlay: Dict[str, Any] = namespaces.get(active_ns) or {}
                # Do not carry the top-level 'namespaces' key into the final config
                base_no_ns = {k: v for k, v in processed.items() if k != "namespaces"}
                merged = _deep_merge_dicts(base_no_ns, ns_overlay)
                logger.debug(f"Applied namespace overlay: {active_ns}")
            else:
                # Remove namespaces key to avoid leaking overlays to consumers
                merged = {k: v for k, v in processed.items() if k != "namespaces"}

        # Store active namespace for introspection
        merged["_active_namespace"] = active_ns
        return merged

    def reload_if_stale(self, force: bool = False) -> None:
        """Reload the config if the source file's mtime has changed or if forced.

        Args:
            force: Force reload even if file hasn't changed
        """
        try:
            mtime = os.path.getmtime(self._config_path) if os.path.exists(self._config_path) else 0.0
        except Exception:
            mtime = 0.0

        if force or (mtime and mtime > self._last_mtime):
            logger.debug("Config file changed; reloading configuration")
            self._load_config()

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration dictionary

        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    def get_validated_config(self) -> ValidatedConfigDict:
        """Get configuration wrapped in ValidatedConfigDict for fail-fast access

        Returns:
            ValidatedConfigDict instance
        """
        return ValidatedConfigDict(self._config, "config")

    def get_section(self, section_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a specific configuration section

        Args:
            section_name: Name of the configuration section
            default: Default value if section doesn't exist

        Returns:
            Configuration section dictionary
        """
        return self._config.get(section_name, default or {})

    def get_store_config(self, store_name: str) -> Dict[str, Any]:
        """Get store configuration, falling back to graph_db if store config is empty

        Args:
            store_name: Name of the store

        Returns:
            Store configuration dictionary
        """
        store_cfg = self.get_section(store_name)
        if not store_cfg:
            return self.get_section("graph_db")
        return store_cfg

    def validate_config(self) -> None:
        """Validate the current configuration using ConfigValidator

        Raises:
            KeyError: If required keys are missing
            ValueError: If configuration values are invalid
        """
        # Validate pipeline configuration if present
        if "pipeline" in self._config:
            ConfigValidator.validate_pipeline_config(self._config["pipeline"])

        # Validate service connection configurations
        for service in ["vector", "cache", "graph_db", "mongodb"]:
            if service in self._config:
                service_config = self._config[service]
                if isinstance(service_config, dict) and service_config:
                    try:
                        ConfigValidator.validate_connection_config(
                            service_config, service, required_keys=["host"] if service != "mongodb" else None
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Configuration validation warning for {service}: {e}")

        logger.debug("Configuration validation completed")

    @property
    def config_path(self) -> str:
        """Get the resolved configuration file path"""
        return self._config_path

    @property
    def active_namespace(self) -> Optional[str]:
        """Get the currently active namespace"""
        return self._config.get("_active_namespace")
