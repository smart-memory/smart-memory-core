"""
Environment Variable Handler

Provides leaf-key environment overrides for configuration dictionaries.
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class EnvironmentHandler:
    """Environment variable handling for configuration overrides."""

    @staticmethod
    def override_leaf_keys(d: Dict[str, Any], prefix: str = None) -> Dict[str, Any]:
        """Recursively override leaf keys in dict d with env vars of form PREFIX_KEY.

        This is the sole mechanism for env var → config value injection.

        Args:
            d: Dictionary to process
            prefix: Environment variable prefix

        Returns:
            Dictionary with leaf keys overridden by environment variables
        """
        out = {}
        for k, v in d.items():
            env_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                out[k] = EnvironmentHandler.override_leaf_keys(v, env_key.upper())
            else:
                env_val = os.environ.get(env_key.upper())
                out[k] = env_val if env_val is not None else v
        return out

    @classmethod
    def process_config_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration dictionary with environment overrides.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Processed configuration with leaf-key env overrides applied
        """
        return cls.override_leaf_keys(config_dict)

    @staticmethod
    def resolve_config_path(path_candidate: str) -> str:
        """Find an existing config file based on candidate and fallbacks.

        Args:
            path_candidate: Primary config file path

        Returns:
            Absolute path to config file (may not exist)
        """
        # Expand env vars and user tilde in the provided path
        primary = os.path.expanduser(os.path.expandvars(path_candidate))

        if os.path.exists(primary):
            return os.path.abspath(primary)

        basename = os.path.basename(primary)
        current_path = os.path.join(os.getcwd(), basename)
        if os.path.exists(current_path):
            return os.path.abspath(current_path)

        parent_path = os.path.join(os.path.dirname(os.getcwd()), basename)
        if os.path.exists(parent_path):
            return os.path.abspath(parent_path)

        logger.debug(
            f"Could not find config file at {primary}, {current_path}, or {parent_path}. Using empty config."
        )
        return os.path.abspath(primary)

    @staticmethod
    def set_config_path_env(resolved_path: str):
        """Set the resolved config path in environment for process-wide consistency.

        Args:
            resolved_path: Absolute path to config file
        """
        os.environ["SMARTMEMORY_CONFIG"] = resolved_path
        logger.debug(f"Set SMARTMEMORY_CONFIG={resolved_path}")

    @staticmethod
    def get_namespace() -> str:
        """Get active namespace from environment or config.

        Returns:
            Active namespace name or None
        """
        return os.environ.get("TEST_NAMESPACE")
