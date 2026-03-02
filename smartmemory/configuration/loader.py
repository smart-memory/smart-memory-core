"""Unified configuration loader. One call per service at startup."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_loaded = False


def load_environment(config_dir: Optional[Path] = None) -> None:
    """Load base.env + secrets.env + {env}.env into os.environ.

    OS/Docker environment has highest precedence (setdefault only fills gaps).
    Within files: base -> secrets -> env override (later wins).
    """
    global _loaded
    if _loaded:
        logger.debug("load_environment() already called, skipping")
        return

    config_dir = config_dir or _find_config_dir()
    if config_dir is None:
        logger.debug("No config directory found, skipping load_environment()")
        _loaded = True
        return

    env = os.environ.get("ENVIRONMENT", "development")
    env_name = {
        "development": None,  # base.env IS the dev config
        "production": "production",
        "test": "ci",
    }.get(env, env)

    merged: dict[str, str] = {}
    filenames = ["base.env", "secrets.env"]
    if env_name:
        filenames.append(f"{env_name}.env")

    for filename in filenames:
        filepath = config_dir / filename
        if filepath.exists():
            merged.update(_parse_env_file(filepath))
            logger.debug("Loaded %s (%d vars)", filepath, len(merged))

    for key, value in merged.items():
        os.environ.setdefault(key, value)

    # Also set SMARTMEMORY_CONFIG so ConfigManager can find config.json
    # from any working directory (per-service copies were removed in CFG-1c).
    if not os.environ.get("SMARTMEMORY_CONFIG"):
        config_json = _find_config_json()
        if config_json is not None:
            os.environ["SMARTMEMORY_CONFIG"] = str(config_json)
            logger.debug("Set SMARTMEMORY_CONFIG=%s", config_json)

    logger.debug("load_environment() complete: %d vars from %s", len(merged), config_dir)
    _loaded = True


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load config.json structural defaults. No interpolation."""
    config_path = config_path or _find_config_json()
    if config_path is None or not config_path.exists():
        logger.debug("Config file not found: %s", config_path)
        return {}
    with open(config_path) as f:
        return json.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins for leaf values."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _find_config_dir() -> Optional[Path]:
    """Find smart-memory-infra/config/ directory."""
    env_dir = os.environ.get("SMARTMEMORY_CONFIG_DIR", "")
    candidates = [
        Path(env_dir) if env_dir else None,
        Path.cwd().parent / "smart-memory-infra" / "config",
        Path.cwd().parent.parent / "smart-memory-infra" / "config",
        Path.home() / "reg/my/SmartMemory/smart-memory-infra/config",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _find_config_json() -> Optional[Path]:
    """Find config.json in smart-memory-core/ core library."""
    env_path = os.environ.get("SMARTMEMORY_CONFIG", "")
    candidates = [
        Path(env_path) if env_path else None,
        Path.cwd() / "config.json",
        Path.cwd().parent / "smart-memory-core" / "config.json",
        Path.cwd().parent.parent / "smart-memory-core" / "config.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _parse_env_file(filepath: Path) -> dict[str, str]:
    """Parse .env file into dict. Handles comments, empty lines, quoted values."""
    result: dict[str, str] = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            elif " #" in value:
                # Strip inline comments (only for unquoted values)
                value = value[: value.index(" #")].rstrip()
            result[key] = value
    return result
