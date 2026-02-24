import threading
from datetime import datetime, timezone


def flatten_dict(d, parent_key='', sep='__'):
    """
    Recursively flattens a nested dict using double underscore as separator.
    
    This ensures single underscores in field names are preserved.
    Example: {'metadata': {'user_name': 'John'}} -> {'metadata__user_name': 'John'}
    
    Args:
        d: Dictionary to flatten
        parent_key: Internal use for recursion
        sep: Separator (default '__' to avoid conflicts with field names)
    
    Returns:
        Flattened dictionary with nested keys joined by separator
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='__'):
    """
    Reconstructs a nested dict from a flat dict with double underscore separator.
    
    Only splits on the separator (default '__'), preserving single underscores in keys.
    Example: {'metadata__user_name': 'John'} -> {'metadata': {'user_name': 'John'}}
    
    Args:
        d: Flattened dictionary
        sep: Separator to split on (default '__')
    
    Returns:
        Nested dictionary structure
    """
    result = {}
    for k, v in d.items():
        # Only split if separator is present
        if sep in k:
            parts = k.split(sep)
            cur = result
            for part in parts[:-1]:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]
            cur[parts[-1]] = v
        else:
            # No separator - keep key as-is (preserves single underscores)
            result[k] = v
    return result


def now():
    """Return the current UTC time as an aware datetime object."""
    return datetime.now(timezone.utc)


_config_cache = None
_config_lock = threading.RLock()


def get_config(section: str = None, config_path: str = None):
    """One-liner to get a MemoryConfig instance subsection (e.g., 'adapter', 'extractor').
    Uses caching to prevent excessive config.json loading.
    Returns ConfigDict for fail-fast configuration access with clear error messages.

    Args:
        section: Optional section name to return (e.g., 'cache', 'graph_db')
        config_path: Optional path to config directory (defaults to current working directory)
    """
    # Lazy import to break circular dependency:
    # utils -> configuration -> models -> memory_item -> utils
    from smartmemory.configuration import MemoryConfig
    from smartmemory.configuration.models import ConfigDict

    global _config_cache

    with _config_lock:
        # Load config only once and cache it
        if _config_cache is None:
            _config_cache = MemoryConfig(config_path=config_path)
        # Reload transparently if source changed
        _config_cache.reload_if_stale()

        config = _config_cache
    if section is None:
        # Return full config with backward compatibility mapping
        full_config = config._config.copy()
        # Add 'graph' key for backward compatibility (maps to graph_db)
        if 'graph_db' in full_config and 'graph' not in full_config:
            full_config['graph'] = {
                'backend': full_config['graph_db'].get('backend_class', 'FalkorDBBackend')
            }
        # Surface the active namespace (if any) for downstream stages
        try:
            active_ns = getattr(config, 'active_namespace', None)
        except Exception:
            active_ns = None
        if active_ns is not None:
            full_config['active_namespace'] = active_ns
        return ConfigDict(full_config, "config")

    # Return section as ConfigDict
    section_data = config._config.get(section) or {}
    return ConfigDict(section_data, f"config.{section}")


def clear_config_cache():
    """Clear the config cache. Useful for testing or when config changes."""
    global _config_cache
    with _config_lock:
        _config_cache = None
