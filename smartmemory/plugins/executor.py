"""
Secure Plugin Executor

Wraps plugin execution with security controls and sandboxing.
"""

import logging
from typing import Any, Callable, Optional, Dict

from smartmemory.plugins.security import (
    ResourceLimits, PluginSandbox,
    get_security_profile, validate_plugin_security,
    SecurityException, PermissionLevel
)

logger = logging.getLogger(__name__)


class SecurePluginExecutor:
    """
    Executes plugins with security controls.
    
    This class wraps plugin method calls with:
    - Permission checks
    - Resource limits
    - Timeout enforcement
    - Security validation
    """
    
    def __init__(self):
        """Initialize the secure executor."""
        self._sandboxes: Dict[str, PluginSandbox] = {}
        self._security_warnings: Dict[str, list] = {}
    
    def register_plugin(self, plugin_name: str, plugin_class: type, 
                       metadata: Any) -> None:
        """
        Register a plugin with security controls.
        
        Args:
            plugin_name: Name of the plugin
            plugin_class: The plugin class
            metadata: Plugin metadata with security settings
        """
        # Get security profile from metadata
        profile_name = getattr(metadata, 'security_profile', 'standard')
        
        try:
            permissions = get_security_profile(profile_name)
        except ValueError:
            logger.warning(f"Unknown security profile '{profile_name}' for plugin '{plugin_name}', using 'standard'")
            permissions = get_security_profile('standard')
        
        # Override permissions based on metadata requirements
        if getattr(metadata, 'requires_network', False):
            permissions.network_access = True
        if getattr(metadata, 'requires_llm', False):
            permissions.llm_access = True
        if getattr(metadata, 'requires_file_access', False):
            permissions.file_access = PermissionLevel.READ
        
        # Create resource limits
        limits = ResourceLimits()
        
        # Create sandbox
        sandbox = PluginSandbox(permissions, limits)
        self._sandboxes[plugin_name] = sandbox
        
        # Validate plugin security
        warnings = validate_plugin_security(plugin_class, permissions)
        if warnings:
            self._security_warnings[plugin_name] = warnings
            logger.warning(f"Security warnings for plugin '{plugin_name}': {warnings}")
    
    def execute(self, plugin_name: str, plugin_method: Callable, 
                *args, **kwargs) -> Any:
        """
        Execute a plugin method with security controls.
        
        Args:
            plugin_name: Name of the plugin
            plugin_method: The method to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            The result of the plugin method
        
        Raises:
            SecurityException: If security violation occurs
        """
        sandbox = self._sandboxes.get(plugin_name)
        
        if sandbox is None:
            # No sandbox registered - use default restricted sandbox
            logger.warning(f"No sandbox for plugin '{plugin_name}', using restricted profile")
            permissions = get_security_profile('restricted')
            limits = ResourceLimits()
            sandbox = PluginSandbox(permissions, limits)
        
        try:
            return sandbox.execute(plugin_method, *args, **kwargs)
        except SecurityException as e:
            logger.error(f"Security violation in plugin '{plugin_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing plugin '{plugin_name}': {e}")
            raise
    
    def get_security_warnings(self, plugin_name: str) -> list:
        """
        Get security warnings for a plugin.
        
        Args:
            plugin_name: Name of the plugin
        
        Returns:
            List of security warnings
        """
        return self._security_warnings.get(plugin_name, [])
    
    def get_sandbox(self, plugin_name: str) -> Optional[PluginSandbox]:
        """
        Get the sandbox for a plugin.
        
        Args:
            plugin_name: Name of the plugin
        
        Returns:
            PluginSandbox or None if not registered
        """
        return self._sandboxes.get(plugin_name)


# Global executor instance
_global_executor = None


def get_plugin_executor() -> SecurePluginExecutor:
    """Get the global plugin executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = SecurePluginExecutor()
    return _global_executor


def reset_plugin_executor():
    """Reset the global plugin executor (mainly for testing)."""
    global _global_executor
    _global_executor = None
