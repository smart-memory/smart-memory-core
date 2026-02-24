"""
Plugin Security and Sandboxing

This module provides security controls for plugin execution including:
- Resource limits (CPU, memory, time)
- Permission system (read/write/network)
- Execution sandboxing
- Security validation
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for plugin operations."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    FULL = "full"


@dataclass
class PluginPermissions:
    """
    Defines what a plugin is allowed to do.
    
    Attributes:
        memory_access: Permission level for memory operations (read/write)
        network_access: Whether plugin can make network requests
        file_access: Permission level for file system operations
        llm_access: Whether plugin can call LLM APIs
        graph_access: Permission level for graph database operations
        allowed_modules: List of allowed Python modules to import
        blocked_modules: List of blocked Python modules
    """
    memory_access: PermissionLevel = PermissionLevel.READ
    network_access: bool = False
    file_access: PermissionLevel = PermissionLevel.NONE
    llm_access: bool = False
    graph_access: PermissionLevel = PermissionLevel.READ
    allowed_modules: List[str] = field(default_factory=list)
    blocked_modules: List[str] = field(default_factory=lambda: [
        'os.system', 'subprocess', 'eval', 'exec', '__import__'
    ])
    
    def can_read_memory(self) -> bool:
        """Check if plugin can read from memory."""
        return self.memory_access in [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.FULL]
    
    def can_write_memory(self) -> bool:
        """Check if plugin can write to memory."""
        return self.memory_access in [PermissionLevel.WRITE, PermissionLevel.FULL]
    
    def can_access_network(self) -> bool:
        """Check if plugin can make network requests."""
        return self.network_access
    
    def can_read_files(self) -> bool:
        """Check if plugin can read files."""
        return self.file_access in [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.FULL]
    
    def can_write_files(self) -> bool:
        """Check if plugin can write files."""
        return self.file_access in [PermissionLevel.WRITE, PermissionLevel.FULL]


@dataclass
class ResourceLimits:
    """
    Resource limits for plugin execution.
    
    Attributes:
        max_execution_time_seconds: Maximum execution time (0 = unlimited)
        max_memory_mb: Maximum memory usage in MB (0 = unlimited)
        max_cpu_percent: Maximum CPU usage percentage (0 = unlimited)
        max_network_requests: Maximum number of network requests (0 = unlimited)
    """
    max_execution_time_seconds: float = 30.0
    max_memory_mb: int = 512
    max_cpu_percent: int = 80
    max_network_requests: int = 10


class SecurityException(Exception):
    """Raised when a security violation is detected."""
    pass


class ResourceLimitException(SecurityException):
    """Raised when a resource limit is exceeded."""
    pass


class PermissionDeniedException(SecurityException):
    """Raised when a permission check fails."""
    pass


class PluginSecurityContext:
    """
    Security context for plugin execution.
    
    Tracks resource usage and enforces limits during plugin execution.
    """
    
    def __init__(self, permissions: PluginPermissions, limits: ResourceLimits):
        self.permissions = permissions
        self.limits = limits
        self.start_time: Optional[float] = None
        self.network_requests_made: int = 0
        self._is_executing: bool = False
    
    def start_execution(self):
        """Mark the start of plugin execution."""
        self.start_time = time.time()
        self._is_executing = True
        self.network_requests_made = 0
    
    def end_execution(self):
        """Mark the end of plugin execution."""
        self._is_executing = False
    
    def check_execution_time(self):
        """Check if execution time limit is exceeded."""
        if not self._is_executing or self.start_time is None:
            return
        
        if self.limits.max_execution_time_seconds <= 0:
            return  # No limit
        
        elapsed = time.time() - self.start_time
        if elapsed > self.limits.max_execution_time_seconds:
            raise ResourceLimitException(
                f"Plugin execution time limit exceeded: {elapsed:.2f}s > {self.limits.max_execution_time_seconds}s"
            )
    
    def check_network_request(self):
        """Check if network request is allowed and within limits."""
        if not self.permissions.can_access_network():
            raise PermissionDeniedException("Plugin does not have network access permission")
        
        if self.limits.max_network_requests > 0:
            if self.network_requests_made >= self.limits.max_network_requests:
                raise ResourceLimitException(
                    f"Network request limit exceeded: {self.network_requests_made} >= {self.limits.max_network_requests}"
                )
        
        self.network_requests_made += 1
    
    def check_memory_read(self):
        """Check if memory read is allowed."""
        if not self.permissions.can_read_memory():
            raise PermissionDeniedException("Plugin does not have memory read permission")
    
    def check_memory_write(self):
        """Check if memory write is allowed."""
        if not self.permissions.can_write_memory():
            raise PermissionDeniedException("Plugin does not have memory write permission")
    
    def check_file_read(self):
        """Check if file read is allowed."""
        if not self.permissions.can_read_files():
            raise PermissionDeniedException("Plugin does not have file read permission")
    
    def check_file_write(self):
        """Check if file write is allowed."""
        if not self.permissions.can_write_files():
            raise PermissionDeniedException("Plugin does not have file write permission")


class PluginSandbox:
    """
    Sandbox for executing plugins with security controls.
    
    This class wraps plugin execution with:
    - Resource limits
    - Permission checks
    - Timeout enforcement
    - Error handling
    """
    
    def __init__(self, permissions: Optional[PluginPermissions] = None,
                 limits: Optional[ResourceLimits] = None):
        """
        Initialize plugin sandbox.
        
        Args:
            permissions: Plugin permissions (default: read-only)
            limits: Resource limits (default: moderate limits)
        """
        self.permissions = permissions or PluginPermissions()
        self.limits = limits or ResourceLimits()
        self.context = PluginSecurityContext(self.permissions, self.limits)
    
    def execute(self, plugin_method: Callable, *args, **kwargs) -> Any:
        """
        Execute a plugin method with security controls.
        
        Args:
            plugin_method: The plugin method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
        
        Returns:
            The result of the plugin method
        
        Raises:
            ResourceLimitException: If resource limits are exceeded
            PermissionDeniedException: If permission check fails
            TimeoutError: If execution times out
        """
        self.context.start_execution()
        
        try:
            # Execute with timeout
            if self.limits.max_execution_time_seconds > 0:
                result = self._execute_with_timeout(
                    plugin_method,
                    self.limits.max_execution_time_seconds,
                    *args,
                    **kwargs
                )
            else:
                result = plugin_method(*args, **kwargs)
            
            return result
            
        except SecurityException:
            # Re-raise security exceptions
            raise
        except Exception as e:
            logger.error(f"Plugin execution error: {e}")
            raise
        finally:
            self.context.end_execution()
    
    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Execute a function with a timeout.
        
        Args:
            func: Function to execute
            timeout: Timeout in seconds
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Thread is still running - timeout exceeded
            raise TimeoutError(f"Plugin execution exceeded timeout of {timeout}s")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


# Predefined security profiles
SECURITY_PROFILES = {
    'trusted': PluginPermissions(
        memory_access=PermissionLevel.FULL,
        network_access=True,
        file_access=PermissionLevel.READ,
        llm_access=True,
        graph_access=PermissionLevel.FULL
    ),
    'standard': PluginPermissions(
        memory_access=PermissionLevel.WRITE,
        network_access=True,
        file_access=PermissionLevel.NONE,
        llm_access=True,
        graph_access=PermissionLevel.WRITE
    ),
    'restricted': PluginPermissions(
        memory_access=PermissionLevel.READ,
        network_access=False,
        file_access=PermissionLevel.NONE,
        llm_access=False,
        graph_access=PermissionLevel.READ
    ),
    'untrusted': PluginPermissions(
        memory_access=PermissionLevel.READ,
        network_access=False,
        file_access=PermissionLevel.NONE,
        llm_access=False,
        graph_access=PermissionLevel.NONE
    )
}


def get_security_profile(profile_name: str) -> PluginPermissions:
    """
    Get a predefined security profile.
    
    Args:
        profile_name: Name of the profile ('trusted', 'standard', 'restricted', 'untrusted')
    
    Returns:
        PluginPermissions object
    
    Raises:
        ValueError: If profile name is invalid
    """
    if profile_name not in SECURITY_PROFILES:
        raise ValueError(f"Unknown security profile: {profile_name}. "
                        f"Available: {list(SECURITY_PROFILES.keys())}")
    return SECURITY_PROFILES[profile_name]


def validate_plugin_security(plugin_class, permissions: PluginPermissions) -> List[str]:
    """
    Validate that a plugin respects security constraints.
    
    This performs static analysis and metadata checks to detect potential security issues.
    
    Args:
        plugin_class: The plugin class to validate
        permissions: The permissions the plugin will run with
    
    Returns:
        List of security warnings (empty if no issues)
    """
    warnings = []
    
    # Check plugin metadata requirements against permissions
    try:
        if hasattr(plugin_class, 'metadata'):
            metadata = plugin_class.metadata()
            
            # Check network requirement
            if metadata.requires_network and not permissions.can_access_network():
                warnings.append(f"Plugin '{metadata.name}' requires network access but lacks network permission")
            
            # Check LLM requirement
            if metadata.requires_llm and not permissions.llm_access:
                warnings.append(f"Plugin '{metadata.name}' requires LLM access but lacks LLM permission")
    except Exception as e:
        logger.warning(f"Could not check plugin metadata: {e}")
    
    # Check for dangerous imports via static analysis
    import inspect
    try:
        source = inspect.getsource(plugin_class)
        
        # Check for blocked modules
        dangerous_imports = ['subprocess', 'os.system', 'eval', 'exec', '__import__']
        for dangerous in dangerous_imports:
            if dangerous in source:
                warnings.append(f"Plugin contains potentially dangerous code: {dangerous}")
        
        # Check network access
        if not permissions.can_access_network():
            network_modules = ['requests', 'urllib', 'http.client', 'socket']
            for net_module in network_modules:
                if f"import {net_module}" in source or f"from {net_module}" in source:
                    warnings.append(f"Plugin imports network module '{net_module}' but lacks network permission")
        
        # Check file access
        if not permissions.can_write_files():
            if 'open(' in source and ('w' in source or 'a' in source):
                warnings.append("Plugin may write files but lacks file write permission")
    
    except Exception as e:
        logger.warning(f"Could not validate plugin security: {e}")
    
    return warnings
