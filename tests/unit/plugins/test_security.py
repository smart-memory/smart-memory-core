"""
Tests for plugin security and sandboxing.
"""

import pytest


pytestmark = pytest.mark.unit
import time
from smartmemory.plugins.security import (
    PluginPermissions, ResourceLimits, PluginSandbox,
    PermissionLevel, ResourceLimitException,
    PermissionDeniedException, get_security_profile, validate_plugin_security
)
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata


class TestPluginPermissions:
    """Test plugin permission system."""
    
    def test_default_permissions(self):
        """Test default permissions are read-only."""
        perms = PluginPermissions()
        
        assert perms.can_read_memory()
        assert not perms.can_write_memory()
        assert not perms.can_access_network()
        assert not perms.can_write_files()
    
    def test_full_permissions(self):
        """Test full permissions."""
        perms = PluginPermissions(
            memory_access=PermissionLevel.FULL,
            network_access=True,
            file_access=PermissionLevel.FULL,
            llm_access=True
        )
        
        assert perms.can_read_memory()
        assert perms.can_write_memory()
        assert perms.can_access_network()
        assert perms.can_read_files()
        assert perms.can_write_files()
    
    def test_restricted_permissions(self):
        """Test restricted permissions."""
        perms = PluginPermissions(
            memory_access=PermissionLevel.NONE,
            network_access=False,
            file_access=PermissionLevel.NONE
        )
        
        assert not perms.can_read_memory()
        assert not perms.can_write_memory()
        assert not perms.can_access_network()


class TestResourceLimits:
    """Test resource limit configuration."""
    
    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        
        assert limits.max_execution_time_seconds == 30.0
        assert limits.max_memory_mb == 512
        assert limits.max_cpu_percent == 80
        assert limits.max_network_requests == 10
    
    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_execution_time_seconds=10.0,
            max_memory_mb=256,
            max_cpu_percent=50,
            max_network_requests=5
        )
        
        assert limits.max_execution_time_seconds == 10.0
        assert limits.max_memory_mb == 256


class TestPluginSandbox:
    """Test plugin sandbox execution."""
    
    def test_successful_execution(self):
        """Test executing a simple function."""
        sandbox = PluginSandbox()
        
        def simple_func(x, y):
            return x + y
        
        result = sandbox.execute(simple_func, 2, 3)
        assert result == 5
    
    def test_execution_timeout(self):
        """Test that long-running functions timeout."""
        limits = ResourceLimits(max_execution_time_seconds=1.0)
        sandbox = PluginSandbox(limits=limits)
        
        def slow_func():
            time.sleep(5)
            return "done"
        
        with pytest.raises(TimeoutError):
            sandbox.execute(slow_func)
    
    def test_execution_with_exception(self):
        """Test that exceptions are propagated."""
        sandbox = PluginSandbox()
        
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            sandbox.execute(failing_func)
    
    def test_execution_context_tracking(self):
        """Test that execution context is tracked."""
        sandbox = PluginSandbox()
        
        assert not sandbox.context._is_executing
        
        def check_context():
            assert sandbox.context._is_executing
            return "ok"
        
        result = sandbox.execute(check_context)
        assert result == "ok"
        assert not sandbox.context._is_executing


class TestSecurityProfiles:
    """Test predefined security profiles."""
    
    def test_trusted_profile(self):
        """Test trusted security profile."""
        perms = get_security_profile('trusted')
        
        assert perms.memory_access == PermissionLevel.FULL
        assert perms.network_access == True
        assert perms.llm_access == True
    
    def test_standard_profile(self):
        """Test standard security profile."""
        perms = get_security_profile('standard')
        
        assert perms.memory_access == PermissionLevel.WRITE
        assert perms.network_access == True
        assert perms.llm_access == True
    
    def test_restricted_profile(self):
        """Test restricted security profile."""
        perms = get_security_profile('restricted')
        
        assert perms.memory_access == PermissionLevel.READ
        assert perms.network_access == False
        assert perms.llm_access == False
    
    def test_untrusted_profile(self):
        """Test untrusted security profile."""
        perms = get_security_profile('untrusted')
        
        assert perms.memory_access == PermissionLevel.READ
        assert perms.network_access == False
        assert perms.graph_access == PermissionLevel.NONE
    
    def test_invalid_profile(self):
        """Test that invalid profile raises error."""
        with pytest.raises(ValueError, match="Unknown security profile"):
            get_security_profile('invalid')


class TestSecurityValidation:
    """Test plugin security validation."""
    
    def test_validate_safe_plugin(self):
        """Test validation of a safe plugin."""
        class SafePlugin(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="safe_plugin",
                    version="1.0.0",
                    author="Test",
                    description="Safe plugin",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {"safe": True}
        
        perms = get_security_profile('standard')
        warnings = validate_plugin_security(SafePlugin, perms)
        
        assert len(warnings) == 0
    
    def test_validate_plugin_with_network(self):
        """Test validation detects network usage."""
        class NetworkPlugin(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="network_plugin",
                    version="1.0.0",
                    author="Test",
                    description="Network plugin",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {"data": "from network"}
        
        perms = get_security_profile('restricted')  # No network access
        warnings = validate_plugin_security(NetworkPlugin, perms)
        
        assert len(warnings) > 0
        assert any('network' in w.lower() for w in warnings)


class TestSecurityContext:
    """Test security context tracking."""
    
    def test_context_initialization(self):
        """Test security context initialization."""
        from smartmemory.plugins.security import PluginSecurityContext
        
        perms = PluginPermissions()
        limits = ResourceLimits()
        context = PluginSecurityContext(perms, limits)
        
        assert context.permissions == perms
        assert context.limits == limits
        assert context.start_time is None
        assert context.network_requests_made == 0
    
    def test_execution_tracking(self):
        """Test execution time tracking."""
        from smartmemory.plugins.security import PluginSecurityContext
        
        perms = PluginPermissions()
        limits = ResourceLimits()
        context = PluginSecurityContext(perms, limits)
        
        context.start_execution()
        assert context._is_executing
        assert context.start_time is not None
        
        context.end_execution()
        assert not context._is_executing
    
    def test_network_request_tracking(self):
        """Test network request counting."""
        from smartmemory.plugins.security import PluginSecurityContext
        
        perms = PluginPermissions(network_access=True)
        limits = ResourceLimits(max_network_requests=3)
        context = PluginSecurityContext(perms, limits)
        
        # Should allow up to 3 requests
        context.check_network_request()
        assert context.network_requests_made == 1
        
        context.check_network_request()
        assert context.network_requests_made == 2
        
        context.check_network_request()
        assert context.network_requests_made == 3
        
        # 4th request should fail
        with pytest.raises(ResourceLimitException):
            context.check_network_request()
    
    def test_permission_checks(self):
        """Test permission checking."""
        from smartmemory.plugins.security import PluginSecurityContext
        
        perms = PluginPermissions(
            memory_access=PermissionLevel.READ,
            network_access=False
        )
        limits = ResourceLimits()
        context = PluginSecurityContext(perms, limits)
        
        # Should allow memory read
        context.check_memory_read()  # No exception
        
        # Should deny memory write
        with pytest.raises(PermissionDeniedException):
            context.check_memory_write()
        
        # Should deny network access
        with pytest.raises(PermissionDeniedException):
            context.check_network_request()
