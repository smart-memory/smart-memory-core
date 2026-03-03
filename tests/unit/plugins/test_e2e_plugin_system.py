"""
End-to-end test demonstrating the complete plugin system workflow.
"""
from datetime import UTC

import pytest


pytestmark = pytest.mark.unit
from smartmemory.plugins import (
    PluginMetadata,
    EnricherPlugin,
    EvolverPlugin,
    PluginRegistry,
    PluginManager,
    get_plugin_registry,
    reset_plugin_registry,
)


class TestPluginSystemE2E:
    """End-to-end tests for the complete plugin system."""
    
    def test_complete_plugin_lifecycle(self):
        """Test the complete lifecycle of a plugin from creation to usage."""
        
        # Step 1: Create a custom enricher plugin
        class CustomEnricher(EnricherPlugin):
            """A custom enricher that adds a timestamp."""
            
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="custom_enricher",
                    version="1.0.0",
                    author="Test Developer",
                    description="Adds custom enrichment",
                    plugin_type="enricher",
                    dependencies=["datetime"],
                    tags=["timestamp", "metadata"]
                )
            
            def __init__(self, config=None):
                self.config = config or {}
            
            def enrich(self, item, node_ids=None):
                """Add custom enrichment."""
                from datetime import datetime
                return {
                    "properties": {
                        "enriched_at": datetime.now(UTC).isoformat(),
                        "enricher_version": "1.0.0"
                    },
                    "tags": ["enriched", "custom"]
                }
        
        # Step 2: Create a registry and manager
        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        
        # Step 3: Register the plugin
        metadata = CustomEnricher.metadata()
        registry.register_enricher(metadata.name, CustomEnricher, metadata)
        manager._loaded_plugins.add(metadata.name)
        
        # Step 4: Verify registration
        assert registry.has_plugin("custom_enricher", "enricher")
        assert "custom_enricher" in manager.loaded_plugins
        
        # Step 5: Retrieve and use the plugin
        enricher_class = registry.get_enricher("custom_enricher")
        assert enricher_class is not None
        assert enricher_class == CustomEnricher
        
        # Step 6: Instantiate and use the enricher
        enricher = enricher_class()
        
        # Mock item
        class MockItem:
            content = "Test content"
        
        result = enricher.enrich(MockItem())
        
        # Step 7: Verify enrichment results
        assert "properties" in result
        assert "enriched_at" in result["properties"]
        assert "tags" in result
        assert "enriched" in result["tags"]
        
        # Step 8: Verify metadata is accessible
        plugin_metadata = registry.get_metadata("custom_enricher")
        assert plugin_metadata.name == "custom_enricher"
        assert plugin_metadata.version == "1.0.0"
        assert "timestamp" in plugin_metadata.tags
        
        # Step 9: List all plugins
        all_enrichers = registry.list_plugins("enricher")
        assert "custom_enricher" in all_enrichers
        
        # Step 10: Unload the plugin
        success = manager.unload_plugin("custom_enricher")
        assert success is True
        assert "custom_enricher" not in manager.loaded_plugins
        assert not registry.has_plugin("custom_enricher")
    
    def test_multiple_plugin_types(self):
        """Test managing multiple plugin types simultaneously."""
        
        # Create enricher
        class TestEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="test_enricher",
                    version="1.0.0",
                    author="Test",
                    description="Test enricher",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {"enricher": "data"}
        
        # Create evolver
        class TestEvolver(EvolverPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="test_evolver",
                    version="1.0.0",
                    author="Test",
                    description="Test evolver",
                    plugin_type="evolver"
                )
            
            def evolve(self, memory, logger=None):
                pass
        
        # Register both
        registry = PluginRegistry()
        
        enricher_meta = TestEnricher.metadata()
        registry.register_enricher(enricher_meta.name, TestEnricher, enricher_meta)
        
        evolver_meta = TestEvolver.metadata()
        registry.register_evolver(evolver_meta.name, TestEvolver, evolver_meta)
        
        # Verify both are registered
        assert registry.count() == 2
        assert registry.count("enricher") == 1
        assert registry.count("evolver") == 1
        
        # List by type
        enrichers = registry.list_plugins("enricher")
        evolvers = registry.list_plugins("evolver")
        
        assert "test_enricher" in enrichers
        assert "test_evolver" in evolvers
        assert "test_enricher" not in evolvers
        assert "test_evolver" not in enrichers
        
        # Get all plugins
        all_plugins = registry.list_plugins()
        assert len(all_plugins) == 2
        assert "test_enricher" in all_plugins
        assert "test_evolver" in all_plugins
    
    def test_plugin_with_configuration(self):
        """Test plugin with custom configuration."""
        
        class ConfigurableEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="configurable",
                    version="1.0.0",
                    author="Test",
                    description="Configurable enricher",
                    plugin_type="enricher"
                )
            
            def __init__(self, config=None):
                self.config = config or {}
                self.prefix = self.config.get("prefix", "default")
            
            def enrich(self, item, node_ids=None):
                return {
                    "properties": {
                        "value": f"{self.prefix}_{item.content}"
                    }
                }
        
        # Register plugin
        registry = PluginRegistry()
        metadata = ConfigurableEnricher.metadata()
        registry.register_enricher(metadata.name, ConfigurableEnricher, metadata)
        
        # Use with custom config
        enricher_class = registry.get_enricher("configurable")
        enricher = enricher_class(config={"prefix": "custom"})
        
        class MockItem:
            content = "test"
        
        result = enricher.enrich(MockItem())
        assert result["properties"]["value"] == "custom_test"
        
        # Use with default config
        enricher_default = enricher_class()
        result_default = enricher_default.enrich(MockItem())
        assert result_default["properties"]["value"] == "default_test"
    
    def test_plugin_metadata_validation(self):
        """Test that plugin metadata is properly validated."""
        
        # Valid metadata
        valid_metadata = PluginMetadata(
            name="valid_plugin",
            version="1.0.0",
            author="Test",
            description="Valid",
            plugin_type="enricher"
        )
        assert valid_metadata.name == "valid_plugin"
        
        # Invalid plugin type
        with pytest.raises(ValueError, match="Invalid plugin_type"):
            PluginMetadata(
                name="invalid",
                version="1.0.0",
                author="Test",
                description="Invalid",
                plugin_type="invalid_type"
            )
        
        # Empty name
        with pytest.raises(ValueError, match="name cannot be empty"):
            PluginMetadata(
                name="",
                version="1.0.0",
                author="Test",
                description="Empty name",
                plugin_type="enricher"
            )
    
    def test_global_registry_singleton(self):
        """Test that global registry maintains singleton pattern."""
        reset_plugin_registry()
        
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()
        
        assert registry1 is registry2
        
        # Register in one, should be visible in the other
        class TestEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="singleton_test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {}
        
        metadata = TestEnricher.metadata()
        registry1.register_enricher(metadata.name, TestEnricher, metadata)
        
        assert registry2.has_plugin("singleton_test")
        assert registry2.get_enricher("singleton_test") == TestEnricher


class TestPluginSystemDocumentation:
    """Tests that serve as documentation examples."""
    
    def test_creating_simple_enricher(self):
        """Example: Creating a simple enricher plugin."""
        
        # Define your enricher
        class SentimentEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="sentiment",
                    version="1.0.0",
                    author="Your Name",
                    description="Adds sentiment analysis",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                # Simple sentiment analysis (mock)
                sentiment = "positive" if "good" in item.content.lower() else "neutral"
                return {
                    "properties": {
                        "sentiment": sentiment
                    }
                }
        
        # Use the enricher
        registry = PluginRegistry()
        metadata = SentimentEnricher.metadata()
        registry.register_enricher(metadata.name, SentimentEnricher, metadata)
        
        enricher = SentimentEnricher()
        
        class Item:
            content = "This is a good example"
        
        result = enricher.enrich(Item())
        assert result["properties"]["sentiment"] == "positive"
    
    def test_creating_simple_evolver(self):
        """Example: Creating a simple evolver plugin."""
        
        # Define your evolver
        class SimpleEvolver(EvolverPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="simple_evolver",
                    version="1.0.0",
                    author="Your Name",
                    description="Simple memory evolution",
                    plugin_type="evolver"
                )
            
            def evolve(self, memory, logger=None):
                # Evolution logic here
                if logger:
                    logger.info("Running simple evolution")
        
        # Register and use
        registry = PluginRegistry()
        metadata = SimpleEvolver.metadata()
        registry.register_evolver(metadata.name, SimpleEvolver, metadata)
        
        evolver = SimpleEvolver()
        evolver.evolve(memory=None)  # Would pass actual memory object
