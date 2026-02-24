"""
Comprehensive investigation of ALL SmartMemory plugin architectures.

Tests all plugin types to understand their patterns:
- Extractors: Entity/relation extraction
- Evolvers: Memory evolution (working→episodic→semantic)
- Resolvers: External resolution
- Embedding: Embedding generation

Goal: Document architectural patterns before standardization.
"""

import pytest


pytestmark = pytest.mark.integration
from smartmemory.models.memory_item import MemoryItem


class TestExtractorPluginArchitectures:
    """Test extractor plugins to understand their architectural patterns."""
    
    @pytest.fixture
    def test_content(self):
        """Test content for extraction."""
        return {
            'entity_text': "Albert Einstein was born in Germany and later moved to Princeton University.",
            'relation_text': "The theory of relativity was developed by Einstein in 1905.",
            'complex_text': "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
        }
    
    def test_llm_extractor_architecture(self, test_content):
        """Test LLM extractor - understand its interface and return patterns."""
        try:
            from smartmemory.plugins.extractors.llm import LLMExtractor
            
            # Test instantiation
            extractor = LLMExtractor()
            
            print("✅ LLMExtractor Architecture:")
            print(f"   Class: {extractor.__class__.__name__}")
            print(f"   Module: {extractor.__class__.__module__}")
            
            # Check available methods
            methods = [method for method in dir(extractor) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            # Test if it has extract method
            if hasattr(extractor, 'extract'):
                print("   Has extract method: True")
                # Try to understand method signature
                import inspect
                sig = inspect.signature(extractor.extract)
                print(f"   Extract signature: {sig}")
                
                # Test extraction (may fail due to missing config)
                try:
                    result = extractor.extract(test_content['entity_text'])
                    print(f"   Extract result type: {type(result)}")
                    print(f"   Extract result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                except Exception as e:
                    print(f"   Extract test failed (expected): {type(e).__name__}")
            
        except ImportError as e:
            print(f"❌ LLMExtractor import failed: {e}")
    
    def test_spacy_extractor_architecture(self, test_content):
        """Test SpaCy extractor - understand its interface and return patterns."""
        try:
            from smartmemory.plugins.extractors.spacy import SpacyExtractor
            
            extractor = SpacyExtractor()
            
            print("✅ SpacyExtractor Architecture:")
            print(f"   Class: {extractor.__class__.__name__}")
            
            # Check available methods
            methods = [method for method in dir(extractor) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            # Test extraction interface
            if hasattr(extractor, 'extract'):
                import inspect
                sig = inspect.signature(extractor.extract)
                print(f"   Extract signature: {sig}")
                
                try:
                    result = extractor.extract(test_content['entity_text'])
                    print(f"   Extract result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"   Extract result keys: {list(result.keys())}")
                        # Check for entities/relations structure
                        if 'entities' in result:
                            print(f"   Entities count: {len(result['entities'])}")
                        if 'relations' in result:
                            print(f"   Relations count: {len(result['relations'])}")
                except Exception as e:
                    print(f"   Extract test failed: {type(e).__name__}: {e}")
                    
        except ImportError as e:
            print(f"❌ SpacyExtractor import failed: {e}")
    
    def test_gliner_extractor_architecture(self, test_content):
        """Test GLiNER extractor - understand its interface."""
        try:
            from smartmemory.plugins.extractors.gliner import GLiNERExtractor
            
            extractor = GLiNERExtractor()
            
            print("✅ GLiNERExtractor Architecture:")
            print(f"   Class: {extractor.__class__.__name__}")
            
            methods = [method for method in dir(extractor) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            if hasattr(extractor, 'extract'):
                import inspect
                sig = inspect.signature(extractor.extract)
                print(f"   Extract signature: {sig}")
                
        except ImportError as e:
            print(f"❌ GLiNERExtractor import failed: {e}")


class TestEvolverPluginArchitectures:
    """Test evolver plugins to understand memory evolution patterns."""
    
    @pytest.fixture
    def test_memory_items(self):
        """Test memory items for evolution."""
        return {
            'working': MemoryItem(
                content="I need to remember to buy groceries today.",
                metadata={'user_id': 'test_user', 'memory_type': 'working'}
            ),
            'episodic': MemoryItem(
                content="Yesterday I went to the store and bought apples.",
                metadata={'user_id': 'test_user', 'memory_type': 'episodic'}
            ),
            'semantic': MemoryItem(
                content="Apples are a type of fruit that grows on trees.",
                metadata={'user_id': 'test_user', 'memory_type': 'semantic'}
            )
        }
    
    def test_working_to_episodic_evolver(self, test_memory_items):
        """Test working→episodic evolution pattern."""
        try:
            from smartmemory.plugins.evolvers.working_to_episodic import WorkingToEpisodicEvolver
            
            evolver = WorkingToEpisodicEvolver()
            
            print("✅ WorkingToEpisodicEvolver Architecture:")
            print(f"   Class: {evolver.__class__.__name__}")
            
            methods = [method for method in dir(evolver) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            # Test evolution interface
            if hasattr(evolver, 'evolve'):
                import inspect
                sig = inspect.signature(evolver.evolve)
                print(f"   Evolve signature: {sig}")
                
                try:
                    result = evolver.evolve(test_memory_items['working'])
                    print(f"   Evolve result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"   Evolve result keys: {list(result.keys())}")
                    elif hasattr(result, '__dict__'):
                        print(f"   Evolve result attributes: {list(result.__dict__.keys())}")
                except Exception as e:
                    print(f"   Evolve test failed: {type(e).__name__}: {e}")
                    
        except ImportError as e:
            print(f"❌ WorkingToEpisodicEvolver import failed: {e}")
    
    def test_episodic_to_semantic_evolver(self, test_memory_items):
        """Test episodic→semantic evolution pattern."""
        try:
            from smartmemory.plugins.evolvers.episodic_to_semantic import EpisodicToSemanticEvolver
            
            evolver = EpisodicToSemanticEvolver()
            
            print("✅ EpisodicToSemanticEvolver Architecture:")
            print(f"   Class: {evolver.__class__.__name__}")
            
            methods = [method for method in dir(evolver) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            if hasattr(evolver, 'evolve'):
                import inspect
                sig = inspect.signature(evolver.evolve)
                print(f"   Evolve signature: {sig}")
                
        except ImportError as e:
            print(f"❌ EpisodicToSemanticEvolver import failed: {e}")
    
    def test_episodic_decay_evolver(self, test_memory_items):
        """Test episodic decay pattern."""
        try:
            from smartmemory.plugins.evolvers.episodic_decay import EpisodicDecayEvolver
            
            evolver = EpisodicDecayEvolver()
            
            print("✅ EpisodicDecayEvolver Architecture:")
            print(f"   Class: {evolver.__class__.__name__}")
            
            methods = [method for method in dir(evolver) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
        except ImportError as e:
            print(f"❌ EpisodicDecayEvolver import failed: {e}")


class TestResolverPluginArchitectures:
    """Test resolver plugins to understand resolution patterns."""
    
    def test_external_resolver_architecture(self):
        """Test external resolver pattern."""
        try:
            from smartmemory.plugins.resolvers.external_resolver import ExternalResolver
            
            resolver = ExternalResolver()
            
            print("✅ ExternalResolver Architecture:")
            print(f"   Class: {resolver.__class__.__name__}")
            
            methods = [method for method in dir(resolver) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            # Test resolution interface
            if hasattr(resolver, 'resolve'):
                import inspect
                sig = inspect.signature(resolver.resolve)
                print(f"   Resolve signature: {sig}")
                
        except ImportError as e:
            print(f"❌ ExternalResolver import failed: {e}")


class TestEmbeddingPluginArchitecture:
    """Test embedding plugin architecture."""
    
    def test_embedding_plugin_architecture(self):
        """Test embedding generation pattern."""
        try:
            from smartmemory.plugins.embedding import EmbeddingPlugin
            
            plugin = EmbeddingPlugin()
            
            print("✅ EmbeddingPlugin Architecture:")
            print(f"   Class: {plugin.__class__.__name__}")
            
            methods = [method for method in dir(plugin) if not method.startswith('_')]
            print(f"   Public methods: {methods}")
            
            # Test embedding interface
            if hasattr(plugin, 'generate_embedding'):
                import inspect
                sig = inspect.signature(plugin.generate_embedding)
                print(f"   Generate embedding signature: {sig}")
                
        except ImportError as e:
            print(f"❌ EmbeddingPlugin import failed: {e}")


class TestZettelkastenArchitecture:
    """Test Zettelkasten system architecture."""
    
    def test_zettel_memory_architecture(self):
        """Test ZettelMemory pattern."""
        try:
            # Check if zettelkasten module exists
            
            # Look for zettelkasten in various locations
            possible_locations = [
                'smartmemory.plugins.zettelkasten.zettel_memory',
                'smartmemory.zettelkasten.zettel_memory',
                'smartmemory.memory.zettel_memory'
            ]
            
            zettel_class = None
            for location in possible_locations:
                try:
                    module = __import__(location, fromlist=['ZettelMemory'])
                    zettel_class = getattr(module, 'ZettelMemory', None)
                    if zettel_class:
                        print(f"✅ Found ZettelMemory at: {location}")
                        break
                except ImportError:
                    continue
            
            if zettel_class:
                zettel = zettel_class()
                print("✅ ZettelMemory Architecture:")
                print(f"   Class: {zettel.__class__.__name__}")
                
                methods = [method for method in dir(zettel) if not method.startswith('_')]
                print(f"   Public methods: {methods}")
            else:
                print("❌ ZettelMemory not found in expected locations")
                
        except Exception as e:
            print(f"❌ ZettelMemory investigation failed: {e}")


class TestOntologyArchitecture:
    """Test ontology system architecture."""
    
    def test_ontology_system_architecture(self):
        """Test ontology management pattern."""
        try:
            # Look for ontology modules
            possible_locations = [
                'smartmemory.plugins.ontology',
                'smartmemory.ontology',
                'smartmemory.memory.ontology'
            ]
            
            for location in possible_locations:
                try:
                    module = __import__(location, fromlist=[''])
                    print(f"✅ Found ontology module at: {location}")
                    
                    # List module contents
                    module_contents = [attr for attr in dir(module) if not attr.startswith('_')]
                    print(f"   Module contents: {module_contents}")
                    
                except ImportError:
                    continue
                    
        except Exception as e:
            print(f"❌ Ontology investigation failed: {e}")
