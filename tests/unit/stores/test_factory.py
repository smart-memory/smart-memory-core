"""Unit tests for stores.factory — StoreFactory and StoreType."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.stores.factory import StoreFactory, StoreType


@pytest.fixture(autouse=True)
def _clean_factory_registry():
    """Snapshot and restore the factory registry around each test."""
    snapshot = dict(StoreFactory._store_registry)
    yield
    StoreFactory._store_registry.clear()
    StoreFactory._store_registry.update(snapshot)


class TestStoreType:
    def test_enum_values(self):
        assert StoreType.GRAPH.value == "graph"
        assert StoreType.VECTOR.value == "vector"
        assert StoreType.SEMANTIC_GRAPH.value == "semantic_graph"
        assert StoreType.EPISODIC_GRAPH.value == "episodic_graph"
        assert StoreType.PROCEDURAL_GRAPH.value == "procedural_graph"
        assert StoreType.ZETTEL_GRAPH.value == "zettel_graph"


def _mock_store_cls(return_value="instance", side_effect=None):
    """Create a mock class with __name__ so register_store_type's log works."""
    cls = MagicMock(return_value=return_value, side_effect=side_effect)
    cls.__name__ = "MockStoreClass"
    return cls


class TestStoreFactoryRegistration:
    def test_register_and_check(self):
        StoreFactory.register_store_type(StoreType.GRAPH, _mock_store_cls())
        assert StoreFactory.is_type_registered(StoreType.GRAPH)

    def test_unregistered_type(self):
        StoreFactory._store_registry.clear()
        assert not StoreFactory.is_type_registered(StoreType.GRAPH)

    def test_get_registered_types(self):
        StoreFactory._store_registry.clear()
        StoreFactory.register_store_type(StoreType.GRAPH, _mock_store_cls())
        StoreFactory.register_store_type(StoreType.VECTOR, _mock_store_cls())
        types = StoreFactory.get_registered_types()
        assert StoreType.GRAPH in types
        assert StoreType.VECTOR in types


class TestStoreFactoryCreate:
    def test_create_store_unknown_type_raises(self):
        StoreFactory._store_registry.clear()
        with pytest.raises(ValueError, match="Unknown store type"):
            StoreFactory.create_store(StoreType.GRAPH)

    def test_create_store_calls_class(self):
        mock_cls = _mock_store_cls("instance")
        StoreFactory.register_store_type(StoreType.VECTOR, mock_cls)
        result = StoreFactory.create_store(StoreType.VECTOR)
        assert result == "instance"

    def test_create_store_with_config(self):
        mock_cls = _mock_store_cls("configured")
        StoreFactory.register_store_type(StoreType.SEMANTIC_GRAPH, mock_cls)
        config = MagicMock()
        result = StoreFactory.create_store(StoreType.SEMANTIC_GRAPH, config=config)
        assert result == "configured"
        mock_cls.assert_called_once_with(config=config)

    def test_create_store_failure_raises_runtime(self):
        mock_cls = _mock_store_cls(side_effect=ConnectionError("no DB"))
        StoreFactory.register_store_type(StoreType.GRAPH, mock_cls)
        with pytest.raises(RuntimeError, match="Store creation failed"):
            StoreFactory.create_store(StoreType.GRAPH)


class TestStoreFactoryConvenienceMethods:
    def test_create_graph_store(self):
        StoreFactory.register_store_type(StoreType.GRAPH, _mock_store_cls("graph"))
        assert StoreFactory.create_graph_store() == "graph"

    def test_create_vector_store(self):
        StoreFactory.register_store_type(StoreType.VECTOR, _mock_store_cls("vector"))
        assert StoreFactory.create_vector_store() == "vector"


class TestStoreFactoryValidation:
    def test_validate_config_none_ok(self):
        # Should not raise
        StoreFactory._validate_config(StoreType.GRAPH, None)

    def test_validate_config_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid config"):
            StoreFactory._validate_config(StoreType.GRAPH, 42)

    def test_validate_config_dict_ok(self):
        StoreFactory._validate_config(StoreType.GRAPH, {"key": "val"})

    def test_validate_config_object_ok(self):
        config = MagicMock()
        StoreFactory._validate_config(StoreType.VECTOR, config)


class TestCreateAllGraphStores:
    def test_batch_creation(self):
        for st in [StoreType.SEMANTIC_GRAPH, StoreType.EPISODIC_GRAPH,
                    StoreType.PROCEDURAL_GRAPH, StoreType.ZETTEL_GRAPH]:
            StoreFactory.register_store_type(st, _mock_store_cls(f"{st.value}_inst"))
        stores = StoreFactory.create_all_graph_stores()
        assert len(stores) == 4

    def test_batch_skips_failures(self):
        StoreFactory.register_store_type(StoreType.SEMANTIC_GRAPH, _mock_store_cls("ok"))
        StoreFactory.register_store_type(StoreType.EPISODIC_GRAPH, _mock_store_cls(side_effect=RuntimeError("fail")))
        stores = StoreFactory.create_all_graph_stores()
        assert StoreType.SEMANTIC_GRAPH in stores
        assert StoreType.EPISODIC_GRAPH not in stores
