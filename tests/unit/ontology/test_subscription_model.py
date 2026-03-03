"""Tests for OntologySubscription and Ontology serialization with layer fields."""

import pytest

pytestmark = pytest.mark.unit


from datetime import datetime, UTC

from smartmemory.ontology.models import Ontology, OntologySubscription


class TestOntologySubscriptionRoundtrip:
    def test_roundtrip_basic(self):
        sub = OntologySubscription(
            base_registry_id="base-123",
            subscribed_by="user1",
        )
        d = sub.to_dict()
        restored = OntologySubscription.from_dict(d)

        assert restored.base_registry_id == "base-123"
        assert restored.pinned_version is None
        assert restored.hidden_types == set()
        assert restored.subscribed_by == "user1"

    def test_roundtrip_with_pinned_version(self):
        sub = OntologySubscription(
            base_registry_id="base-123",
            pinned_version="2.0.0",
        )
        d = sub.to_dict()
        restored = OntologySubscription.from_dict(d)
        assert restored.pinned_version == "2.0.0"

    def test_hidden_types_serialized_as_sorted_list(self):
        sub = OntologySubscription(
            base_registry_id="base-123",
            hidden_types={"zebra", "animal", "plant"},
        )
        d = sub.to_dict()
        assert d["hidden_types"] == ["animal", "plant", "zebra"]

    def test_hidden_types_deserialized_to_set(self):
        d = {
            "base_registry_id": "base-123",
            "hidden_types": ["animal", "plant"],
            "subscribed_at": datetime.now(UTC).isoformat(),
        }
        restored = OntologySubscription.from_dict(d)
        assert isinstance(restored.hidden_types, set)
        assert restored.hidden_types == {"animal", "plant"}


class TestOntologySerializationWithLayers:
    def test_roundtrip_without_subscription(self):
        o = Ontology("test", "1.0.0")
        d = o.to_dict()
        restored = Ontology.from_dict(d)
        assert restored.subscription is None
        assert restored.is_base_layer is False

    def test_roundtrip_with_subscription(self):
        o = Ontology("test", "1.0.0")
        o.subscription = OntologySubscription(
            base_registry_id="base-123",
            pinned_version="2.0.0",
            hidden_types={"animal"},
            subscribed_by="user1",
        )
        o.is_base_layer = True

        d = o.to_dict()
        assert "subscription" in d
        assert d["is_base_layer"] is True

        restored = Ontology.from_dict(d)
        assert restored.subscription is not None
        assert restored.subscription.base_registry_id == "base-123"
        assert restored.subscription.pinned_version == "2.0.0"
        assert restored.subscription.hidden_types == {"animal"}
        assert restored.is_base_layer is True

    def test_backward_compatible_missing_subscription(self):
        """Older serialized ontologies won't have subscription key."""
        o = Ontology("test")
        d = o.to_dict()
        # Simulate old format by removing new fields
        del d["is_base_layer"]
        # subscription key is only present when not None, so it's already missing

        restored = Ontology.from_dict(d)
        assert restored.subscription is None
        assert restored.is_base_layer is False

    def test_subscription_none_not_serialized(self):
        o = Ontology("test")
        d = o.to_dict()
        assert "subscription" not in d
