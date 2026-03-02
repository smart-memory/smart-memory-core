"""Tests for smartmemory/relations/discovery.py (CORE-EXT-1c)."""

from unittest.mock import MagicMock, call

from smartmemory.relations.discovery import (
    RELOAD_CHANNEL_PREFIX,
    RelationCandidate,
    RelationCluster,
    RelationDiscoveryService,
)


class TestRelationCluster:
    def test_construction(self):
        cluster = RelationCluster(
            centroid_label="supervises",
            members=["supervises", "oversees"],
            total_frequency=7,
        )
        assert cluster.centroid_label == "supervises"
        assert len(cluster.members) == 2
        assert cluster.total_frequency == 7

    def test_defaults(self):
        cluster = RelationCluster(centroid_label="x")
        assert cluster.members == []
        assert cluster.total_frequency == 0
        assert cluster.inferred_type_pairs == []
        assert cluster.embedding_coherence == 0.0


class TestRelationCandidate:
    def test_construction(self):
        candidate = RelationCandidate(
            proposed_name="supervises",
            category="discovered",
            aliases=["oversees"],
            type_pairs=[("person", "person")],
            confidence=0.95,
        )
        assert candidate.proposed_name == "supervises"
        assert candidate.status == "proposed"

    def test_defaults(self):
        candidate = RelationCandidate(proposed_name="x")
        assert candidate.category == "custom"
        assert candidate.aliases == []
        assert candidate.type_pairs == []
        assert candidate.confidence == 0.0
        assert candidate.status == "proposed"


class TestRelationDiscoveryService:
    def _mock_ontology(self, labels=None):
        og = MagicMock()
        og.get_novel_relation_labels.return_value = labels or []
        og.promote_relation_type.return_value = True
        return og

    def test_cluster_string_equality(self):
        """Without embedding_fn, labels cluster by normalized key."""
        labels = [
            {"name": "supervises", "frequency": 5, "source_types": ["person"], "target_types": ["person"]},
            {"name": "SUPERVISES", "frequency": 3, "source_types": ["person"], "target_types": ["person"]},
        ]
        og = self._mock_ontology(labels)
        service = RelationDiscoveryService(og)
        clusters = service.cluster_novel_labels(workspace_id="test", min_frequency=1)
        assert len(clusters) == 1
        assert clusters[0].total_frequency == 8
        assert clusters[0].centroid_label == "supervises"

    def test_cluster_distinct_labels(self):
        """Different normalized keys produce separate clusters."""
        labels = [
            {"name": "supervises", "frequency": 5, "source_types": [], "target_types": []},
            {"name": "mentors", "frequency": 3, "source_types": [], "target_types": []},
        ]
        og = self._mock_ontology(labels)
        service = RelationDiscoveryService(og)
        clusters = service.cluster_novel_labels(workspace_id="test", min_frequency=1)
        assert len(clusters) == 2

    def test_propose_filters_low_frequency(self):
        """Clusters below min_cluster_frequency are excluded from candidates."""
        clusters = [
            RelationCluster(centroid_label="high", total_frequency=10, embedding_coherence=1.0),
            RelationCluster(centroid_label="low", total_frequency=2, embedding_coherence=1.0),
        ]
        service = RelationDiscoveryService(self._mock_ontology())
        candidates = service.propose_candidates(clusters, min_cluster_frequency=5)
        assert len(candidates) == 1
        assert candidates[0].proposed_name == "high"

    def test_propose_sets_aliases_from_members(self):
        """Members other than centroid become aliases."""
        clusters = [
            RelationCluster(
                centroid_label="supervises",
                members=["supervises", "oversees", "manages_directly"],
                total_frequency=10,
                embedding_coherence=1.0,
            ),
        ]
        service = RelationDiscoveryService(self._mock_ontology())
        candidates = service.propose_candidates(clusters, min_cluster_frequency=5)
        assert set(candidates[0].aliases) == {"oversees", "manages_directly"}

    def test_auto_promote_writes_to_ontology(self):
        """Verifies promote_relation_type() is called for qualifying candidates."""
        og = self._mock_ontology()
        service = RelationDiscoveryService(og)
        candidates = [
            RelationCandidate(
                proposed_name="supervises",
                aliases=["oversees"],
                type_pairs=[("person", "person")],
                confidence=0.9,
                total_frequency=15,
            ),
        ]
        count = service.auto_promote(candidates, workspace_id="test", min_frequency=10)
        assert count == 1
        og.promote_relation_type.assert_called_once_with(
            name="supervises",
            category="custom",
            aliases=["oversees"],
            type_pairs=[("person", "person")],
        )
        assert candidates[0].status == "promoted"

    def test_auto_promote_skips_low_confidence(self):
        """Candidates with confidence < 0.5 are not promoted."""
        og = self._mock_ontology()
        service = RelationDiscoveryService(og)
        candidates = [
            RelationCandidate(proposed_name="dubious", confidence=0.3, total_frequency=20),
        ]
        count = service.auto_promote(candidates, workspace_id="test")
        assert count == 0
        og.promote_relation_type.assert_not_called()

    def test_auto_promote_skips_low_frequency(self):
        """Candidates below min_frequency are not promoted even with high confidence."""
        og = self._mock_ontology()
        service = RelationDiscoveryService(og)
        candidates = [
            RelationCandidate(proposed_name="rare_pred", confidence=0.9, total_frequency=3),
        ]
        count = service.auto_promote(candidates, workspace_id="test", min_frequency=10)
        assert count == 0
        og.promote_relation_type.assert_not_called()

    def test_auto_promote_calls_reload_fn(self):
        """Verifies callback is invoked after promotions."""
        og = self._mock_ontology()
        reload_fn = MagicMock()
        service = RelationDiscoveryService(og)
        candidates = [
            RelationCandidate(proposed_name="x", confidence=0.9, total_frequency=15),
        ]
        service.auto_promote(candidates, workspace_id="test", normalizer_reload_fn=reload_fn, min_frequency=1)
        reload_fn.assert_called_once()

    def test_auto_promote_publishes_redis(self):
        """With mock redis_client, verify publish() called with correct channel."""
        og = self._mock_ontology()
        redis = MagicMock()
        service = RelationDiscoveryService(og, redis_client=redis)
        candidates = [
            RelationCandidate(proposed_name="x", confidence=0.9, total_frequency=15),
        ]
        service.auto_promote(candidates, workspace_id="ws123", min_frequency=1)
        redis.publish.assert_called_once_with(f"{RELOAD_CHANNEL_PREFIX}:ws123", "reload")

    def test_auto_promote_no_redis_no_error(self):
        """Without redis_client, promotion still succeeds."""
        og = self._mock_ontology()
        service = RelationDiscoveryService(og, redis_client=None)
        candidates = [
            RelationCandidate(proposed_name="x", confidence=0.9, total_frequency=15),
        ]
        count = service.auto_promote(candidates, workspace_id="test", min_frequency=1)
        assert count == 1

    def test_empty_labels_returns_empty(self):
        """No labels above threshold → empty clusters."""
        og = self._mock_ontology([])
        service = RelationDiscoveryService(og)
        clusters = service.cluster_novel_labels()
        assert clusters == []
