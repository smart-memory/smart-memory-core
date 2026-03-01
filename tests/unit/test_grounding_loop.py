"""Tests for SPARQL client circuit breaker and PublicKnowledgeGrounder.

Circuit breaker tests (Task 4):
- State transitions: CLOSED → OPEN → HALF_OPEN → CLOSED
- Negative cache hit/miss

Grounding loop tests (Task 6, added later):
- Alias hit grounds correctly
- SPARQL fallback absorbs + grounds
- Circuit-open skips SPARQL
- Ungrounded logging
"""

import time
from unittest.mock import MagicMock

import pytest

from smartmemory.grounding.sparql_client import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Circuit breaker: CLOSED → OPEN (5 failures) → HALF_OPEN (recovery) → CLOSED."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_allows_request_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_single_failure_stays_closed(self):
        cb = CircuitBreaker()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_blocks_request_when_open(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.allow_request() is False

    def test_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_closes_from_half_open(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_reopens_from_half_open(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        cb.allow_request()  # transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should be back to 0 failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # 2 failures, not 3

    def test_default_threshold_is_five(self):
        cb = CircuitBreaker()
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestNegativeCache:
    """In-memory negative cache for SPARQL misses."""

    def test_miss_not_cached(self):
        from smartmemory.grounding.sparql_client import WDQSClient
        client = WDQSClient()
        assert client._is_negative_cached("Python") is False

    def test_miss_cached_after_set(self):
        from smartmemory.grounding.sparql_client import WDQSClient
        client = WDQSClient()
        client._set_negative_cache("FooBarBaz")
        assert client._is_negative_cached("FooBarBaz") is True

    def test_different_surface_not_cached(self):
        from smartmemory.grounding.sparql_client import WDQSClient
        client = WDQSClient()
        client._set_negative_cache("FooBarBaz")
        assert client._is_negative_cached("Other") is False


# --- Task 6: PublicKnowledgeGrounder tests ---

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.public_knowledge_grounder import PublicKnowledgeGrounder
from smartmemory.grounding.sqlite_store import SQLitePublicKnowledgeStore


@pytest.fixture
def test_store(tmp_path):
    """SQLite store pre-loaded with test entities."""
    store = SQLitePublicKnowledgeStore(str(tmp_path / "test.sqlite"))
    store.absorb(PublicEntity(
        qid="Q28865",
        label="Python",
        aliases=["Python programming language"],
        description="General-purpose programming language",
        entity_type="Technology",
        instance_of=["Q9143"],
        domain="software",
        confidence=0.95,
    ))
    store.absorb(PublicEntity(
        qid="Q2622004",
        label="Django",
        aliases=["Django framework"],
        description="Web framework for Python",
        entity_type="Technology",
        instance_of=["Q271680"],
        domain="software",
        confidence=0.9,
    ))
    return store


class TestGroundingAliasLookup:
    """Grounding via alias lookup (no SPARQL needed)."""

    def test_single_candidate_grounds(self, test_store):
        graph = MagicMock()
        graph.get_node.return_value = None  # not already grounded
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "Django", "entity_type": "technology", "item_id": "entity:1"}]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) > 0
        # Should have created a wikidata:Q2622004 node
        graph.add_node.assert_called()
        call_kwargs = graph.add_node.call_args
        assert "wikidata:Q2622004" in str(call_kwargs)

    def test_already_grounded_reuses_node(self, test_store):
        graph = MagicMock()
        graph.get_node.return_value = {"qid": "Q2622004"}  # already exists
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "Django", "entity_type": "technology", "item_id": "entity:1"}]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) > 0
        graph.add_node.assert_not_called()
        assert grounder._graph_hits == 1

    def test_empty_entities_returns_empty(self, test_store):
        graph = MagicMock()
        grounder = PublicKnowledgeGrounder(test_store)
        result = grounder.ground(MagicMock(), [], graph)
        assert result == []

    def test_no_match_logs_ungrounded(self, test_store):
        graph = MagicMock()
        graph.get_node.return_value = None
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "NonexistentThing", "entity_type": "concept", "item_id": "entity:2"}]
        result = grounder.ground(MagicMock(), entities, graph)
        # Should not crash; returns empty (no match, no SPARQL)
        assert len(result) == 0
        # Should have a grounding decision logged
        assert len(grounder.decisions) == 1
        assert grounder.decisions[0].source == "ungrounded"

    def test_graph_hits_counter(self, test_store):
        grounder = PublicKnowledgeGrounder(test_store)
        assert grounder._graph_hits == 0

    def test_edge_created_for_grounded_entity(self, test_store):
        graph = MagicMock()
        graph.get_node.return_value = None
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "Django", "entity_type": "technology", "item_id": "entity:1"}]
        grounder.ground(MagicMock(), entities, graph)
        graph.add_edge.assert_called()
        call_args = graph.add_edge.call_args
        assert call_args[1].get("is_global") is True or (len(call_args[0]) > 3 and call_args[0][3] is True)


class TestDisambiguation:
    """Multi-candidate disambiguation in _select_candidate."""

    def test_single_low_confidence_still_selected(self, test_store):
        """Single candidate with confidence <= 0.8 falls through to confidence ranking."""
        graph = MagicMock()
        graph.get_node.return_value = None
        test_store.absorb(PublicEntity(
            qid="Q999", label="WeakEntity", entity_type="Concept", confidence=0.5,
        ))
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "WeakEntity", "entity_type": "concept", "item_id": "e:10"}]
        result = grounder.ground(MagicMock(), entities, graph)
        # Should still ground — confidence ranking picks the only candidate
        assert len(result) == 1
        assert "wikidata:Q999" in result[0]

    def test_multi_candidate_type_match_wins(self, test_store):
        """When multiple candidates exist, type-compatible one wins."""
        graph = MagicMock()
        graph.get_node.return_value = None
        # Add two entities with same label but different types
        test_store.absorb(PublicEntity(
            qid="Q190391", label="Python", aliases=["Python"],
            entity_type="Concept", confidence=0.85, domain="biology",
        ))
        # Q28865 "Python" Technology already in store from fixture
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "Python", "entity_type": "Technology", "item_id": "e:11"}]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) == 1
        assert "Q28865" in result[0]  # Technology match wins

    def test_multi_candidate_no_type_context(self, test_store):
        """No entity_type context — highest confidence wins."""
        graph = MagicMock()
        graph.get_node.return_value = None
        test_store.absorb(PublicEntity(
            qid="Q190391", label="Python", aliases=["Python"],
            entity_type="Concept", confidence=0.5, domain="biology",
        ))
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "Python", "entity_type": "", "item_id": "e:12"}]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) == 1
        assert "Q28865" in result[0]  # Higher confidence (0.95 > 0.5)

    def test_multi_candidate_no_type_match_fallback(self, test_store):
        """Type filter finds no match — falls back to confidence ranking."""
        graph = MagicMock()
        graph.get_node.return_value = None
        test_store.absorb(PublicEntity(
            qid="Q190391", label="Python", aliases=["Python"],
            entity_type="Concept", confidence=0.5, domain="biology",
        ))
        grounder = PublicKnowledgeGrounder(test_store)
        # Request type "location" — neither candidate matches
        entities = [{"name": "Python", "entity_type": "Location", "item_id": "e:13"}]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) == 1
        assert "Q28865" in result[0]  # Highest confidence wins as fallback

    def test_entity_without_item_id_skips_edge(self, test_store):
        """Entity with no item_id logs warning but doesn't crash."""
        graph = MagicMock()
        graph.get_node.return_value = None
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"name": "Django", "entity_type": "technology"}]  # no item_id
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) == 1
        # Edge should NOT be created (no source node)
        # But wikidata node should still be created
        graph.add_node.assert_called()

    def test_entity_with_metadata_name(self, test_store):
        """Entity with name in metadata dict is extracted correctly."""
        graph = MagicMock()
        graph.get_node.return_value = None
        grounder = PublicKnowledgeGrounder(test_store)
        entities = [{"metadata": {"name": "Django"}, "item_id": "e:14"}]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) == 1
        assert "Q2622004" in result[0]

    def test_object_entity_with_metadata(self, test_store):
        """Non-dict entity with metadata attribute is extracted."""
        graph = MagicMock()
        graph.get_node.return_value = None
        grounder = PublicKnowledgeGrounder(test_store)

        class FakeEntity:
            metadata = {"name": "Django"}
            item_id = "e:15"

        entities = [FakeEntity()]
        result = grounder.ground(MagicMock(), entities, graph)
        assert len(result) == 1
        assert "Q2622004" in result[0]


class TestGroundingSPARQLFallback:
    """SPARQL fallback when alias lookup misses."""

    def test_sparql_fallback_absorbs_on_hit(self, test_store):
        """When SPARQL finds an entity, it should absorb into the store."""
        sparql = MagicMock()
        sparql._circuit = MagicMock()
        sparql._circuit.allow_request.return_value = True
        sparql.query_entity.return_value = [
            PublicEntity(qid="Q79", label="Rust", entity_type="Technology", confidence=0.9)
        ]
        graph = MagicMock()
        graph.get_node.return_value = None

        grounder = PublicKnowledgeGrounder(test_store, sparql_client=sparql)
        entities = [{"name": "Rust", "entity_type": "technology", "item_id": "entity:3"}]
        result = grounder.ground(MagicMock(), entities, graph)

        assert len(result) > 0
        # Rust should have been absorbed into the store
        assert test_store.lookup_by_qid("Q79") is not None

    def test_sparql_fallback_passes_type_filter(self, test_store):
        """SPARQL query must include type_filter_qids from extraction context."""
        sparql = MagicMock()
        sparql._circuit = MagicMock()
        sparql._circuit.allow_request.return_value = True
        sparql.query_entity.return_value = [
            PublicEntity(qid="Q79", label="Rust", entity_type="Technology", confidence=0.9)
        ]
        graph = MagicMock()
        graph.get_node.return_value = None

        grounder = PublicKnowledgeGrounder(test_store, sparql_client=sparql)
        entities = [{"name": "Rust", "entity_type": "Technology", "item_id": "entity:3"}]
        grounder.ground(MagicMock(), entities, graph)

        # query_entity must be called with type_filter_qids for Technology
        call_args = sparql.query_entity.call_args
        assert call_args is not None
        type_filter = call_args.kwargs.get("type_filter_qids") or (call_args.args[1] if len(call_args.args) > 1 else None)
        assert type_filter is not None, "SPARQL fallback must pass type_filter_qids"
        assert "Q9143" in type_filter  # programming language → Technology

    def test_sparql_fallback_no_type_filter_when_no_entity_type(self, test_store):
        """When entity has no type context, SPARQL should query without type filter."""
        sparql = MagicMock()
        sparql._circuit = MagicMock()
        sparql._circuit.allow_request.return_value = True
        sparql.query_entity.return_value = []
        graph = MagicMock()
        graph.get_node.return_value = None

        grounder = PublicKnowledgeGrounder(test_store, sparql_client=sparql)
        entities = [{"name": "Unknown", "item_id": "entity:5"}]  # no entity_type
        grounder.ground(MagicMock(), entities, graph)

        call_args = sparql.query_entity.call_args
        assert call_args is not None
        type_filter = call_args.kwargs.get("type_filter_qids")
        assert type_filter is None, "No type filter when entity has no type context"

    def test_circuit_open_skips_sparql(self, test_store):
        """When circuit is open, SPARQL is not called."""
        sparql = MagicMock()
        sparql._circuit = MagicMock()
        sparql._circuit.allow_request.return_value = False
        sparql.query_entity.return_value = []
        graph = MagicMock()
        graph.get_node.return_value = None

        grounder = PublicKnowledgeGrounder(test_store, sparql_client=sparql)
        entities = [{"name": "Unknown", "entity_type": "concept", "item_id": "entity:4"}]
        result = grounder.ground(MagicMock(), entities, graph)

        sparql.query_entity.assert_not_called()
        assert len(result) == 0
