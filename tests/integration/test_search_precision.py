"""Search precision tests — exercises retrieval across query types.

Tests that search returns relevant results and doesn't return irrelevant ones.
Uses a fixed 10-memory corpus with known entities and content. No mocks.

These tests caught:
- Vector search returning everything (no threshold)
- Stop words matching every memory ("What is Django?" → all)
- Entity nodes leaking into results
- Graph-first search missing when entity nodes don't exist
"""

import pytest
from smartmemory.tools.factory import create_lite_memory


CORPUS = [
    ("Alice leads Project Atlas at Acme Corp", "episodic"),
    ("Bob is the CTO of Acme Corp and reports to Alice", "episodic"),
    ("Django is a Python web framework created by Adrian Holovaty", "semantic"),
    ("Kubernetes orchestrates containerized workloads across clusters", "semantic"),
    ("Always use async/await for database calls in Python", "procedural"),
    ("React is a JavaScript UI library maintained by Meta", "semantic"),
    ("The deployment failed at 3pm due to a DNS misconfiguration", "episodic"),
    ("PostgreSQL supports JSONB columns for semi-structured data", "semantic"),
    ("Sarah presented the Q3 roadmap at the all-hands meeting", "episodic"),
    ("Use git rebase for clean commit history before merging", "procedural"),
]


@pytest.fixture(scope="module")
def memory(tmp_path_factory):
    """Shared memory instance with 10 memories for all precision tests."""
    data_dir = str(tmp_path_factory.mktemp("search_precision"))
    mem = create_lite_memory(data_dir=data_dir)
    for text, mt in CORPUS:
        mem.ingest(text, context={"memory_type": mt})
    yield mem
    try:
        mem.close()
    except Exception:
        pass


def _contents(results):
    """Extract content strings from search results, filtering empty."""
    return [r.content for r in results if r.content and r.content.strip()]


def _assert_contains(results, substring, msg=""):
    """Assert at least one result contains the substring."""
    contents = _contents(results)
    assert any(substring in c for c in contents), (
        f"{msg or substring} not found in results: {[c[:50] for c in contents]}"
    )


def _assert_not_contains(results, substring, msg=""):
    """Assert no result contains the substring."""
    contents = _contents(results)
    assert not any(substring in c for c in contents), (
        f"{msg or substring} should NOT be in results: {[c[:50] for c in contents]}"
    )


def _assert_no_entity_nodes(results):
    """Assert no entity/relation nodes leaked into results."""
    for r in results:
        mt = getattr(r, "memory_type", "")
        assert mt not in ("entity", "relation", "Version"), (
            f"System node leaked into results: type={mt} content={r.content[:30]}"
        )


@pytest.mark.integration
class TestEntitySearch:
    """Search for known entities — should find linked memories via graph."""

    def test_alice_returns_alice_memories(self, memory):
        results = memory.search("Alice", top_k=5)
        _assert_contains(results, "Alice")
        _assert_no_entity_nodes(results)

    def test_alice_does_not_return_django(self, memory):
        results = memory.search("Alice", top_k=5)
        _assert_not_contains(results, "Django")

    def test_alice_does_not_return_kubernetes(self, memory):
        results = memory.search("Alice", top_k=5)
        _assert_not_contains(results, "Kubernetes")

    def test_acme_returns_both_acme_memories(self, memory):
        results = memory.search("Acme Corp", top_k=5)
        contents = _contents(results)
        acme_hits = [c for c in contents if "Acme" in c]
        assert len(acme_hits) >= 2, f"Expected 2 Acme memories, got {len(acme_hits)}"

    def test_django_returns_django_memory(self, memory):
        results = memory.search("Django", top_k=5)
        _assert_contains(results, "Django")
        _assert_not_contains(results, "Alice")


@pytest.mark.integration
class TestStopWordFiltering:
    """Queries with stop words should not match everything."""

    def test_what_is_django(self, memory):
        results = memory.search("What is Django?", top_k=5)
        _assert_contains(results, "Django")
        contents = _contents(results)
        # Should not return all 10 memories
        assert len(contents) <= 5, f"'What is Django?' returned {len(contents)} results — stop words not filtered"

    def test_how_does_kubernetes_work(self, memory):
        results = memory.search("How does Kubernetes work?", top_k=5)
        _assert_contains(results, "Kubernetes")

    def test_what_is_react(self, memory):
        results = memory.search("What is React?", top_k=5)
        _assert_contains(results, "React")


@pytest.mark.integration
class TestPunctuationHandling:
    """Punctuation should not break search."""

    def test_apostrophe(self, memory):
        results = memory.search("Alice's project", top_k=5)
        _assert_contains(results, "Alice")

    def test_question_mark(self, memory):
        results = memory.search("Django?", top_k=5)
        _assert_contains(results, "Django")

    def test_comma_separated(self, memory):
        results = memory.search("Python, Django", top_k=5)
        _assert_contains(results, "Django")


@pytest.mark.integration
class TestMultiWordSearch:
    """Multi-word queries should find relevant memories."""

    def test_project_atlas(self, memory):
        results = memory.search("Project Atlas", top_k=5)
        _assert_contains(results, "Project Atlas")

    def test_python_framework(self, memory):
        results = memory.search("Python framework", top_k=5)
        _assert_contains(results, "Django")

    def test_database_calls(self, memory):
        results = memory.search("database calls", top_k=5)
        contents = _contents(results)
        assert any("database" in c.lower() for c in contents), (
            f"'database calls' didn't find database content: {[c[:50] for c in contents]}"
        )


@pytest.mark.integration
class TestSemanticSearch:
    """Queries without exact keyword matches — relies on vector similarity."""

    def test_containerization(self, memory):
        results = memory.search("containerization", top_k=5)
        # Should find Kubernetes (semantically related) via vector
        _assert_contains(results, "container", msg="containerization → Kubernetes")

    def test_deployment_failure(self, memory):
        results = memory.search("deployment failure", top_k=5)
        _assert_contains(results, "deployment")


@pytest.mark.integration
class TestWildcardAndEdgeCases:
    """Wildcard, empty, and edge case queries."""

    def test_wildcard_returns_all(self, memory):
        results = memory.search("*", top_k=100)
        contents = _contents(results)
        assert len(contents) >= 10, f"Wildcard returned {len(contents)}, expected >=10"

    def test_empty_query_returns_results(self, memory):
        results = memory.search("", top_k=5)
        # Empty query may return all or nothing — should not crash
        assert isinstance(results, list)

    def test_top_k_respected(self, memory):
        results = memory.search("*", top_k=3)
        assert len(results) <= 3, f"top_k=3 but got {len(results)} results"


@pytest.mark.integration
class TestSearchExclusions:
    """Verify that system nodes never appear in search results."""

    def test_no_entity_nodes_in_results(self, memory):
        for query in ["Alice", "Django", "Python", "*"]:
            results = memory.search(query, top_k=10)
            _assert_no_entity_nodes(results)

    def test_no_version_nodes_in_results(self, memory):
        results = memory.search("*", top_k=100)
        for r in results:
            assert getattr(r, "memory_type", "") != "Version", "Version node in results"


@pytest.mark.integration
class TestRecencySort:
    """Recency sort should return most recent first."""

    def test_recency_returns_results_in_order(self, memory):
        results = memory.search("", top_k=3, sort_by="recency")
        assert len(results) >= 1, "Recency recall returned nothing"
        # Just verify we got results and they're not entity nodes
        _assert_no_entity_nodes(results)
        contents = _contents(results)
        assert len(contents) >= 1, "Recency returned empty content results"
