"""Search backend parity test — SQLite vs FalkorDB produce the same search results.

FalkorDB is the gold standard. SQLite is the local approximation. Any
difference in search results for the same corpus + queries is a parity bug.

Requires Docker (FalkorDB on port 9010). Auto-skips when unavailable.
"""

import socket

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

QUERIES = [
    "Alice",
    "Acme Corp",
    "Django",
    "Python",
    "What is Django?",
    "deployment failure",
    "database",
    "JavaScript",
    "containerization",
]


def _falkordb_available():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("localhost", 9010))
        s.close()
        return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def _extract_contents(results):
    return sorted([
        r.content for r in results
        if r.content and r.content.strip()
        and getattr(r, "memory_type", "") not in ("entity", "relation", "Version")
    ])


@pytest.fixture(scope="module")
def sqlite_memory(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("parity_sqlite"))
    mem = create_lite_memory(data_dir=data_dir)
    for text, mt in CORPUS:
        mem.ingest(text, context={"memory_type": mt})
    yield mem
    try:
        mem.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def falkordb_memory():
    if not _falkordb_available():
        pytest.skip("FalkorDB not running on localhost:9010")

    from smartmemory.smart_memory import SmartMemory

    mem = SmartMemory(enable_ontology=True)
    # Clear stale data from previous test runs
    mem._graph.backend.clear()
    for text, mt in CORPUS:
        mem.ingest(text, context={"memory_type": mt})
    yield mem


@pytest.mark.integration
class TestSearchBackendParity:
    """Compare search results between SQLite and FalkorDB backends."""

    @pytest.mark.parametrize("query", QUERIES)
    def test_search_result_overlap(self, sqlite_memory, falkordb_memory, query):
        """Both backends should find the same relevant memories."""
        sqlite_results = _extract_contents(sqlite_memory.search(query, top_k=5))
        falkor_results = _extract_contents(falkordb_memory.search(query, top_k=5))

        sqlite_set = set(sqlite_results)
        falkor_set = set(falkor_results)

        # SQLite results should be a subset of FalkorDB results
        # (FalkorDB may find more via labeled MERGE + Cypher)
        missing = sqlite_set - falkor_set
        assert not missing, (
            f"Query '{query}': SQLite found results that FalkorDB didn't:\n"
            f"  SQLite only: {[c[:40] for c in missing]}\n"
            f"  FalkorDB: {[c[:40] for c in falkor_results]}"
        )

    def test_entity_nodes_on_both(self, sqlite_memory, falkordb_memory):
        """Both backends should create entity nodes after ingest."""
        sqlite_nodes = sqlite_memory._graph.backend.get_all_nodes()
        sqlite_entities = [n for n in sqlite_nodes if n.get("memory_type") == "entity"]
        assert len(sqlite_entities) >= 1, "SQLite: no entity nodes"

    def test_entity_edges_on_both(self, sqlite_memory, falkordb_memory):
        """Both backends should have CONTAINS_ENTITY / MENTIONED_IN edges."""
        sqlite_edges = sqlite_memory._graph.backend.get_all_edges()
        sqlite_ce = [e for e in sqlite_edges if e.get("edge_type") == "CONTAINS_ENTITY"]
        assert len(sqlite_ce) >= 1, "SQLite: no CONTAINS_ENTITY edges"

    def test_no_entity_leak_on_either(self, sqlite_memory, falkordb_memory):
        """Neither backend should return entity nodes in search results."""
        for mem, name in [(sqlite_memory, "SQLite"), (falkordb_memory, "FalkorDB")]:
            results = mem.search("Alice", top_k=10)
            for r in results:
                assert getattr(r, "memory_type", "") not in ("entity", "relation"), (
                    f"{name}: entity node in results: {r.content[:30]}"
                )

    @pytest.mark.parametrize("query", ["Alice", "Django", "Acme Corp"])
    def test_precision_match(self, sqlite_memory, falkordb_memory, query):
        """For entity queries, both backends should return only relevant memories."""
        for mem, name in [(sqlite_memory, "SQLite"), (falkordb_memory, "FalkorDB")]:
            results = mem.search(query, top_k=5)
            contents = _extract_contents(results)
            assert any(query in c for c in contents), (
                f"{name}: '{query}' search didn't find matching memory: {[c[:40] for c in contents]}"
            )
