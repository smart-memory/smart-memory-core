import pytest
from unittest.mock import Mock

pytestmark = pytest.mark.unit


def _make_evolver(config=None):
    from smartmemory.plugins.evolvers.stale_memory import StaleMemoryEvolver, StaleMemoryConfig
    cfg = config or StaleMemoryConfig()
    evolver = StaleMemoryEvolver.__new__(StaleMemoryEvolver)
    evolver.config = cfg
    return evolver


def _make_memory(code_nodes=None, candidate_nodes=None):
    """Return candidates only for the 'episodic' type query to avoid 5x duplication.

    evolve() calls search_nodes once per non-code memory type (5 calls total) and
    extends a single list.  Returning candidate_nodes for every non-code call would
    make each item appear 5 times, causing multiple memory.update() calls and
    breaking assert_called_once_with assertions.  Pinning candidates to one type
    gives deterministic, single-occurrence results.
    """
    memory = Mock()
    memory._graph = Mock()

    def _side_effect(q):
        mtype = q.get("memory_type")
        if mtype == "code":
            return code_nodes or []
        if mtype == "episodic":
            return candidate_nodes or []
        return []

    memory._graph.search_nodes = Mock(side_effect=_side_effect)
    memory.update = Mock()
    return memory


class TestMetadata:
    def test_fields(self):
        from smartmemory.plugins.evolvers.stale_memory import StaleMemoryEvolver
        m = StaleMemoryEvolver.metadata()
        assert m.name == "stale_memory"
        assert m.plugin_type == "evolver"


class TestConfig:
    def test_defaults(self):
        from smartmemory.plugins.evolvers.stale_memory import StaleMemoryConfig
        cfg = StaleMemoryConfig()
        assert cfg.enabled is True
        assert cfg.batch_size == 100

    def test_raises_without_typed_config(self):
        from smartmemory.plugins.evolvers.stale_memory import StaleMemoryEvolver
        evolver = StaleMemoryEvolver.__new__(StaleMemoryEvolver)
        evolver.config = object()  # no 'enabled' attr
        with pytest.raises(TypeError):
            evolver.evolve(Mock())


class TestEvolve:
    def test_no_code_nodes_returns_early(self):
        evolver = _make_evolver()
        memory = _make_memory(code_nodes=[])
        evolver.evolve(memory)
        memory.update.assert_not_called()

    def test_marks_stale_on_commit_mismatch(self):
        from smartmemory.plugins.evolvers.stale_memory import StaleMemoryConfig

        code_node = Mock()
        code_node.metadata = {"repo": "my-repo", "file_path": "src/auth.py", "commit_hash": "newcommit"}

        mem_item = Mock()
        mem_item.metadata = {
            "source_code_refs": [
                {"repo": "my-repo", "file_path": "src/auth.py", "commit_hash": "oldcommit"}
            ]
        }
        mem_item.item_id = "item-1"

        evolver = _make_evolver(StaleMemoryConfig(batch_size=10))
        memory = _make_memory(code_nodes=[code_node], candidate_nodes=[mem_item])
        evolver.evolve(memory)

        assert mem_item.metadata.get("stale") is True
        memory.update.assert_called_once_with(mem_item)

    def test_no_stale_on_commit_match(self):
        code_node = Mock()
        code_node.metadata = {"repo": "my-repo", "file_path": "src/auth.py", "commit_hash": "abc123"}

        mem_item = Mock()
        mem_item.metadata = {
            "source_code_refs": [
                {"repo": "my-repo", "file_path": "src/auth.py", "commit_hash": "abc123"}
            ]
        }

        evolver = _make_evolver()
        memory = _make_memory(code_nodes=[code_node], candidate_nodes=[mem_item])
        evolver.evolve(memory)

        assert "stale" not in mem_item.metadata
        memory.update.assert_not_called()

    def test_no_stale_on_missing_source_code_refs(self):
        code_node = Mock()
        code_node.metadata = {"repo": "my-repo", "file_path": "src/auth.py", "commit_hash": "abc123"}

        mem_item = Mock()
        mem_item.metadata = {}  # no source_code_refs

        evolver = _make_evolver()
        memory = _make_memory(code_nodes=[code_node], candidate_nodes=[mem_item])
        evolver.evolve(memory)

        memory.update.assert_not_called()

    def test_disabled_config_is_noop(self):
        from smartmemory.plugins.evolvers.stale_memory import StaleMemoryConfig
        evolver = _make_evolver(StaleMemoryConfig(enabled=False))
        memory = Mock()
        evolver.evolve(memory)
        memory._graph.search_nodes.assert_not_called()

    def test_file_deleted_no_false_positive(self):
        """File deleted from repo (not in current code index) — guard returns '' for current_commit, skip."""
        code_node = Mock()
        code_node.metadata = {"repo": "my-repo", "file_path": "src/other.py", "commit_hash": "abc123"}

        mem_item = Mock()
        mem_item.metadata = {
            "source_code_refs": [
                {"repo": "my-repo", "file_path": "src/deleted.py", "commit_hash": "oldcommit"}
            ]
        }

        evolver = _make_evolver()
        memory = _make_memory(code_nodes=[code_node], candidate_nodes=[mem_item])
        evolver.evolve(memory)

        assert "stale" not in mem_item.metadata
        memory.update.assert_not_called()

    def test_per_item_exception_does_not_abort_cycle(self):
        code_node = Mock()
        code_node.metadata = {"repo": "r", "file_path": "f.py", "commit_hash": "new"}

        bad_item = Mock()
        bad_item.metadata = Mock()
        bad_item.metadata.get = Mock(side_effect=RuntimeError("broken"))
        bad_item.item_id = "bad-1"

        good_item = Mock()
        good_item.metadata = {
            "source_code_refs": [{"repo": "r", "file_path": "f.py", "commit_hash": "old"}]
        }
        good_item.item_id = "good-1"

        logger = Mock()
        evolver = _make_evolver()
        memory = _make_memory(code_nodes=[code_node], candidate_nodes=[bad_item, good_item])
        evolver.evolve(memory, logger=logger)

        assert good_item.metadata.get("stale") is True
        logger.warning.assert_called()

    def test_no_logger_does_not_raise_on_exception(self):
        code_node = Mock()
        code_node.metadata = {"repo": "r", "file_path": "f.py", "commit_hash": "new"}

        bad_item = Mock()
        bad_item.metadata = Mock()
        bad_item.metadata.get = Mock(side_effect=RuntimeError("broken"))
        bad_item.item_id = "bad-1"

        evolver = _make_evolver()
        memory = _make_memory(code_nodes=[code_node], candidate_nodes=[bad_item])
        evolver.evolve(memory, logger=None)  # must not raise
