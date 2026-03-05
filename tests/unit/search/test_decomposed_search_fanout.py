"""Fan-out verification tests for SmartMemory._decomposed_search().

Validates call counts and top_k values passed to _search.search —
no timing assertions, purely structural verification.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from smartmemory.smart_memory import SmartMemory


@dataclass
class MockResult:
    item_id: str
    content: str


def _make_mock_search(results_per_call=3):
    """Create a mock _search.search that returns canned results and tracks calls."""
    mock = MagicMock()
    call_counter = {"count": 0}

    def side_effect(query, top_k=5, memory_type=None, **kwargs):
        call_counter["count"] += 1
        return [MockResult(f"id-{call_counter['count']}-{i}", f"content-{i}") for i in range(results_per_call)]

    mock.side_effect = side_effect
    return mock, call_counter


class TestDecomposedSearchFanout:
    def _make_sm(self, mock_search):
        """Create a SmartMemory with mocked internals."""
        sm = object.__new__(SmartMemory)
        sm._search = MagicMock()
        sm._search.search = mock_search
        sm.scope_provider = None
        sm._di_context = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        return sm

    def test_single_subquery_no_fanout(self):
        mock_search, counter = _make_mock_search()
        sm = self._make_sm(mock_search)
        sm._decomposed_search("simple query", top_k=5)
        assert counter["count"] == 1

    def test_compound_query_fanout_count(self):
        mock_search, counter = _make_mock_search()
        sm = self._make_sm(mock_search)
        sm._decomposed_search("auth and caching", top_k=5)
        # "auth and caching" decomposes to: ["auth and caching", "auth", "caching"] = 3 calls
        assert counter["count"] == 3

    def test_max_fanout_bounded(self):
        mock_search, counter = _make_mock_search()
        sm = self._make_sm(mock_search)
        sm._decomposed_search("auth, caching, logging, metrics, tracing", top_k=5)
        # 5 fragments but cap is 4 sub-queries
        assert counter["count"] <= 4

    def test_no_overfetch_per_subquery(self):
        mock_search, _ = _make_mock_search()
        sm = self._make_sm(mock_search)
        sm._decomposed_search("auth and caching", top_k=5)
        # Each call should receive top_k=5, not top_k=10
        for call in mock_search.call_args_list:
            _, kwargs = call
            assert kwargs.get("top_k", call[0][1] if len(call[0]) > 1 else None) == 5


class TestWorkingMemoryDecomposeBypass:
    """TECHDEBT-SEARCH-1: working-memory decomposition bypass logging."""

    def _make_sm(self):
        sm = object.__new__(SmartMemory)
        sm._search = MagicMock()
        sm._search.search = MagicMock(return_value=[])
        sm.scope_provider = None
        sm._di_context = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        sm._working_buffer = []
        return sm

    @patch("smartmemory.utils.get_config", return_value={"persist": True})
    def test_working_memory_decompose_emits_warning(self, _cfg, caplog):
        import logging
        sm = self._make_sm()
        with caplog.at_level(logging.WARNING, logger="smartmemory.smart_memory"):
            sm.search("auth and caching", memory_type="working", decompose_query=True)
        assert any("working_memory_exempt" in r.message or getattr(r, "reason", None) == "working_memory_exempt" for r in caplog.records)

    @patch("smartmemory.utils.get_config", return_value={"persist": True})
    def test_non_working_memory_no_exemption_warning(self, _cfg, caplog):
        import logging
        sm = self._make_sm()
        with patch("smartmemory.smart_memory.trace_span", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())):
            with caplog.at_level(logging.WARNING, logger="smartmemory.smart_memory"):
                sm.search("auth and caching", memory_type="semantic", decompose_query=True)
        assert not any("working_memory_exempt" in r.message for r in caplog.records)

    @patch("smartmemory.utils.get_config", return_value={"persist": True})
    def test_working_memory_no_decompose_no_warning(self, _cfg, caplog):
        import logging
        sm = self._make_sm()
        with caplog.at_level(logging.WARNING, logger="smartmemory.smart_memory"):
            sm.search("auth", memory_type="working", decompose_query=False)
        assert not any("working_memory_exempt" in r.message for r in caplog.records)
