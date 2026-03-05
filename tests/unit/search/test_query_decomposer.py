"""Unit tests for smartmemory.search.query_decomposer.decompose()."""

from smartmemory.search.query_decomposer import decompose


class TestDecompose:
    def test_basic_and_decomposition(self):
        result = decompose("auth flow and caching strategy")
        assert result == ["auth flow and caching strategy", "auth flow", "caching strategy"]

    def test_or_decomposition(self):
        result = decompose("auth or caching")
        assert len(result) == 3
        assert result[0] == "auth or caching"
        assert "auth" in result
        assert "caching" in result

    def test_comma_decomposition(self):
        result = decompose("auth, caching, logging")
        assert len(result) == 4
        assert result[0] == "auth, caching, logging"
        assert "auth" in result
        assert "caching" in result
        assert "logging" in result

    def test_semicolon_decomposition(self):
        result = decompose("auth; caching")
        assert len(result) == 3
        assert result[0] == "auth; caching"
        assert "auth" in result
        assert "caching" in result

    def test_no_decomposition(self):
        assert decompose("simple query") == ["simple query"]

    def test_single_word(self):
        assert decompose("auth") == ["auth"]

    def test_empty_string(self):
        assert decompose("") == []

    def test_none_input(self):
        assert decompose(None) == []

    def test_short_query(self):
        assert decompose("ab") == ["ab"]

    def test_max_sub_queries_cap(self):
        result = decompose("auth, caching, logging, metrics, tracing")
        assert len(result) <= 4
        assert result[0] == "auth, caching, logging, metrics, tracing"

    def test_short_fragments_filtered(self):
        result = decompose("auth and ab")
        assert result == ["auth and ab", "auth"]

    def test_original_always_first(self):
        result = decompose("auth and caching")
        assert result[0] == "auth and caching"

    def test_mixed_conjunctions(self):
        result = decompose("auth and caching, logging")
        assert "auth" in result
        assert "caching" in result
        assert "logging" in result

    def test_case_insensitive_and(self):
        result = decompose("auth AND caching")
        assert len(result) == 3
        assert result[0] == "auth AND caching"

    def test_whitespace_stripped(self):
        result = decompose("  auth  and  caching  ")
        for fragment in result[1:]:
            assert fragment == fragment.strip()
