"""Tests for the Code Connector AST parser and indexer.

Tests parsing of Python files into code entities and relationships.
Uses a fixture file to validate extraction of classes, functions, imports,
routes, tests, and various relationship types. Also tests CodeIndexer
orchestration with a mock graph.
"""

import os
from unittest.mock import MagicMock

import pytest

from smartmemory.code.models import CodeEntity, ImportSymbol, IndexResult
from smartmemory.code.indexer import CodeIndexer, SymbolTable
from smartmemory.code.parser import CodeParser, collect_python_files


# -- Fixtures --

SAMPLE_PYTHON_CODE = '''
"""A sample module for testing the code parser."""

import os
from typing import Optional, List

from fastapi import APIRouter
from some_module import helper_function


router = APIRouter()


class BaseModel:
    """A base model class."""
    pass


class UserService(BaseModel):
    """Service for managing users."""

    def create_user(self, name: str) -> dict:
        """Create a new user."""
        result = helper_function(name)
        return {"name": name, "result": result}

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID."""
        return True


@router.get("/users")
def list_users(limit: int = 10) -> list:
    """List all users."""
    service = UserService()
    return service.create_user("test")


@router.post("/users")
async def create_user_endpoint(name: str) -> dict:
    """Create a user via API."""
    service = UserService()
    return service.create_user(name)


def helper_function(x):
    """A helper that does something."""
    return x


class TestUserService:
    """Tests for UserService."""

    def test_create_user(self):
        """Test user creation."""
        svc = UserService()
        result = svc.create_user("alice")
        assert result["name"] == "alice"

    def test_delete_user(self):
        """Test user deletion."""
        svc = UserService()
        assert svc.delete_user("123") is True
'''


@pytest.fixture
def sample_file(tmp_path):
    """Create a temporary Python file with sample code."""
    file_path = tmp_path / "sample_service.py"
    file_path.write_text(SAMPLE_PYTHON_CODE)
    return str(file_path), str(tmp_path)


@pytest.fixture
def parser(sample_file):
    """Create a parser for the sample file."""
    _, repo_root = sample_file
    return CodeParser(repo="test-repo", repo_root=repo_root)


# -- Model Tests --


class TestCodeEntity:
    def test_item_id_deterministic(self):
        entity = CodeEntity(
            name="MyClass",
            entity_type="class",
            file_path="src/models.py",
            line_number=10,
            repo="my-repo",
        )
        assert entity.item_id == "code::my-repo::src/models.py::MyClass"

    def test_to_properties(self):
        entity = CodeEntity(
            name="get_users",
            entity_type="route",
            file_path="routes/users.py",
            line_number=42,
            repo="api",
            docstring="Get all users",
            http_method="GET",
            http_path="/users",
            decorators=["router.get"],
            bases=[],
        )
        props = entity.to_properties()
        assert props["item_id"] == "code::api::routes/users.py::get_users"
        assert props["name"] == "get_users"
        assert props["entity_type"] == "route"
        assert props["memory_type"] == "code"
        assert props["http_method"] == "GET"
        assert props["http_path"] == "/users"
        assert props["decorators"] == "router.get"

    def test_to_properties_omits_empty_optional_fields(self):
        entity = CodeEntity(
            name="foo",
            entity_type="function",
            file_path="a.py",
            line_number=1,
            repo="r",
        )
        props = entity.to_properties()
        assert "docstring" not in props
        assert "decorators" not in props
        assert "bases" not in props
        assert "http_method" not in props


class TestIndexResult:
    def test_defaults(self):
        result = IndexResult(repo="my-repo")
        assert result.files_parsed == 0
        assert result.entities_created == 0
        assert result.edges_created == 0
        assert result.errors == []


# -- Parser Tests --


class TestCodeParser:
    def test_parse_module_entity(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        modules = [e for e in result.entities if e.entity_type == "module"]
        assert len(modules) == 1
        assert modules[0].name == "sample_service"
        assert modules[0].line_number == 1

    def test_parse_classes(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        classes = [e for e in result.entities if e.entity_type == "class"]
        class_names = {c.name for c in classes}
        assert "BaseModel" in class_names
        assert "UserService" in class_names

    def test_parse_functions(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        functions = [e for e in result.entities if e.entity_type == "function"]
        func_names = {f.name for f in functions}
        assert "helper_function" in func_names
        assert "UserService.create_user" in func_names
        assert "UserService.delete_user" in func_names

    def test_parse_routes(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        routes = [e for e in result.entities if e.entity_type == "route"]
        assert len(routes) == 2

        get_route = next(r for r in routes if r.http_method == "GET")
        assert get_route.name == "list_users"
        assert get_route.http_path == "/users"

        post_route = next(r for r in routes if r.http_method == "POST")
        assert post_route.name == "create_user_endpoint"
        assert post_route.http_path == "/users"

    def test_parse_tests(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        tests = [e for e in result.entities if e.entity_type == "test"]
        test_names = {t.name for t in tests}
        assert "TestUserService" in test_names
        assert "TestUserService.test_create_user" in test_names
        assert "TestUserService.test_delete_user" in test_names

    def test_parse_docstrings(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        user_service = next(e for e in result.entities if e.name == "UserService")
        assert user_service.docstring == "Service for managing users."

    def test_parse_inheritance(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        inherits = [r for r in result.relations if r.relation_type == "INHERITS"]
        assert len(inherits) >= 1
        # UserService inherits BaseModel
        us_inherits = [r for r in inherits if "UserService" in r.source_id and "BaseModel" in r.target_id]
        assert len(us_inherits) == 1

    def test_parse_defines_edges(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        defines = [r for r in result.relations if r.relation_type == "DEFINES"]
        # Module should define classes + top-level functions
        assert (
            len(defines) >= 5
        )  # BaseModel, UserService, list_users, create_user_endpoint, helper_function, TestUserService + methods

    def test_parse_imports(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        imports = [r for r in result.relations if r.relation_type == "IMPORTS"]
        assert len(imports) >= 3  # os, typing, fastapi, some_module

    def test_parse_calls(self, parser, sample_file):
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        calls = [r for r in result.relations if r.relation_type == "CALLS"]
        # Functions call helper_function, UserService, create_user, etc.
        assert len(calls) >= 1

    def test_parse_tests_edges(self, parser, sample_file):
        """Test that TESTS edges link test methods to tested functions."""
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        tests_edges = [r for r in result.relations if r.relation_type == "TESTS"]
        assert len(tests_edges) >= 1
        # test_create_user should link to UserService.create_user
        test_targets = {r.target_id for r in tests_edges}
        assert any("create_user" in t for t in test_targets)

    def test_parse_tests_edges_class_qualified(self, tmp_path):
        """Test that class-qualified test methods (TestFoo.test_bar) create TESTS edges."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(1, 2) == 3
"""
        file_path = tmp_path / "test_calc.py"
        file_path.write_text(code)
        parser = CodeParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(file_path))

        tests_edges = [r for r in result.relations if r.relation_type == "TESTS"]
        assert len(tests_edges) == 1
        # TestCalculator.test_add should link to Calculator.add
        assert "Calculator.add" in tests_edges[0].target_id

    def test_handles_syntax_error(self, parser, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:\n    pass")
        result = parser.parse_file(str(bad_file))
        assert len(result.errors) == 1
        assert "SyntaxError" in result.errors[0]

    def test_handles_empty_file(self, parser, tmp_path):
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        result = parser.parse_file(str(empty_file))
        # Should still create a module entity
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "module"

    def test_init_py_module_name(self, tmp_path):
        """__init__.py files should use package name, not include .__init__ suffix."""
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        init_file = pkg / "__init__.py"
        init_file.write_text('"""Package docstring."""\n')
        parser = CodeParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(init_file))

        modules = [e for e in result.entities if e.entity_type == "module"]
        assert len(modules) == 1
        assert modules[0].name == "mypkg"  # NOT mypkg.__init__

    def test_defines_edges_class_owns_methods(self, parser, sample_file):
        """DEFINES edges for methods should source from the class, not the module."""
        file_path, _ = sample_file
        result = parser.parse_file(file_path)

        # Find the DEFINES edge for UserService.create_user specifically
        create_user_defines = [
            r
            for r in result.relations
            if r.relation_type == "DEFINES" and r.target_id.endswith("::UserService.create_user")
        ]
        assert len(create_user_defines) == 1
        # Source should be UserService (class), not the module
        assert create_user_defines[0].source_id.endswith("::UserService")


# -- File Collection Tests --


class TestCollectPythonFiles:
    def test_collects_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.txt").write_text("not python")

        files = collect_python_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_excludes_directories(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("y = 2")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "hooks.py").write_text("z = 3")

        files = collect_python_files(str(tmp_path))
        file_names = [os.path.basename(f) for f in files]
        assert "app.py" in file_names
        assert "cached.py" not in file_names
        assert "hooks.py" not in file_names

    def test_recurses_into_subdirectories(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "deep.py").write_text("x = 1")

        files = collect_python_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("deep.py")


# -- Indexer Tests --


class TestCodeIndexer:
    """Tests for CodeIndexer orchestration with a mock graph."""

    def test_golden_flow(self, tmp_path):
        """Index a small directory and verify bulk graph calls."""
        (tmp_path / "app.py").write_text('class App:\n    """Main app."""\n    pass\n')
        (tmp_path / "utils.py").write_text("def helper():\n    return 1\n")

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 4  # module + class + module + function
        graph.add_edges_bulk.return_value = 2  # DEFINES edges

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.files_parsed == 2
        assert result.entities_created > 0
        assert result.edges_created >= 0
        assert result.errors == []
        # Verify delete was called
        graph.execute_query.assert_called_once()
        assert "DETACH DELETE" in graph.execute_query.call_args[0][0]
        # Verify bulk methods were called once each
        graph.add_nodes_bulk.assert_called_once()
        graph.add_edges_bulk.assert_called_once()
        # Verify the bulk node list contains properties dicts
        bulk_nodes = graph.add_nodes_bulk.call_args[0][0]
        assert all(isinstance(n, dict) and "item_id" in n for n in bulk_nodes)

    def test_skips_dangling_edges(self, tmp_path):
        """Edges referencing non-existent entities are filtered out."""
        code = "from external_lib import something\ndef foo():\n    something()\n"
        (tmp_path / "caller.py").write_text(code)

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 2
        graph.add_edges_bulk.return_value = 1

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        indexer.index()

        # The bulk edge list should only contain edges where both endpoints exist
        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        entity_ids = {n["item_id"] for n in graph.add_nodes_bulk.call_args[0][0]}
        for source_id, target_id, _edge_type, _ in bulk_edges:
            assert source_id in entity_ids and target_id in entity_ids

    def test_handles_graph_write_errors(self, tmp_path):
        """Graph write failures are captured as errors, not raised."""
        (tmp_path / "simple.py").write_text("x = 1\n")

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.side_effect = RuntimeError("Graph unavailable")
        graph.add_edges_bulk.return_value = 0

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.entities_created == 0
        assert len(result.errors) > 0
        assert "Graph unavailable" in result.errors[0]

    def test_delete_includes_workspace_scope(self, tmp_path):
        """When scope_provider has workspace_id, delete query should include it."""
        (tmp_path / "app.py").write_text("x = 1\n")

        # Mock graph with scope_provider returning workspace_id
        scope_provider = MagicMock()
        scope_provider.get_isolation_filters.return_value = {"workspace_id": "ws-123"}
        backend = MagicMock()
        backend.scope_provider = scope_provider

        graph = MagicMock(spec=["execute_query", "add_nodes_bulk", "add_edges_bulk", "nodes", "get_scope_filters"])
        graph.nodes.backend = backend
        graph.get_scope_filters.return_value = {"workspace_id": "ws-123"}
        graph.execute_query.return_value = []

        indexer = CodeIndexer(graph=graph, repo="test-repo", repo_root=str(tmp_path))
        indexer.index()

        # The delete query should contain workspace_id filter
        delete_call = graph.execute_query.call_args
        query_str = delete_call[0][0]
        params = delete_call[0][1]
        assert "workspace_id" in query_str
        assert params["workspace_id"] == "ws-123"
        assert "DETACH DELETE" in query_str


# -- Symbol Table Tests (CODE-DEV-5) --


class TestSymbolTable:
    """Unit tests for the SymbolTable used in cross-file call resolution."""

    def _make_entity(self, name: str, entity_type: str, file_path: str, repo: str = "repo") -> CodeEntity:
        return CodeEntity(name=name, entity_type=entity_type, file_path=file_path, line_number=1, repo=repo)

    def test_symbol_table_built_from_imports(self):
        """Symbol table maps imported name to target entity's item_id.

        Scenario: file_b.py defines CodeParser; file_a.py imports it with
        ``from file_b import CodeParser``. After building the table, resolving
        ("file_a.py", "CodeParser") should return CodeParser's item_id.
        """
        repo = "my-repo"

        entity_b = self._make_entity("CodeParser", "class", "file_b.py", repo=repo)

        sym = ImportSymbol(
            importing_file="file_a.py",
            local_name="CodeParser",
            module_path="file_b",
            original_name="CodeParser",
            is_module_import=False,
        )

        table = SymbolTable()
        table.register_entity(entity_b)
        table.register_import(sym)

        resolved = table.resolve("file_a.py", "CodeParser")
        assert resolved == entity_b.item_id, f"Expected {entity_b.item_id!r}, got {resolved!r}"

    def test_symbol_table_alias_import(self):
        """Aliased imports (``from foo import Bar as B``) resolve via alias."""
        repo = "r"
        entity = self._make_entity("Bar", "class", "foo.py", repo=repo)

        sym = ImportSymbol(
            importing_file="caller.py",
            local_name="B",
            module_path="foo",
            original_name="Bar",
            is_module_import=False,
        )

        table = SymbolTable()
        table.register_entity(entity)
        table.register_import(sym)

        assert table.resolve("caller.py", "B") == entity.item_id
        # The original name is NOT a key for this file — the alias is
        assert table.resolve("caller.py", "Bar") is None

    def test_symbol_table_module_import(self):
        """``import foo as f`` followed by ``f.Bar()`` resolves Bar's item_id."""
        repo = "r"
        module_entity = self._make_entity("foo", "module", "foo.py", repo=repo)
        bar_entity = self._make_entity("Bar", "class", "foo.py", repo=repo)

        sym = ImportSymbol(
            importing_file="caller.py",
            local_name="f",
            module_path="foo",
            original_name="foo",
            is_module_import=True,
        )

        table = SymbolTable()
        table.register_entity(module_entity)
        table.register_entity(bar_entity)
        table.register_import(sym)

        # Attribute call "f.Bar" should resolve to Bar's item_id
        assert table.resolve("caller.py", "f.Bar") == bar_entity.item_id

    def test_symbol_table_unknown_name_returns_none(self):
        """Resolving a name that was never imported returns None."""
        table = SymbolTable()
        assert table.resolve("any_file.py", "UnknownFunction") is None

    def test_symbol_table_scoped_to_importing_file(self):
        """Import in file_a does not leak into file_b's scope."""
        repo = "r"
        entity = self._make_entity("Foo", "class", "lib.py", repo=repo)

        sym = ImportSymbol(
            importing_file="file_a.py",
            local_name="Foo",
            module_path="lib",
            original_name="Foo",
            is_module_import=False,
        )

        table = SymbolTable()
        table.register_entity(entity)
        table.register_import(sym)

        assert table.resolve("file_a.py", "Foo") == entity.item_id
        # file_b.py never imported Foo — must not resolve
        assert table.resolve("file_b.py", "Foo") is None


class TestCrossFileCallResolution:
    """Integration-level tests for cross-file CALLS edge resolution via CodeIndexer."""

    def _make_graph(self) -> MagicMock:
        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 0
        graph.add_edges_bulk.return_value = 0
        graph.get_scope_filters.return_value = {}
        return graph

    def test_cross_file_calls_resolved(self, tmp_path):
        """CALLS edge to an imported function resolves to the correct item_id.

        File layout:
          utils.py   — defines ``process()``
          caller.py  — does ``from utils import process`` then calls ``process()``

        After indexing, the CALLS edge from caller.process_wrapper → process
        should target ``code::repo::utils.py::process``, not the dangling
        same-file guess ``code::repo::caller.py::process``.
        """
        (tmp_path / "utils.py").write_text(
            "def process(data):\n"
            "    '''Process the data.'''\n"
            "    return data\n"
        )
        (tmp_path / "caller.py").write_text(
            "from utils import process\n"
            "\n"
            "def process_wrapper(x):\n"
            "    return process(x)\n"
        )

        graph = self._make_graph()
        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        indexer.index()

        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        calls_edges = [(src, tgt) for src, tgt, etype, _ in bulk_edges if etype == "CALLS"]

        # The resolved CALLS edge target must point to utils.py::process
        resolved_targets = [tgt for _, tgt in calls_edges]
        assert any("utils.py::process" in t for t in resolved_targets), (
            f"Expected a CALLS edge targeting utils.py::process. "
            f"Got CALLS targets: {resolved_targets}"
        )

        # There must be no dangling caller.py::process target (wrong same-file guess)
        assert not any("caller.py::process" in t for t in resolved_targets), (
            f"Same-file dangling target caller.py::process should have been resolved away. "
            f"Got: {resolved_targets}"
        )

    def test_unresolvable_calls_remain_dropped(self, tmp_path):
        """Calls to names that are not imported from any indexed file are still filtered out.

        ``external_lib`` is not present in the repo, so ``something()`` should
        remain a dangling CALLS edge and must NOT appear in the bulk_edges list.
        """
        (tmp_path / "caller.py").write_text(
            "from external_lib import something\n"
            "\n"
            "def run():\n"
            "    something()\n"
        )

        graph = self._make_graph()
        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        indexer.index()

        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        entity_ids = {n["item_id"] for n in graph.add_nodes_bulk.call_args[0][0]}

        for source_id, target_id, _etype, _ in bulk_edges:
            assert source_id in entity_ids, f"Dangling source: {source_id}"
            assert target_id in entity_ids, f"Dangling target: {target_id}"

    def test_import_symbols_populated_by_parser(self, tmp_path):
        """ParseResult.import_symbols is populated for both import styles."""
        (tmp_path / "a.py").write_text(
            "from mylib.utils import helper, Processor\n"
            "import os\n"
            "import mylib.models as models\n"
        )
        parser = CodeParser(repo="repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(tmp_path / "a.py"))

        syms = {s.local_name: s for s in result.import_symbols}

        # from-import: bare names recorded
        assert "helper" in syms
        assert syms["helper"].module_path == "mylib.utils"
        assert syms["helper"].original_name == "helper"
        assert syms["helper"].is_module_import is False

        assert "Processor" in syms
        assert syms["Processor"].original_name == "Processor"

        # plain import: recorded as module import
        assert "os" in syms
        assert syms["os"].is_module_import is True

        # aliased module import
        assert "models" in syms
        assert syms["models"].module_path == "mylib.models"
        assert syms["models"].is_module_import is True

    def test_module_import_alias_cross_file_calls_resolved(self, tmp_path):
        """Chained alias calls (import lib as f; f.baz()) resolve cross-file.

        With the fixed _get_call_name, the callee stored in the CALLS edge
        properties is 'f.baz' (full dotted name), which matches the symbol
        table key pre-registered by register_import for module imports.
        """
        (tmp_path / "lib.py").write_text("def baz():\n    return 1\n")
        (tmp_path / "caller.py").write_text(
            "import lib as f\n"
            "\n"
            "def run():\n"
            "    result = f.baz()\n"
        )

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 0
        graph.add_edges_bulk.return_value = 0
        graph.get_scope_filters.return_value = {}

        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        indexer.index()

        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        calls_edges = [(src, tgt) for src, tgt, etype, _ in bulk_edges if etype == "CALLS"]
        resolved_targets = [tgt for _, tgt in calls_edges]

        assert any("lib.py::baz" in t for t in resolved_targets), (
            f"Expected CALLS edge targeting lib.py::baz. Got: {resolved_targets}"
        )

    def test_wildcard_import_is_gracefully_ignored(self, tmp_path):
        """``from foo import *`` does not crash and does not pollute the symbol table.

        The wildcard name '*' cannot be resolved to any specific entity, so
        the ImportSymbol with local_name='*' should be silently dropped during
        register_import without raising an exception or adding a spurious entry.
        """
        (tmp_path / "user.py").write_text(
            "from external_lib import *\n"
            "\n"
            "def use_something():\n"
            "    result = some_func()\n"
        )

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 0
        graph.add_edges_bulk.return_value = 0
        graph.get_scope_filters.return_value = {}

        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.errors == []
        assert result.files_parsed == 1

        # No dangling edges should sneak past the entity_ids filter
        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        entity_ids = {n["item_id"] for n in graph.add_nodes_bulk.call_args[0][0]}
        for source_id, target_id, _etype, _ in bulk_edges:
            assert source_id in entity_ids, f"Dangling source: {source_id}"
            assert target_id in entity_ids, f"Dangling target: {target_id}"

    def test_circular_imports_do_not_infinite_loop(self, tmp_path):
        """Circular imports (a imports b, b imports a) complete without error.

        The indexer processes each file independently in a single parse pass
        (AST only, no execution), so circular import chains cannot cause
        recursion or blocking.
        """
        (tmp_path / "a.py").write_text("from b import something\n\ndef foo():\n    pass\n")
        (tmp_path / "b.py").write_text("from a import foo\n\ndef something():\n    pass\n")

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 0
        graph.add_edges_bulk.return_value = 0
        graph.get_scope_filters.return_value = {}

        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.files_parsed == 2
        assert result.errors == []

    def test_empty_repo_does_not_crash(self, tmp_path):
        """Indexing an empty directory produces an empty IndexResult without errors."""
        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 0
        graph.add_edges_bulk.return_value = 0
        graph.get_scope_filters.return_value = {}

        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.files_parsed == 0
        assert result.entities_created == 0
        assert result.errors == []

    def test_module_with_only_module_entity(self, tmp_path):
        """A file containing only constants (no functions or classes) produces
        a single module entity and no CALLS or DEFINES edges, without crashing.
        """
        (tmp_path / "constants.py").write_text(
            '"""Project constants."""\n'
            "VERSION = 1\n"
            "MAX_RETRIES = 3\n"
        )

        graph = MagicMock()
        graph.execute_query.return_value = []
        graph.add_nodes_bulk.return_value = 0
        graph.add_edges_bulk.return_value = 0
        graph.get_scope_filters.return_value = {}

        indexer = CodeIndexer(graph=graph, repo="repo", repo_root=str(tmp_path))
        result = indexer.index()

        assert result.files_parsed == 1
        assert result.errors == []
        # Only the module entity is written; no functions/classes to create edges for
        bulk_nodes = graph.add_nodes_bulk.call_args[0][0]
        assert len(bulk_nodes) == 1
        assert bulk_nodes[0]["entity_type"] == "module"
        bulk_edges = graph.add_edges_bulk.call_args[0][0]
        assert bulk_edges == []
