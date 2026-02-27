"""Unit tests for the TypeScript/JavaScript AST parser (CODE-DEV-7).

Tests parsing of TSX/TS/JS files into code entities and relationships
using tree-sitter. Covers entity type detection (function, hook, component,
class, method), relation extraction (DEFINES, IMPORTS, CALLS), item_id
format, and language dispatch.
"""

import os
from unittest.mock import patch

import pytest

from smartmemory.code.models import CodeEntity, ParseResult
from smartmemory.code.ts_parser import TSParser, collect_ts_files
from smartmemory.code.parser import parse_file, CodeParser


# ── Sample source fixtures ──────────────────────────────────────────────────

SAMPLE_TSX = b"""
import React from 'react';
import { useState } from 'react';
export function UserProfile({ userId }) { return <div>{userId}</div>; }
export const useAuth = () => ({ isLoggedIn: true });
export const Button = ({ label }) => <button>{label}</button>;
class AuthService { validate(token) { return token.length > 0; } }
"""

SAMPLE_TS = b"""
export function calculateTotal(items: number[]): number {
  return items.reduce((a, b) => a + b, 0);
}

export class PaymentProcessor {
  process(amount: number): boolean {
    return amount > 0;
  }
}
"""

SAMPLE_JS = b"""
const greet = (name) => `Hello, ${name}`;
function farewell(name) { return `Goodbye, ${name}`; }
"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_tsx(source: bytes, tmp_path, filename: str = "sample.tsx") -> ParseResult:
    """Write *source* to a temp file and parse it as TSX."""
    f = tmp_path / filename
    f.write_bytes(source)
    parser = TSParser(repo="test-repo", repo_root=str(tmp_path))
    return parser.parse_file(str(f))


def _entity_by_name(result: ParseResult, name: str) -> CodeEntity:
    """Return the first entity with *name*, or raise AssertionError."""
    matches = [e for e in result.entities if e.name == name]
    assert matches, f"No entity named '{name}' in {[e.name for e in result.entities]}"
    return matches[0]


# ── Entity type extraction ───────────────────────────────────────────────────


class TestTSParserEntityTypes:
    def test_parse_tsx_extracts_function(self, tmp_path):
        """Named function returning JSX → component; named function not returning JSX → function."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        entity = _entity_by_name(result, "UserProfile")
        # UserProfile returns <div> — it's a component
        assert entity.entity_type == "component"

    def test_parse_tsx_extracts_hook(self, tmp_path):
        """Arrow const starting with 'use' + uppercase → hook."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        entity = _entity_by_name(result, "useAuth")
        assert entity.entity_type == "hook"

    def test_parse_tsx_extracts_component(self, tmp_path):
        """Arrow const returning JSX (no 'use' prefix) → component."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        entity = _entity_by_name(result, "Button")
        assert entity.entity_type == "component"

    def test_parse_tsx_extracts_class(self, tmp_path):
        """Class declaration → class."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        entity = _entity_by_name(result, "AuthService")
        assert entity.entity_type == "class"

    def test_parse_tsx_extracts_method(self, tmp_path):
        """Method definition inside class → function."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        entity = _entity_by_name(result, "validate")
        assert entity.entity_type == "function"

    def test_parse_tsx_creates_module_entity(self, tmp_path):
        """Every parsed file produces exactly one module entity."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        modules = [e for e in result.entities if e.entity_type == "module"]
        assert len(modules) == 1

    def test_parse_ts_extracts_function(self, tmp_path):
        """Plain .ts function declaration (no JSX) → function."""
        f = tmp_path / "util.ts"
        f.write_bytes(SAMPLE_TS)
        parser = TSParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))

        entity = _entity_by_name(result, "calculateTotal")
        assert entity.entity_type == "function"

    def test_parse_ts_extracts_class_and_method(self, tmp_path):
        """TS class and its method are extracted."""
        f = tmp_path / "payment.ts"
        f.write_bytes(SAMPLE_TS)
        parser = TSParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))

        cls = _entity_by_name(result, "PaymentProcessor")
        assert cls.entity_type == "class"

        method = _entity_by_name(result, "process")
        assert method.entity_type == "function"

    def test_parse_js_extracts_arrow_function(self, tmp_path):
        """Arrow const in plain .js → function (no JSX)."""
        f = tmp_path / "helpers.js"
        f.write_bytes(SAMPLE_JS)
        parser = TSParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))

        entity = _entity_by_name(result, "greet")
        assert entity.entity_type == "function"

    def test_parse_js_extracts_function_declaration(self, tmp_path):
        """Named function declaration in .js → function."""
        f = tmp_path / "helpers.js"
        f.write_bytes(SAMPLE_JS)
        parser = TSParser(repo="test-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))

        entity = _entity_by_name(result, "farewell")
        assert entity.entity_type == "function"


# ── Hook detection edge cases ─────────────────────────────────────────────────


class TestHookDetection:
    def test_use_lowercase_only_not_a_hook(self, tmp_path):
        """'use' without a following uppercase is NOT a hook."""
        src = b"const useless = () => 42;\n"
        result = _parse_tsx(src, tmp_path)
        entity = _entity_by_name(result, "useless")
        assert entity.entity_type == "function"

    def test_use_prefix_with_uppercase_is_hook(self, tmp_path):
        """'useAuth', 'useMemory', etc. are hooks."""
        src = b"const useMemory = () => ({ memories: [] });\n"
        result = _parse_tsx(src, tmp_path)
        entity = _entity_by_name(result, "useMemory")
        assert entity.entity_type == "hook"

    def test_hook_as_function_declaration(self, tmp_path):
        """Function declaration named 'useFoo' without JSX → hook."""
        src = b"function useFoo() { return null; }\n"
        result = _parse_tsx(src, tmp_path)
        entity = _entity_by_name(result, "useFoo")
        assert entity.entity_type == "hook"


# ── item_id format ───────────────────────────────────────────────────────────


class TestItemIdFormat:
    def test_ts_item_id_format(self, tmp_path):
        """item_id must be code::{repo}::{file_path}::{name}."""
        f = tmp_path / "services" / "auth.ts"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"export function login() {}\n")

        parser = TSParser(repo="my-repo", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))

        fn = _entity_by_name(result, "login")
        rel_path = os.path.relpath(str(f), str(tmp_path))
        expected = f"code::my-repo::{rel_path}::login"
        assert fn.item_id == expected

    def test_item_id_is_deterministic(self, tmp_path):
        """Parsing the same file twice gives identical item_ids."""
        f = tmp_path / "comp.tsx"
        f.write_bytes(b"const Foo = () => <div />;\n")

        parser = TSParser(repo="r", repo_root=str(tmp_path))
        r1 = parser.parse_file(str(f))
        r2 = parser.parse_file(str(f))

        ids1 = {e.item_id for e in r1.entities}
        ids2 = {e.item_id for e in r2.entities}
        assert ids1 == ids2


# ── Language dispatch ────────────────────────────────────────────────────────


class TestLanguageDispatch:
    def test_dispatch_routes_py_to_python_parser(self, tmp_path):
        """parse_file() routes .py to CodeParser, not TSParser."""
        f = tmp_path / "app.py"
        f.write_text("def hello(): pass\n")

        result = parse_file(str(f), repo="r", repo_root=str(tmp_path))
        # Python parser uses dotted module name; TS parser uses file path
        modules = [e for e in result.entities if e.entity_type == "module"]
        assert modules
        # Python parser module name is dotted path: "app", not "app.py"
        assert modules[0].name == "app"

    def test_dispatch_routes_tsx_to_ts_parser(self, tmp_path):
        """parse_file() routes .tsx to TSParser."""
        f = tmp_path / "App.tsx"
        f.write_bytes(b"const App = () => <div />;\n")

        result = parse_file(str(f), repo="r", repo_root=str(tmp_path))
        modules = [e for e in result.entities if e.entity_type == "module"]
        assert modules
        # TS parser module name is the relative file path (includes extension)
        assert modules[0].name.endswith(".tsx")

    def test_dispatch_routes_ts_to_ts_parser(self, tmp_path):
        """parse_file() routes .ts to TSParser."""
        f = tmp_path / "utils.ts"
        f.write_bytes(b"export function add(a: number, b: number) { return a + b; }\n")
        result = parse_file(str(f), repo="r", repo_root=str(tmp_path))
        fns = [e for e in result.entities if e.entity_type == "function"]
        assert any(e.name == "add" for e in fns)

    def test_dispatch_routes_js_to_ts_parser(self, tmp_path):
        """parse_file() routes .js to TSParser (JS grammar)."""
        f = tmp_path / "helpers.js"
        f.write_bytes(b"function greet(name) { return name; }\n")
        result = parse_file(str(f), repo="r", repo_root=str(tmp_path))
        fns = [e for e in result.entities if e.entity_type == "function"]
        assert any(e.name == "greet" for e in fns)

    def test_dispatch_unsupported_extension(self, tmp_path):
        """parse_file() returns an error ParseResult for unsupported extensions."""
        f = tmp_path / "data.json"
        f.write_text("{}")
        result = parse_file(str(f), repo="r", repo_root=str(tmp_path))
        assert result.errors
        assert "Unsupported" in result.errors[0]


# ── Relations ────────────────────────────────────────────────────────────────


class TestRelations:
    def test_defines_edges_created(self, tmp_path):
        """DEFINES edges link module to top-level entities."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        defines = [r for r in result.relations if r.relation_type == "DEFINES"]
        assert len(defines) >= 3  # module → UserProfile, useAuth, Button, AuthService

    def test_defines_edge_class_to_method(self, tmp_path):
        """DEFINES edge from class to its method, not from module."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        auth_service = _entity_by_name(result, "AuthService")
        validate = _entity_by_name(result, "validate")

        class_defines = [
            r
            for r in result.relations
            if r.relation_type == "DEFINES" and r.source_id == auth_service.item_id and r.target_id == validate.item_id
        ]
        assert len(class_defines) == 1

    def test_imports_edge_created_for_local_module(self, tmp_path):
        """IMPORTS edges are created for local relative imports (not bare package imports)."""
        # Create a target module file so the specifier resolves
        (tmp_path / "utils.ts").write_bytes(b"export const x = 1;\n")
        src = b"import { x } from './utils';\nexport const y = x;\n"
        result = _parse_tsx(src, tmp_path)
        imports = [r for r in result.relations if r.relation_type == "IMPORTS"]
        assert len(imports) >= 1

    def test_bare_package_imports_are_skipped(self, tmp_path):
        """Bare package specifiers (e.g. 'react') produce no IMPORTS edges."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        imports = [r for r in result.relations if r.relation_type == "IMPORTS"]
        # SAMPLE_TSX only imports from 'react' — bare package, no local file exists
        assert len(imports) == 0

    def test_no_errors_on_valid_tsx(self, tmp_path):
        """Valid TSX should produce no errors (or only partial-parse warnings)."""
        result = _parse_tsx(SAMPLE_TSX, tmp_path)
        # No hard errors — partial-parse warnings are acceptable
        hard_errors = [e for e in result.errors if "Error reading" in e]
        assert not hard_errors

    def test_calls_edge_has_callee_property(self, tmp_path):
        """CALLS edges must carry a 'callee' property for cross-file resolution."""
        src = b"function foo() { bar(); }\n"
        result = _parse_tsx(src, tmp_path)
        calls = [r for r in result.relations if r.relation_type == "CALLS"]
        assert calls, "Expected at least one CALLS edge"
        for edge in calls:
            assert "callee" in (edge.properties or {}), (
                f"CALLS edge missing 'callee' property: {edge}"
            )
        # The callee value must match the called function name
        callee_names = {r.properties["callee"] for r in calls}
        assert "bar" in callee_names

    def test_imports_edge_target_id_uses_resolved_path(self, tmp_path):
        """IMPORTS edge target_id is keyed by the resolved file path, not the raw specifier."""
        # Create an importing file and the target file
        (tmp_path / "lib.ts").write_bytes(b"export const x = 1;\n")
        src = b"import { x } from './lib';\nexport const y = x;\n"
        result = _parse_tsx(src, tmp_path)
        imports = [r for r in result.relations if r.relation_type == "IMPORTS"]
        assert imports, "Expected an IMPORTS edge for local relative import"
        # Target ID must use the resolved file path (lib.ts), not the raw specifier (./lib)
        target_id = imports[0].target_id
        assert "lib.ts" in target_id, f"Expected resolved path in target_id, got: {target_id}"
        assert "./lib" not in target_id, f"Raw specifier must not appear in target_id: {target_id}"

    def test_resolve_ts_specifier_returns_none_for_external(self, tmp_path):
        """_resolve_ts_specifier returns None for bare package names (no ./ prefix)."""
        parser = TSParser(repo="r", repo_root=str(tmp_path))
        assert parser._resolve_ts_specifier("react", "src/app.ts") is None
        assert parser._resolve_ts_specifier("lodash/merge", "src/app.ts") is None

    def test_resolve_ts_specifier_finds_ts_file(self, tmp_path):
        """_resolve_ts_specifier resolves ./utils to utils.ts when it exists."""
        (tmp_path / "utils.ts").write_bytes(b"export const x = 1;\n")
        parser = TSParser(repo="r", repo_root=str(tmp_path))
        resolved = parser._resolve_ts_specifier("./utils", "app.ts")
        assert resolved == "utils.ts"

    def test_resolve_ts_specifier_finds_index_file(self, tmp_path):
        """_resolve_ts_specifier resolves ./components to components/index.ts."""
        (tmp_path / "components").mkdir()
        (tmp_path / "components" / "index.ts").write_bytes(b"export const x = 1;\n")
        parser = TSParser(repo="r", repo_root=str(tmp_path))
        resolved = parser._resolve_ts_specifier("./components", "app.ts")
        assert resolved == os.path.join("components", "index.ts")

    def test_resolve_ts_specifier_handles_explicit_extension(self, tmp_path):
        """_resolve_ts_specifier resolves specifiers that already include a file extension."""
        (tmp_path / "utils.js").write_bytes(b"export const x = 1;\n")
        parser = TSParser(repo="r", repo_root=str(tmp_path))
        resolved = parser._resolve_ts_specifier("./utils.js", "app.ts")
        assert resolved == "utils.js", f"Expected 'utils.js', got {resolved!r}"


# ── File collection ──────────────────────────────────────────────────────────


class TestCollectTsFiles:
    def test_collects_ts_jsx_files(self, tmp_path):
        """collect_ts_files picks up .ts, .tsx, .js, .jsx files."""
        for name in ["a.ts", "b.tsx", "c.js", "d.jsx", "e.py", "f.txt"]:
            (tmp_path / name).write_bytes(b"x = 1")

        files = collect_ts_files(str(tmp_path))
        basenames = {os.path.basename(f) for f in files}
        assert {"a.ts", "b.tsx", "c.js", "d.jsx"} == basenames

    def test_excludes_node_modules(self, tmp_path):
        """node_modules directory is excluded by default."""
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.ts").write_bytes(b"export const x = 1;")
        (tmp_path / "app.ts").write_bytes(b"export const y = 2;")

        files = collect_ts_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "app.ts" in basenames
        assert "lib.ts" not in basenames

    def test_recurses_into_subdirs(self, tmp_path):
        """collect_ts_files recurses into nested directories."""
        (tmp_path / "src" / "components").mkdir(parents=True)
        (tmp_path / "src" / "components" / "Button.tsx").write_bytes(b"export const Button = () => null;")

        files = collect_ts_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("Button.tsx")


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_unsupported_extension_returns_error(self, tmp_path):
        """Passing a .css file to TSParser returns an error, not an exception."""
        f = tmp_path / "style.css"
        f.write_bytes(b"body { color: red; }")
        parser = TSParser(repo="r", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))
        assert result.errors
        assert "Unsupported extension" in result.errors[0]

    def test_missing_tree_sitter_returns_graceful_error(self, tmp_path):
        """If tree-sitter is not installed, parse_file returns an error result, not a crash."""
        f = tmp_path / "app.tsx"
        f.write_bytes(b"const App = () => <div />;")
        parser = TSParser(repo="r", repo_root=str(tmp_path))

        # tree_sitter.Parser is imported lazily inside parse_file via
        # `from tree_sitter import Parser`.  Patching at the source module
        # exercises the ImportError guard without uninstalling the package.
        with patch("tree_sitter.Parser", side_effect=ImportError("no module named tree_sitter")):
            result = parser.parse_file(str(f))

        assert result.errors, "Expected at least one error when tree-sitter is unavailable"
        assert any("tree-sitter" in e or "tree_sitter" in e for e in result.errors), (
            f"Expected tree-sitter import error message, got: {result.errors}"
        )

    def test_handles_empty_file(self, tmp_path):
        """An empty .ts file produces a module entity and no errors."""
        f = tmp_path / "empty.ts"
        f.write_bytes(b"")
        parser = TSParser(repo="r", repo_root=str(tmp_path))
        result = parser.parse_file(str(f))
        modules = [e for e in result.entities if e.entity_type == "module"]
        assert len(modules) == 1
