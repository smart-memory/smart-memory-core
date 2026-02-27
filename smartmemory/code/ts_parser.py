"""tree-sitter-based TypeScript/JavaScript source code parser.

Extracts classes, functions, React components, hooks, imports, and calls
from .ts, .tsx, .js, and .jsx files.

Validated against tree-sitter==0.25.2, tree-sitter-typescript==0.23.2,
tree-sitter-javascript==0.25.0.
"""

import os
from typing import Optional

from smartmemory.code.models import CodeEntity, CodeRelation, ParseResult

# Lazy-initialised language singletons — only paid if TS parsing is used.
_TSX_LANGUAGE = None
_TS_LANGUAGE = None
_JS_LANGUAGE = None


def _get_tsx_language():  # type: ignore[return]
    global _TSX_LANGUAGE
    if _TSX_LANGUAGE is None:
        from tree_sitter import Language
        import tree_sitter_typescript as tst

        _TSX_LANGUAGE = Language(tst.language_tsx())
    return _TSX_LANGUAGE


def _get_ts_language():  # type: ignore[return]
    global _TS_LANGUAGE
    if _TS_LANGUAGE is None:
        from tree_sitter import Language
        import tree_sitter_typescript as tst

        _TS_LANGUAGE = Language(tst.language_typescript())
    return _TS_LANGUAGE


def _get_js_language():  # type: ignore[return]
    global _JS_LANGUAGE
    if _JS_LANGUAGE is None:
        from tree_sitter import Language
        import tree_sitter_javascript as tsj

        _JS_LANGUAGE = Language(tsj.language())
    return _JS_LANGUAGE


# Extensions that share JS grammar (JSX dialects)
_JS_EXTS = {".js", ".jsx"}
# Extensions that use TS grammar (tsx or plain ts)
_TSX_EXT = ".tsx"
_TS_EXT = ".ts"

# Default directories to skip (mirrors parser.py)
DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    "out",
    "coverage",
}

# Express-style router HTTP methods
_EXPRESS_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "all"}


def _child_by_field(node, field: str):
    """Safe child_by_field_name — returns None when absent."""
    try:
        return node.child_by_field_name(field)
    except Exception:
        return None


def _node_text(node, source: bytes) -> str:
    """Extract raw source text for a node."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _has_jsx_descendant(node) -> bool:
    """Return True if any descendant of *node* is a JSX element or fragment."""
    stack = list(node.children)
    while stack:
        child = stack.pop()
        if child.type in ("jsx_element", "jsx_self_closing_element", "jsx_fragment"):
            return True
        stack.extend(child.children)
    return False


class TSParser:
    """Parse a TypeScript/JavaScript file into code entities and relationships."""

    def __init__(self, repo: str, repo_root: str):
        self.repo = repo
        self.repo_root = os.path.abspath(repo_root)

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single TS/JS file and extract entities + relations.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            ParseResult with entities, relations, and any errors.
        """
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, self.repo_root)
        result = ParseResult(file_path=rel_path)

        ext = os.path.splitext(abs_path)[1].lower()

        try:
            from tree_sitter import Parser

            if ext == _TSX_EXT:
                language = _get_tsx_language()
            elif ext == _TS_EXT:
                language = _get_ts_language()
            elif ext in _JS_EXTS:
                language = _get_js_language()
            else:
                result.errors.append(f"Unsupported extension for TSParser: {ext}")
                return result

            with open(abs_path, "rb") as fh:
                source = fh.read()
            parser = Parser(language)
            tree = parser.parse(source)
        except ImportError as e:
            result.errors.append(f"tree-sitter not installed: {e}")
            return result
        except Exception as e:
            result.errors.append(f"Error reading {rel_path}: {e}")
            return result

        root = tree.root_node
        if root.has_error:
            # Proceed anyway — partial parse is often still useful
            result.errors.append(f"Parse errors in {rel_path} (partial parse)")

        # Module entity — always created
        module_name = rel_path
        module_entity = CodeEntity(
            name=module_name,
            entity_type="module",
            file_path=rel_path,
            line_number=1,
            repo=self.repo,
        )
        result.entities.append(module_entity)

        # Walk the top-level program children
        self._walk(root, parent=module_entity, rel_path=rel_path, source=source, result=result, depth=0)

        return result

    # ── Main walker ──────────────────────────────────────────────────────────

    def _walk(self, node, parent: CodeEntity, rel_path: str, source: bytes, result: ParseResult, depth: int):
        """Recursively walk tree-sitter nodes and extract entities."""
        for child in node.children:
            ntype = child.type

            if ntype == "function_declaration":
                self._extract_function_decl(child, parent, rel_path, source, result, depth)

            elif ntype == "class_declaration":
                self._extract_class_decl(child, parent, rel_path, source, result, depth)

            elif ntype in ("lexical_declaration", "variable_declaration"):
                self._extract_variable_decl(child, parent, rel_path, source, result, depth)

            elif ntype == "method_definition":
                self._extract_method(child, parent, rel_path, source, result, depth)

            elif ntype == "export_statement":
                # Recurse into export wrappers — export function, export class, export const.
                # expression_statement is intentionally NOT included: recursing into it causes
                # every CALLS edge inside a function body to be emitted twice (once by
                # _collect_calls and once via _walk -> expression_statement -> call_expression).
                self._walk(child, parent, rel_path, source, result, depth)

            elif ntype == "import_statement":
                self._extract_import_stmt(child, parent, rel_path, source, result)

            # call_expression is handled exclusively by _collect_calls inside function/method
            # bodies.  Emitting CALLS from _walk would duplicate every edge because
            # _extract_function_decl calls both _collect_calls(body) and _walk(body).

            # Recurse into other block-like containers without creating an entity
            elif ntype in ("program", "statement_block", "class_body"):
                self._walk(child, parent, rel_path, source, result, depth)

    # ── Entity extractors ───────────────────────────────────────────────────

    def _extract_function_decl(
        self,
        node,
        parent: CodeEntity,
        rel_path: str,
        source: bytes,
        result: ParseResult,
        depth: int,
    ):
        """Extract a named function declaration: function foo() { }"""
        name_node = _child_by_field(node, "name")
        if not name_node:
            return

        name = _node_text(name_node, source)
        body = _child_by_field(node, "body")
        has_jsx = body is not None and _has_jsx_descendant(body)

        if has_jsx:
            entity_type = "component"
        elif name.startswith("use") and len(name) > 3 and name[3].isupper():
            entity_type = "hook"
        else:
            entity_type = "function"

        entity = CodeEntity(
            name=name,
            entity_type=entity_type,
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            repo=self.repo,
        )
        result.entities.append(entity)
        result.relations.append(
            CodeRelation(source_id=parent.item_id, target_id=entity.item_id, relation_type="DEFINES")
        )

        # Extract calls inside the function body
        if body:
            self._collect_calls(body, entity, rel_path, source, result)

        # Walk nested declarations (e.g. inner functions)
        if body:
            self._walk(body, entity, rel_path, source, result, depth + 1)

    def _extract_class_decl(
        self,
        node,
        parent: CodeEntity,
        rel_path: str,
        source: bytes,
        result: ParseResult,
        depth: int,
    ):
        """Extract a class declaration: class Foo { }"""
        name_node = _child_by_field(node, "name")
        if not name_node:
            return

        name = _node_text(name_node, source)
        entity = CodeEntity(
            name=name,
            entity_type="class",
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            repo=self.repo,
        )
        result.entities.append(entity)
        result.relations.append(
            CodeRelation(source_id=parent.item_id, target_id=entity.item_id, relation_type="DEFINES")
        )

        # Walk class body — methods live inside class_body
        body = _child_by_field(node, "body")
        if body:
            self._walk(body, entity, rel_path, source, result, depth + 1)

    def _extract_variable_decl(
        self,
        node,
        parent: CodeEntity,
        rel_path: str,
        source: bytes,
        result: ParseResult,
        depth: int,
    ):
        """Extract const/let/var arrow functions and function expressions.

        Handles:
          const Foo = () => <jsx />         → component
          const useFoo = () => ...          → hook
          const bar = () => ...             → function
          const bar = function() { }        → function
        """
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            var_name_node = _child_by_field(child, "name")
            value_node = _child_by_field(child, "value")

            if not var_name_node or not value_node:
                continue

            if value_node.type not in ("arrow_function", "function", "function_expression"):
                continue

            name = _node_text(var_name_node, source)

            # Determine body node — may be block statement or expression
            body = _child_by_field(value_node, "body")

            # Classify
            has_jsx = body is not None and _has_jsx_descendant(body)
            if not has_jsx and value_node.type == "arrow_function":
                # For concise arrow functions the "body" IS the expression
                has_jsx = _has_jsx_descendant(value_node)

            if has_jsx:
                entity_type = "component"
            elif name.startswith("use") and len(name) > 3 and name[3].isupper():
                entity_type = "hook"
            else:
                entity_type = "function"

            entity = CodeEntity(
                name=name,
                entity_type=entity_type,
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
                repo=self.repo,
            )
            result.entities.append(entity)
            result.relations.append(
                CodeRelation(source_id=parent.item_id, target_id=entity.item_id, relation_type="DEFINES")
            )

            # Extract calls inside this function
            if body:
                self._collect_calls(body, entity, rel_path, source, result)
            elif value_node:
                self._collect_calls(value_node, entity, rel_path, source, result)

    def _extract_method(
        self,
        node,
        parent: CodeEntity,
        rel_path: str,
        source: bytes,
        result: ParseResult,
        depth: int,
    ):
        """Extract a method_definition inside a class body."""
        name_node = _child_by_field(node, "name")
        if not name_node:
            return

        method_name = _node_text(name_node, source)
        # Skip getters/setters if they have special keywords (static, get, set)
        # but still record them as functions for completeness
        entity = CodeEntity(
            name=method_name,
            entity_type="function",
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            repo=self.repo,
        )
        result.entities.append(entity)
        result.relations.append(
            CodeRelation(source_id=parent.item_id, target_id=entity.item_id, relation_type="DEFINES")
        )

        body = _child_by_field(node, "body")
        if body:
            self._collect_calls(body, entity, rel_path, source, result)

    # ── Relation extractors ─────────────────────────────────────────────────

    def _resolve_ts_specifier(self, specifier: str, importing_file: str) -> str | None:
        """Resolve a TS/JS import specifier to a repo-relative file path.

        Returns None for bare package imports (no leading ./ or ../).
        Tries extensions: .ts, .tsx, .js, .jsx, /index.ts, /index.tsx, /index.js
        """
        if not specifier.startswith("."):
            return None  # external package — not in repo

        import_dir = os.path.dirname(os.path.join(self.repo_root, importing_file))
        base = os.path.normpath(os.path.join(import_dir, specifier))

        # Check if specifier already includes a recognized extension
        if os.path.isfile(base):
            return os.path.relpath(base, self.repo_root)

        for ext in (".ts", ".tsx", ".js", ".jsx"):
            candidate = base + ext
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, self.repo_root)

        for ext in ("/index.ts", "/index.tsx", "/index.js", "/index.jsx"):
            candidate = base + ext
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, self.repo_root)

        return None

    def _extract_import_stmt(self, node, module: CodeEntity, rel_path: str, source: bytes, result: ParseResult):
        """Extract import declarations as IMPORTS edges."""
        source_node = _child_by_field(node, "source")
        if not source_node:
            return

        specifier = _node_text(source_node, source).strip("'\"")
        resolved = self._resolve_ts_specifier(specifier, rel_path)
        if resolved is None:
            return  # external package or unresolvable — skip
        target_id = f"code::{self.repo}::{resolved}::{resolved}"
        result.relations.append(CodeRelation(source_id=module.item_id, target_id=target_id, relation_type="IMPORTS"))

    def _extract_call_expr(self, node, parent: CodeEntity, rel_path: str, source: bytes, result: ParseResult):
        """Emit a CALLS edge for a top-level call_expression."""
        callee = _child_by_field(node, "function")
        if not callee:
            return
        callee_name = _node_text(callee, source).split("\n")[0][:80]
        if callee_name:
            target_id = f"code::{self.repo}::{rel_path}::{callee_name}"
            result.relations.append(CodeRelation(source_id=parent.item_id, target_id=target_id, relation_type="CALLS"))

    # Node types that introduce a new function scope. _collect_calls stops at these
    # so that calls inside a nested function are not attributed to the outer caller.
    _FUNCTION_SCOPE_TYPES = frozenset({
        "function_declaration",
        "function",
        "function_expression",
        "arrow_function",
        "method_definition",
    })

    def _collect_calls(self, node, caller: CodeEntity, rel_path: str, source: bytes, result: ParseResult):
        """Walk a subtree and emit CALLS edges for every call_expression found.

        Stops at nested function/arrow-function scope boundaries so that calls inside
        inner functions are attributed only to their own entity, not to the outer caller.
        """
        stack = list(node.children)
        while stack:
            child = stack.pop()
            if child.type == "call_expression":
                callee = _child_by_field(child, "function")
                if callee:
                    callee_name = _node_text(callee, source).split("\n")[0][:80].strip()
                    if callee_name and not callee_name.startswith("_"):
                        target_id = f"code::{self.repo}::{rel_path}::{callee_name}"
                        result.relations.append(
                            CodeRelation(
                                source_id=caller.item_id,
                                target_id=target_id,
                                relation_type="CALLS",
                                properties={"callee": callee_name, "line": child.start_point[0] + 1},
                            )
                        )
                # Still recurse into call arguments to capture callback calls
                args = _child_by_field(child, "arguments")
                if args:
                    stack.extend(args.children)
            elif child.type in self._FUNCTION_SCOPE_TYPES:
                # Stop here — nested function calls belong to their own entity
                pass
            else:
                stack.extend(child.children)


# ── File collection ──────────────────────────────────────────────────────────

TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}


def collect_ts_files(directory: str, exclude_dirs: Optional[set] = None) -> list[str]:
    """Walk directory and collect TS/JS file paths, skipping excluded dirs."""
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS

    ts_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith(".egg-info")]
        for f in files:
            if os.path.splitext(f)[1].lower() in TS_EXTENSIONS:
                ts_files.append(os.path.join(root, f))
    return sorted(ts_files)
