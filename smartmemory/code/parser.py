"""AST-based Python source code parser.

Extracts classes, functions, imports, FastAPI routes, and pytest tests
from a single Python file.
"""

import ast
import os
from typing import Optional

from smartmemory.code.models import CodeEntity, CodeRelation, ImportSymbol, ParseResult

# FastAPI router method names
ROUTER_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}

# Default directories to skip
DEFAULT_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".tox", ".mypy_cache", ".pytest_cache"}


class CodeParser:
    """Parse a Python file into code entities and relationships."""

    def __init__(self, repo: str, repo_root: str):
        self.repo = repo
        self.repo_root = os.path.abspath(repo_root)

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single Python file and extract entities + relations.

        Args:
            file_path: Absolute path to the Python file.

        Returns:
            ParseResult with entities, relations, and any errors.
        """
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, self.repo_root)
        result = ParseResult(file_path=rel_path)

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            result.errors.append(f"SyntaxError in {rel_path}: {e}")
            return result
        except Exception as e:
            result.errors.append(f"Error reading {rel_path}: {e}")
            return result

        # Module entity (the file itself)
        module_name = rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
        # __init__.py → package name (e.g. smartmemory.code, not smartmemory.code.__init__)
        module_name = module_name.removesuffix(".__init__")
        module_entity = CodeEntity(
            name=module_name,
            entity_type="module",
            file_path=rel_path,
            line_number=1,
            repo=self.repo,
            docstring=self._get_docstring(tree),
        )
        result.entities.append(module_entity)

        # Walk top-level nodes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._extract_class(node, module_entity, rel_path, result)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(node, module_entity, rel_path, result)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._extract_import(node, module_entity, rel_path, result)

        # Resolve test→function links via naming convention
        self._link_tests(result)

        return result

    def _extract_class(self, node: ast.ClassDef, module: CodeEntity, rel_path: str, result: ParseResult):
        """Extract a class definition and its methods."""
        bases = [self._get_name(b) for b in node.bases if self._get_name(b)]
        is_test_class = node.name.startswith("Test")

        entity = CodeEntity(
            name=node.name,
            entity_type="test" if is_test_class else "class",
            file_path=rel_path,
            line_number=node.lineno,
            repo=self.repo,
            docstring=self._get_docstring(node),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            bases=bases,
        )
        result.entities.append(entity)

        # DEFINES edge: module → class
        result.relations.append(
            CodeRelation(
                source_id=module.item_id,
                target_id=entity.item_id,
                relation_type="DEFINES",
            )
        )

        # INHERITS edges
        for base_name in bases:
            base_id = self._resolve_entity_id(base_name, rel_path)
            if base_id:
                result.relations.append(
                    CodeRelation(
                        source_id=entity.item_id,
                        target_id=base_id,
                        relation_type="INHERITS",
                    )
                )

        # Extract methods — class is the parent (not module)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(child, entity, rel_path, result, class_name=node.name)

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent: CodeEntity,
        rel_path: str,
        result: ParseResult,
        class_name: Optional[str] = None,
    ):
        """Extract a function/method definition.

        Args:
            parent: The owning entity — module for top-level functions,
                    class entity for methods.
        """
        full_name = f"{class_name}.{node.name}" if class_name else node.name
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Detect FastAPI routes
        route_info = self._detect_route(node)
        is_test = node.name.startswith("test_")

        if route_info:
            entity_type = "route"
        elif is_test:
            entity_type = "test"
        else:
            entity_type = "function"

        entity = CodeEntity(
            name=full_name,
            entity_type=entity_type,
            file_path=rel_path,
            line_number=node.lineno,
            repo=self.repo,
            docstring=self._get_docstring(node),
            decorators=decorators,
            http_method=route_info[0] if route_info else "",
            http_path=route_info[1] if route_info else "",
        )
        result.entities.append(entity)

        # DEFINES edge: parent → function (class→method or module→function)
        result.relations.append(
            CodeRelation(
                source_id=parent.item_id,
                target_id=entity.item_id,
                relation_type="DEFINES",
            )
        )

        # Extract calls within function body
        self._extract_calls(node, entity, rel_path, result)

    def _extract_import(
        self, node: ast.Import | ast.ImportFrom, module: CodeEntity, rel_path: str, result: ParseResult
    ):
        """Extract import statements as IMPORTS edges and ImportSymbol records.

        ImportSymbol entries are used by CodeIndexer to build the cross-file
        symbol table for CALLS edge resolution (CODE-DEV-5).
        """
        if isinstance(node, ast.Import):
            for alias in node.names:
                target_module = alias.name
                local_name = alias.asname if alias.asname else alias.name
                target_id = f"code::{self.repo}::{self._module_to_path(target_module)}::{target_module}"
                result.relations.append(
                    CodeRelation(
                        source_id=module.item_id,
                        target_id=target_id,
                        relation_type="IMPORTS",
                        properties={"names": local_name},
                    )
                )
                # Record module import symbol so "local_name.Foo" can be resolved.
                result.import_symbols.append(
                    ImportSymbol(
                        importing_file=rel_path,
                        local_name=local_name,
                        module_path=target_module,
                        original_name=target_module,
                        is_module_import=True,
                    )
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            target_module = node.module
            target_id = f"code::{self.repo}::{self._module_to_path(target_module)}::{target_module}"
            names_str = ",".join(a.name for a in node.names)
            result.relations.append(
                CodeRelation(
                    source_id=module.item_id,
                    target_id=target_id,
                    relation_type="IMPORTS",
                    properties={"names": names_str},
                )
            )
            # Record each imported name as an ImportSymbol.
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                result.import_symbols.append(
                    ImportSymbol(
                        importing_file=rel_path,
                        local_name=local_name,
                        module_path=target_module,
                        original_name=alias.name,
                        is_module_import=False,
                    )
                )

    def _extract_calls(self, func_node: ast.AST, caller: CodeEntity, rel_path: str, result: ParseResult):
        """Extract function calls within a function body.

        Emits CALLS edges with a *same-file* target_id as an initial guess.
        The ``callee`` property stores the raw call name so that
        ``CodeIndexer`` can re-resolve cross-file targets using the symbol
        table built from all files' import statements (CODE-DEV-5).
        """
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            callee_name = self._get_call_name(node)
            if not callee_name or callee_name.split(".")[-1].startswith("_"):
                continue
            # Initial same-file guess — will be overridden by the indexer's
            # symbol-table resolution pass when the callee is imported.
            target_id = f"code::{self.repo}::{rel_path}::{callee_name}"
            result.relations.append(
                CodeRelation(
                    source_id=caller.item_id,
                    target_id=target_id,
                    relation_type="CALLS",
                    properties={"line": getattr(node, "lineno", 0), "callee": callee_name},
                )
            )

    def _link_tests(self, result: ParseResult):
        """Link test functions to their tested functions via naming convention.

        test_foo → foo, TestFoo.test_create → create (strips class prefix too)
        """
        test_entities = [e for e in result.entities if e.entity_type == "test" and "test_" in e.name]
        non_test_names = {e.name: e for e in result.entities if e.entity_type not in ("test", "module")}

        for test in test_entities:
            # Handle class-qualified names: TestFoo.test_bar → bar
            base_name = test.name
            if "." in base_name:
                base_name = base_name.rsplit(".", 1)[1]  # Take method name only
            if not base_name.startswith("test_"):
                continue
            tested_name = base_name.removeprefix("test_")

            # Try exact match: test_foo → foo
            if tested_name in non_test_names:
                target = non_test_names[tested_name]
                result.relations.append(
                    CodeRelation(
                        source_id=test.item_id,
                        target_id=target.item_id,
                        relation_type="TESTS",
                        properties={"convention": "name_match"},
                    )
                )
                continue
            # Try class-qualified match: test_create_user → UserService.create_user
            for fname, entity in non_test_names.items():
                if "." in fname and fname.rsplit(".", 1)[1] == tested_name:
                    result.relations.append(
                        CodeRelation(
                            source_id=test.item_id,
                            target_id=entity.item_id,
                            relation_type="TESTS",
                            properties={"convention": "name_match"},
                        )
                    )
                    break

    def _detect_route(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[tuple[str, str]]:
        """Detect FastAPI route decorator. Returns (method, path) or None."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                method = decorator.func.attr
                if method in ROUTER_METHODS and decorator.args:
                    path_arg = decorator.args[0]
                    if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                        return (method.upper(), path_arg.value)
        return None

    def _get_docstring(self, node: ast.AST) -> str:
        """Extract first line of docstring from a node."""
        try:
            ds = ast.get_docstring(node)
            if ds:
                return ds.split("\n")[0].strip()
        except Exception:
            pass
        return ""

    def _get_name(self, node: ast.expr) -> str:
        """Get name from an AST expression (for base classes, etc)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""

    def _get_call_name(self, node: ast.Call) -> str:
        """Get function name from a Call node.

        Returns the fully-qualified dotted name so that module-import style
        calls (``import foo as f; f.Bar()``) produce ``"f.Bar"`` rather than
        just ``"Bar"``.  This allows ``_resolve_cross_file_calls`` to match
        against the symbol table keys registered by ``register_import`` for
        module imports.
        """

        def _walk(n: ast.expr) -> str:
            if isinstance(n, ast.Name):
                return n.id
            elif isinstance(n, ast.Attribute):
                parent = _walk(n.value)
                return f"{parent}.{n.attr}" if parent else n.attr
            return ""

        return _walk(node.func)

    def _resolve_entity_id(self, name: str, current_file: str) -> Optional[str]:
        """Try to resolve an entity name to an item_id within the same file."""
        return f"code::{self.repo}::{current_file}::{name}"

    def _module_to_path(self, module_name: str) -> str:
        """Convert a Python module name to a relative file path guess."""
        return module_name.replace(".", "/") + ".py"


def parse_file(file_path: str, repo: str, repo_root: str) -> ParseResult:
    """Dispatch to Python or TS/JS parser based on file extension.

    Args:
        file_path: Absolute path to the source file.
        repo: Repository identifier (e.g., ``"smart-memory-service"``).
        repo_root: Absolute path to the repository root, used to derive
            relative paths stored on ``CodeEntity.file_path``.

    Returns:
        ``ParseResult`` populated by the appropriate language parser.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".py":
        return CodeParser(repo=repo, repo_root=repo_root).parse_file(file_path)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        from smartmemory.code.ts_parser import TSParser

        return TSParser(repo=repo, repo_root=repo_root).parse_file(file_path)
    else:
        result = ParseResult(file_path=file_path)
        result.errors.append(f"Unsupported extension: {ext}")
        return result


def collect_python_files(directory: str, exclude_dirs: Optional[set[str]] = None) -> list[str]:
    """Walk directory and collect .py file paths, skipping excluded dirs."""
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS

    py_files = []
    for root, dirs, files in os.walk(directory):
        # Filter excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return sorted(py_files)
