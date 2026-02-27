"""Code indexer — orchestrates parsing and graph writes for a directory."""

import logging
import os
import time
from typing import Any, Optional

from smartmemory.code.models import CodeEntity, CodeRelation, ImportSymbol, IndexResult
from smartmemory.code.parser import CodeParser, collect_python_files, parse_file
from smartmemory.code.ts_parser import collect_ts_files

logger = logging.getLogger(__name__)


class SymbolTable:
    """Maps (importing_file, local_name) → item_id for cross-file call resolution.

    Built in two stages during ``CodeIndexer.index()``:

    1. **Register entities** — every parsed ``CodeEntity`` is registered under
       its fully-qualified dotted module path so that import aliases can point
       at it later.

    2. **Register imports** — each ``ImportSymbol`` from ``ParseResult`` binds
       a per-file local name to the item_id of the entity it refers to.

    Lookup during call resolution
    -----------------------------
    For a CALLS edge whose source is in ``importing_file`` and whose raw
    callee name is ``local_name``, call ``resolve(importing_file, local_name)``.
    Returns the resolved ``item_id`` or ``None`` when unresolvable.

    Supported import patterns
    -------------------------
    * ``from smartmemory.code.parser import CodeParser``
      → ``resolve("a/b.py", "CodeParser")`` → entity item_id for CodeParser

    * ``from smartmemory.code.parser import CodeParser as CP``
      → ``resolve("a/b.py", "CP")`` → same item_id

    * ``import smartmemory.code.parser``
      → ``resolve("a/b.py", "smartmemory.code.parser")`` → module item_id

    * ``import smartmemory.code.parser as p``
      → ``resolve("a/b.py", "p")`` → module item_id
      → ``resolve("a/b.py", "p.parse_file")`` → function item_id (if registered)
    """

    def __init__(self) -> None:
        # Primary index: (importing_file, local_name) → item_id
        self._table: dict[tuple[str, str], str] = {}
        # Module-scoped entity registry: module_path → {entity_name → item_id}
        # Used to resolve "alias.EntityName" attribute calls from module imports.
        self._module_entities: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Stage 1 — entity registration
    # ------------------------------------------------------------------

    def register_entity(self, entity: CodeEntity) -> None:
        """Record an entity in the module-entity registry.

        The entity's ``file_path`` is converted to a dotted module path so
        that import statements can look it up by module name.
        """
        if entity.entity_type == "module":
            # The module entity's name IS the dotted path already
            module_path = entity.name
        else:
            # Convert file_path to dotted module: "a/b/c.py" → "a.b.c"
            module_path = entity.file_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
            module_path = module_path.removesuffix(".__init__")

        if module_path not in self._module_entities:
            self._module_entities[module_path] = {}

        # For methods (e.g. "UserService.create_user") register both the
        # qualified and the bare method name so both styles of call resolve.
        self._module_entities[module_path][entity.name] = entity.item_id
        if "." in entity.name:
            bare_name = entity.name.rsplit(".", 1)[1]
            # Only register bare name if it is not already taken by another entity
            # (avoids false positives when two classes share a method name).
            self._module_entities[module_path].setdefault(bare_name, entity.item_id)

    # ------------------------------------------------------------------
    # Stage 2 — import binding
    # ------------------------------------------------------------------

    def register_import(self, sym: ImportSymbol) -> None:
        """Bind a local name in an importing file to a resolved item_id."""
        key = (sym.importing_file, sym.local_name)

        if sym.is_module_import:
            # "import foo.bar" or "import foo.bar as fb"
            # Bind local_name → the module entity's item_id (if it exists).
            mod_entities = self._module_entities.get(sym.module_path, {})
            if sym.module_path in mod_entities:
                self._table[key] = mod_entities[sym.module_path]
            # Also pre-register "local_name.Foo" for every entity in the module
            # so attribute calls ("p.CodeParser()") resolve without runtime info.
            for entity_name, item_id in mod_entities.items():
                attr_key = (sym.importing_file, f"{sym.local_name}.{entity_name}")
                self._table.setdefault(attr_key, item_id)
        else:
            # "from foo import Bar" or "from foo import Bar as B"
            mod_entities = self._module_entities.get(sym.module_path, {})
            resolved = mod_entities.get(sym.original_name)
            if resolved is not None:
                self._table[key] = resolved

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, importing_file: str, local_name: str) -> Optional[str]:
        """Resolve a local call name to an item_id.

        Args:
            importing_file: Relative file path of the caller (as stored on
                ``CodeEntity.file_path``).
            local_name: The raw callee name as it appears in source code
                (e.g. ``"CodeParser"``, ``"p.parse_file"``).

        Returns:
            Resolved ``item_id`` string, or ``None`` if not resolvable.
        """
        return self._table.get((importing_file, local_name))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of entries in the symbol table."""
        return len(self._table)


class CodeIndexer:
    """Index a Python codebase into the SmartMemory knowledge graph."""

    def __init__(self, graph: Any, repo: str, repo_root: str, exclude_dirs: Optional[set[str]] = None):
        """Initialize the indexer.

        Args:
            graph: SmartGraph instance (with scope provider for multi-tenancy)
            repo: Repository identifier (e.g., "smart-memory-service")
            repo_root: Absolute path to the repository root
            exclude_dirs: Directory names to skip during file collection
        """
        self.graph = graph
        self.repo = repo
        self.repo_root = os.path.abspath(repo_root)
        self.parser = CodeParser(repo=repo, repo_root=repo_root)
        self.exclude_dirs = exclude_dirs

    def index(self, languages: Optional[list[str]] = None) -> IndexResult:
        """Index the entire directory and write to graph.

        Args:
            languages: Language names to include. Supported values are
                ``"python"`` and ``"typescript"``. Defaults to
                ``["python"]`` for backwards compatibility.
                Pass ``["python", "typescript"]`` to also index
                ``.ts``, ``.tsx``, ``.js``, and ``.jsx`` files.
        """
        if languages is None:
            languages = ["python"]

        start = time.time()
        result = IndexResult(repo=self.repo)

        # Collect files for requested language groups
        all_files: list[str] = []
        if "python" in languages:
            py_files = collect_python_files(self.repo_root, self.exclude_dirs)
            logger.info("Found %d Python files in %s", len(py_files), self.repo_root)
            all_files.extend(py_files)

        if "typescript" in languages:
            ts_files = collect_ts_files(self.repo_root, self.exclude_dirs)
            logger.info("Found %d TS/JS files in %s", len(ts_files), self.repo_root)
            all_files.extend(ts_files)

        # --- Pass 1: parse all files ---
        all_entities: list[CodeEntity] = []
        all_relations: list[CodeRelation] = []
        all_import_symbols: list[ImportSymbol] = []

        for file_path in all_files:
            parse_result = parse_file(file_path, repo=self.repo, repo_root=self.repo_root)
            if parse_result.errors and not parse_result.entities:
                result.files_skipped += 1
            else:
                result.files_parsed += 1
            all_entities.extend(parse_result.entities)
            all_relations.extend(parse_result.relations)
            all_import_symbols.extend(getattr(parse_result, "import_symbols", []))
            result.errors.extend(parse_result.errors)

        # --- Build symbol table (CODE-DEV-5) ---
        symbol_table = self._build_symbol_table(all_entities, all_import_symbols)
        logger.debug("Symbol table built: %d entries for repo %s", symbol_table.size, self.repo)

        # --- Resolve cross-file CALLS edges ---
        all_relations = self._resolve_cross_file_calls(all_relations, symbol_table)

        # Build entity ID set for edge validation
        entity_ids = {e.item_id for e in all_entities}

        # Delete existing code nodes for this repo (clean slate)
        self._delete_existing(self.repo)

        # Write nodes in bulk
        bulk_nodes = [entity.to_properties() for entity in all_entities]
        try:
            result.entities_created = self.graph.add_nodes_bulk(bulk_nodes)
        except Exception as e:
            logger.error("Bulk node write failed for %s: %s", self.repo, e)
            result.errors.append(f"Bulk node write failed: {e}")

        # Write edges in bulk (only if both endpoints exist)
        bulk_edges = [
            (rel.source_id, rel.target_id, rel.relation_type, rel.properties)
            for rel in all_relations
            if rel.source_id in entity_ids and rel.target_id in entity_ids
        ]
        try:
            result.edges_created = self.graph.add_edges_bulk(bulk_edges)
        except Exception as e:
            logger.error("Bulk edge write failed for %s: %s", self.repo, e)
            result.errors.append(f"Bulk edge write failed: {e}")

        # Generate and store vector embeddings for all entities (CODE-DEV-2/C).
        # Embeddings land in the same ``semantic_memory`` collection queried by
        # ``_search_with_vector_embeddings``, so code nodes become searchable
        # immediately without any changes to the search path.
        result.embeddings_generated = self._generate_embeddings(all_entities)

        # Expose parsed entities on result (used by CODE-DEV-4 pattern seeding
        # via SmartMemory.ingest_code() → seed_patterns_from_code()).
        result.entities = all_entities

        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "Indexed %s: %d files, %d entities, %d edges, %d embeddings in %.2fs",
            self.repo,
            result.files_parsed,
            result.entities_created,
            result.edges_created,
            result.embeddings_generated,
            result.elapsed_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Embedding generation (CODE-DEV-2/C)
    # ------------------------------------------------------------------

    def _generate_embeddings(self, entities: list[CodeEntity]) -> int:
        """Generate and upsert vector embeddings for the given code entities.

        Uses the same ``VectorStore`` and ``create_embeddings`` that the
        pipeline's StoreStage uses.  Embeddings are upserted (idempotent on
        re-index) and linked to the code node via ``node_ids``.

        The embedding text is the composite string recommended by the design
        spec: ``name + entity_type + file_path + docstring``.  Code nodes have
        no ``content`` field, so ``content`` is explicitly excluded.

        Args:
            entities: List of parsed code entities to embed.

        Returns:
            Number of embeddings successfully generated and stored.
        """
        if not entities:
            return 0

        try:
            from smartmemory.plugins.embedding import create_embeddings
            from smartmemory.stores.vector.vector_store import VectorStore
        except ImportError as exc:
            logger.warning("Embedding dependencies unavailable, skipping embedding generation: %s", exc)
            return 0

        vector_store = VectorStore()
        count = 0

        # Determine workspace scope so code embeddings are visible in tenant-scoped search.
        # Falls back to is_global=True only when no workspace is configured (local/Lite mode).
        scope_filters = self.graph.get_scope_filters()
        workspace_id = scope_filters.get("workspace_id")
        is_global = workspace_id is None

        for entity in entities:
            embed_text = f"{entity.name} {entity.entity_type} {entity.file_path} {entity.docstring}".strip()
            if not embed_text:
                continue
            try:
                embedding = create_embeddings(embed_text)
                if embedding is None:
                    logger.debug("No embedding returned for entity %s", entity.item_id)
                    continue
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                metadata: dict = {
                    "memory_type": "code",
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "file_path": entity.file_path,
                    "repo": entity.repo,
                }
                if workspace_id is not None:
                    metadata["workspace_id"] = workspace_id
                vector_store.upsert(
                    item_id=entity.item_id,
                    embedding=embedding_list,
                    metadata=metadata,
                    node_ids=[entity.item_id],
                    is_global=is_global,
                )
                count += 1
            except Exception as exc:
                logger.warning("Embedding failed for entity %s: %s", entity.item_id, exc)

        logger.info("Generated %d/%d embeddings for repo %s", count, len(entities), self.repo)
        return count

    # ------------------------------------------------------------------
    # Symbol table helpers
    # ------------------------------------------------------------------

    def _build_symbol_table(
        self,
        entities: list[CodeEntity],
        import_symbols: list[ImportSymbol],
    ) -> SymbolTable:
        """Build a symbol table from all parsed entities and import statements.

        Two-phase construction:
        1. Register every entity so module-path lookups work.
        2. Bind every ImportSymbol to the resolved item_id.

        Args:
            entities: All ``CodeEntity`` objects from the parse pass.
            import_symbols: All ``ImportSymbol`` objects from the parse pass.

        Returns:
            Populated ``SymbolTable`` ready for call resolution.
        """
        table = SymbolTable()

        # Phase 1: register entities
        for entity in entities:
            table.register_entity(entity)

        # Phase 2: bind import symbols
        for sym in import_symbols:
            table.register_import(sym)

        return table

    def _resolve_cross_file_calls(
        self,
        relations: list[CodeRelation],
        symbol_table: SymbolTable,
    ) -> list[CodeRelation]:
        """Re-resolve CALLS edge targets using the symbol table.

        The parser's initial same-file guess (``code::{repo}::{file}::{name}``)
        is replaced with the correct cross-file ``item_id`` when the callee
        name is found in the symbol table for the caller's file.

        Non-CALLS edges and CALLS edges with no ``callee`` property are passed
        through unchanged.

        Args:
            relations: All relations from the parse pass.
            symbol_table: Populated symbol table.

        Returns:
            Updated relation list with cross-file CALLS targets resolved.
        """
        resolved_count = 0
        updated: list[CodeRelation] = []

        for rel in relations:
            if rel.relation_type != "CALLS":
                updated.append(rel)
                continue

            callee = rel.properties.get("callee")
            if not callee:
                updated.append(rel)
                continue

            # Derive the caller's file from its item_id:
            # item_id format: "code::{repo}::{file_path}::{name}"
            parts = rel.source_id.split("::", 3)
            if len(parts) < 3:
                updated.append(rel)
                continue
            caller_file = parts[2]  # relative file path

            resolved_id = symbol_table.resolve(caller_file, callee)
            if resolved_id is not None and resolved_id != rel.target_id:
                resolved_count += 1
                updated.append(
                    CodeRelation(
                        source_id=rel.source_id,
                        target_id=resolved_id,
                        relation_type=rel.relation_type,
                        properties=rel.properties,
                    )
                )
            else:
                updated.append(rel)

        if resolved_count:
            logger.debug("Resolved %d cross-file CALLS edges for repo %s", resolved_count, self.repo)

        return updated

    def _delete_existing(self, repo: str):
        """Delete all code nodes for a given repo before re-indexing.

        Applies workspace_id scoping from the graph's scope_provider
        to prevent cross-tenant deletion.
        """
        try:
            params: dict[str, Any] = {"repo": repo}
            where_clauses = ["n.repo = $repo"]

            # Inject workspace scope to prevent cross-tenant deletion
            scope_filters = self.graph.get_scope_filters()
            if scope_filters.get("workspace_id"):
                params["workspace_id"] = scope_filters["workspace_id"]
                where_clauses.append("n.workspace_id = $workspace_id")

            where = " AND ".join(where_clauses)
            query = f"MATCH (n:Code) WHERE {where} DETACH DELETE n"
            self.graph.execute_query(query, params)
            logger.info("Deleted existing code nodes for repo: %s", repo)
        except Exception as e:
            logger.warning("Failed to delete existing code nodes: %s", e)
