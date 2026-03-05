import random
from dataclasses import dataclass
from datetime import datetime, UTC

from smartmemory.models.base import MemoryBaseModel
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata


@dataclass
class StaleMemoryConfig(MemoryBaseModel):
    enabled: bool = True
    batch_size: int = 100  # non-code memories checked per run (random sample)


class StaleMemoryEvolver(EvolverPlugin):
    """
    Marks non-code memories stale when their referenced source files
    have been re-indexed at a different commit hash.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="stale_memory",
            version="1.0.0",
            author="SmartMemory Team",
            description="Marks memories stale when referenced source code changes",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
        )

    def evolve(self, memory, logger=None):
        cfg = self.config
        if not hasattr(cfg, "enabled"):
            raise TypeError(
                "StaleMemoryEvolver requires a typed config with 'enabled'. "
                "Provide StaleMemoryConfig or a compatible typed config."
            )
        with trace_span("pipeline.evolve.stale_memory", {}):
            if not cfg.enabled:
                return

            # Step 1: Build (repo, file_path) → current commit_hash from live code nodes.
            # search_nodes() returns MemoryItem list; use .metadata.get(), not dict indexing.
            code_nodes = memory._graph.search_nodes({"memory_type": "code"})
            commit_by_repo_path: dict[tuple[str, str], str] = {}
            for node in code_nodes:
                repo = node.metadata.get("repo", "")
                fp = node.metadata.get("file_path", "")
                ch = node.metadata.get("commit_hash", "")
                if repo and fp and ch:
                    commit_by_repo_path[(repo, fp)] = ch

            if not commit_by_repo_path:
                return  # no code indexed — nothing to check

            # Step 2: Sample non-code memories that carry source_code_refs.
            # Query each type separately; search_nodes filters by single property.
            candidates = []
            for mtype in ("episodic", "procedural", "semantic", "zettel", "working"):
                candidates.extend(memory._graph.search_nodes({"memory_type": mtype}))

            # Filter to items with source_code_refs using per-item exception handling,
            # so a bad item during filtering never excludes the rest of the pool.
            with_refs = []
            for m in candidates:
                try:
                    if m.metadata.get("source_code_refs"):
                        with_refs.append(m)
                except Exception as e:
                    if logger:
                        logger.warning("stale_memory evolver: skipping item %s: %s", getattr(m, "item_id", "?"), e)

            sample = random.sample(with_refs, min(cfg.batch_size, len(with_refs)))

            # Step 3: Compare snapshots vs current commits. Mark stale on mismatch.
            for mem_item in sample:
                try:
                    for ref in mem_item.metadata.get("source_code_refs", []):
                        repo = ref.get("repo", "")
                        fp = ref.get("file_path", "")
                        snapped_commit = ref.get("commit_hash", "")
                        current_commit = commit_by_repo_path.get((repo, fp), "")
                        if repo and fp and snapped_commit and current_commit and snapped_commit != current_commit:
                            self._mark_stale(memory, mem_item, repo, fp, snapped_commit, current_commit)
                            break  # one stale ref is enough per memory
                except Exception as e:
                    if logger:
                        logger.warning("stale_memory evolver: skipping item %s: %s", mem_item.item_id, e)
                    continue

    def _mark_stale(self, memory, item, repo, file_path, old_commit, new_commit):
        item.metadata["stale"] = True
        item.metadata["stale_since"] = datetime.now(UTC).isoformat()
        item.metadata["stale_reason"] = (
            f"{repo}::{file_path}: {old_commit[:8]} → {new_commit[:8]}"
        )
        memory.update(item)
