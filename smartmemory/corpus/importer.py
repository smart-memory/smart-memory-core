"""Corpus import orchestrator: reads JSONL, ingests/adds records with checkpoints and progress."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from smartmemory.corpus.format import CorpusRecord
from smartmemory.corpus.reader import CheckpointState, CorpusReader

logger = logging.getLogger(__name__)


class ImportStats:
    """Tracks import run statistics."""

    def __init__(self) -> None:
        self.imported: int = 0
        self.skipped: int = 0
        self.errors: int = 0
        self.start_time: float = 0.0

    @property
    def total(self) -> int:
        return self.imported + self.skipped + self.errors

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time if self.start_time else 0.0

    @property
    def rate(self) -> float:
        return self.imported / self.elapsed if self.elapsed > 0 else 0.0


class CorpusImporter:
    """Orchestrates corpus JSONL import into SmartMemory.

    Supports two modes:
    - full: runs SmartMemory.ingest() per record (full pipeline with LLM extraction)
    - direct: runs SmartMemory.add() and bulk-inserts pre-extracted entities/relations

    Features: checkpoint/resume, progress callback, dry-run validation.
    """

    def __init__(
        self,
        smart_memory: Any,
        mode: str = "full",
        batch_size: int = 100,
        domain_filter: str | None = None,
    ):
        if mode not in ("full", "direct"):
            raise ValueError(f"mode must be 'full' or 'direct', got {mode!r}")
        self._sm = smart_memory
        self._mode = mode
        self._batch_size = batch_size
        self._domain_filter = domain_filter

    def run(
        self,
        corpus_path: str | Path,
        resume: bool = False,
        dry_run: bool = False,
        progress_callback: Any | None = None,
    ) -> ImportStats:
        """Import a corpus JSONL file.

        Args:
            corpus_path: Path to the .jsonl corpus file.
            resume: If True, resume from last checkpoint.
            dry_run: If True, validate only — don't import.
            progress_callback: Called with (stats, record) after each record.

        Returns:
            ImportStats with final counts.
        """
        reader = CorpusReader(corpus_path)
        header = reader.read_header()
        logger.info(
            "Corpus: source=%s, domain=%s, declared_count=%d",
            header.source,
            header.domain,
            header.item_count,
        )

        checkpoint_path = Path(str(corpus_path) + ".checkpoint.json")
        checkpoint = CheckpointState(checkpoint_path)
        skip_lines = 0
        stats = ImportStats()

        if resume and checkpoint.load():
            skip_lines = checkpoint.last_line
            stats.imported = checkpoint.imported
            stats.skipped = checkpoint.skipped
            stats.errors = checkpoint.errors
            logger.info("Resuming from line %d (imported=%d)", skip_lines, stats.imported)

        stats.start_time = time.monotonic()
        batch_count = 0

        for line_num, record in reader.iter_records(skip_lines=skip_lines, domain_filter=self._domain_filter):
            if dry_run:
                stats.imported += 1
            else:
                try:
                    self._import_record(record)
                    stats.imported += 1
                except Exception as e:
                    logger.warning("Error importing line %d: %s", line_num, e)
                    stats.errors += 1

            checkpoint.last_line = line_num - 1  # data line index (0-based)
            checkpoint.imported = stats.imported
            checkpoint.skipped = stats.skipped
            checkpoint.errors = stats.errors
            batch_count += 1

            if progress_callback:
                progress_callback(stats, record)

            # Checkpoint every batch_size records
            if batch_count >= self._batch_size and not dry_run:
                checkpoint.save()
                batch_count = 0

        # Final checkpoint
        if not dry_run:
            checkpoint.mark_complete()

        logger.info(
            "Import %s: %d imported, %d errors, %.1f items/sec",
            "validated (dry-run)" if dry_run else "complete",
            stats.imported,
            stats.errors,
            stats.rate,
        )
        return stats

    def _import_record(self, record: CorpusRecord) -> None:
        """Import a single record using the configured mode."""
        if self._mode == "full":
            self._import_full(record)
        else:
            self._import_direct(record)

    def _import_full(self, record: CorpusRecord) -> None:
        """Full pipeline import via SmartMemory.ingest()."""
        self._sm.ingest(
            {"content": record.content, "memory_type": record.memory_type},
            **record.metadata,
        )

    def _import_direct(self, record: CorpusRecord) -> None:
        """Direct import: add() for content, bulk insert for entities/relations."""
        from smartmemory.models.memory_item import MemoryItem

        item = MemoryItem(content=record.content, memory_type=record.memory_type)
        if record.metadata:
            item.metadata.update(record.metadata)

        item_id = self._sm.add(item)

        if record.has_extractions and hasattr(self._sm, "graph"):
            graph = self._sm.graph
            if record.entities:
                nodes = []
                for entity in record.entities:
                    node: dict[str, Any] = {
                        "name": entity.name,
                        "entity_type": entity.type,
                        "source_item": item_id,
                    }
                    if entity.qid:
                        node["qid"] = entity.qid
                    nodes.append(node)
                graph.add_nodes_bulk(nodes)

            if record.relations:
                edges = []
                for rel in record.relations:
                    edges.append((rel.source, rel.target, rel.relation, {"source_item": item_id}))
                graph.add_edges_bulk(edges)
