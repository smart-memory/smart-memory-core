"""Streaming JSONL corpus reader with checkpoint tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from smartmemory.corpus.format import CorpusHeader, CorpusRecord

logger = logging.getLogger(__name__)


class CheckpointState:
    """Tracks import progress for resume support."""

    def __init__(self, checkpoint_path: Path):
        self._path = checkpoint_path
        self.last_line: int = 0
        self.imported: int = 0
        self.skipped: int = 0
        self.errors: int = 0

    def load(self) -> bool:
        """Load checkpoint from disk. Returns True if checkpoint existed."""
        if not self._path.exists():
            return False
        try:
            data = json.loads(self._path.read_text())
            self.last_line = data.get("last_line", 0)
            self.imported = data.get("imported", 0)
            self.skipped = data.get("skipped", 0)
            self.errors = data.get("errors", 0)
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt checkpoint %s: %s", self._path, e)
            return False

    def save(self) -> None:
        """Persist checkpoint to disk."""
        self._path.write_text(
            json.dumps(
                {
                    "last_line": self.last_line,
                    "imported": self.imported,
                    "skipped": self.skipped,
                    "errors": self.errors,
                }
            )
        )

    def mark_complete(self) -> None:
        """Rename checkpoint to .complete.json."""
        complete_path = self._path.with_suffix("").with_suffix(".complete.json")
        if self._path.exists():
            self._path.rename(complete_path)


class CorpusReader:
    """Streaming reader for SmartMemory corpus JSONL files.

    Handles header parsing, record iteration, and checkpoint-based resume.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self._path}")
        self._header: CorpusHeader | None = None

    @property
    def path(self) -> Path:
        return self._path

    def read_header(self) -> CorpusHeader:
        """Read and validate the header (first line)."""
        with open(self._path) as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("Empty corpus file")
            self._header = CorpusHeader.parse(first_line)
            return self._header

    def iter_records(
        self,
        skip_lines: int = 0,
        domain_filter: str | None = None,
    ) -> Iterator[tuple[int, CorpusRecord]]:
        """Yield (line_number, record) tuples from the corpus.

        Args:
            skip_lines: Number of data lines to skip (for resume). 0 = start from beginning.
            domain_filter: If set, only yield records whose metadata.domain contains this string.

        Yields:
            (line_number, CorpusRecord) — line_number is 1-indexed (line 1 = header, line 2 = first record).
        """
        with open(self._path) as f:
            # Skip header
            f.readline()

            for line_idx, raw_line in enumerate(f, start=2):
                data_idx = line_idx - 2  # 0-indexed data line number
                if data_idx < skip_lines:
                    continue

                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    record = CorpusRecord.parse(raw_line)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Skipping corrupt record at line %d: %s", line_idx, e)
                    continue

                if domain_filter and domain_filter not in record.metadata.get("domain", ""):
                    continue

                yield line_idx, record

    def count_records(self) -> int:
        """Count data records (excluding header). Reads full file."""
        count = 0
        with open(self._path) as f:
            f.readline()  # skip header
            for line in f:
                if line.strip():
                    count += 1
        return count
