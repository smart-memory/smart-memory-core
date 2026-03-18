"""Streaming JSONL corpus writer with checksum computation."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import IO

from smartmemory.corpus.format import CorpusHeader, CorpusRecord

logger = logging.getLogger(__name__)


class CorpusWriter:
    """Writes SmartMemory corpus JSONL files with streaming checksum.

    Usage:
        writer = CorpusWriter("output.jsonl", source="my-export", domain="tech")
        with writer:
            writer.write_record(record1)
            writer.write_record(record2)
        # On close, header is rewritten with final item_count and checksum.
    """

    def __init__(
        self,
        path: str | Path,
        source: str,
        domain: str = "",
    ):
        self._path = Path(path)
        self._source = source
        self._domain = domain
        self._hasher = hashlib.sha256()
        self._count = 0
        self._file: IO[str] | None = None
        self._header_len: int = 0

    def __enter__(self) -> CorpusWriter:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the file and write a placeholder header."""
        self._file = open(self._path, "w", encoding="utf-8")
        # Write placeholder header — will be overwritten on close
        placeholder = CorpusHeader(source=self._source, domain=self._domain)
        header_line = placeholder.to_json() + "\n"
        self._header_len = len(header_line.encode("utf-8"))
        self._file.write(header_line)

    def write_record(self, record: CorpusRecord) -> None:
        """Write a single data record and update the running checksum."""
        if not self._file:
            raise RuntimeError("Writer not open — use 'with CorpusWriter(...):' or call open()")
        line = record.to_json() + "\n"
        self._file.write(line)
        self._hasher.update(line.encode("utf-8"))
        self._count += 1

    def close(self) -> None:
        """Finalize: rewrite header with item_count and checksum, then close."""
        if not self._file:
            return
        self._file.close()
        self._file = None

        # Rewrite the header line in place with final counts
        final_header = CorpusHeader(
            source=self._source,
            domain=self._domain,
            item_count=self._count,
            checksum=f"sha256:{self._hasher.hexdigest()}",
        )

        # Read all lines, replace first, write back
        lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        if lines:
            lines[0] = final_header.to_json() + "\n"
        self._path.write_text("".join(lines), encoding="utf-8")

        logger.info("Wrote %d records to %s", self._count, self._path)

    @property
    def count(self) -> int:
        return self._count
