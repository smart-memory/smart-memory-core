"""Seed pack system: manifest validation, install orchestration, and idempotent import.

A seed pack is a directory containing:
- manifest.json — metadata, version, checksums
- corpus.jsonl — corpus JSONL (Phase 1 format)
- patterns.jsonl (optional) — EntityRuler patterns
- entities.jsonl (optional) — PublicEntity JSONL for grounding
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PackManifest:
    """Seed pack manifest (manifest.json)."""

    name: str
    version: str
    description: str = ""
    domain: str = ""
    min_smartmemory_version: str = ""
    corpus_checksum: str = ""
    files: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.version:
            errors.append("version is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "domain": self.domain,
            "min_smartmemory_version": self.min_smartmemory_version,
            "corpus_checksum": self.corpus_checksum,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackManifest:
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            domain=data.get("domain", ""),
            min_smartmemory_version=data.get("min_smartmemory_version", ""),
            corpus_checksum=data.get("corpus_checksum", ""),
            files=data.get("files", {}),
        )

    @classmethod
    def load(cls, path: Path) -> PackManifest:
        data = json.loads((path / "manifest.json").read_text())
        return cls.from_dict(data)


@dataclass
class InstallRecord:
    """Tracks which packs are installed."""

    name: str
    version: str
    installed_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "installed_at": self.installed_at}


class InstalledPacks:
    """Manages the installed.json registry."""

    def __init__(self, data_dir: Path):
        self._path = data_dir / "installed_packs.json"
        self._packs: dict[str, InstallRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for name, info in data.items():
                self._packs[name] = InstallRecord(**info)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt installed_packs.json, starting fresh")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {name: record.to_dict() for name, record in self._packs.items()},
                indent=2,
            )
        )

    def is_installed(self, name: str, version: str) -> bool:
        record = self._packs.get(name)
        return record is not None and record.version == version

    def record_install(self, name: str, version: str) -> None:
        from datetime import datetime, timezone

        self._packs[name] = InstallRecord(
            name=name,
            version=version,
            installed_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save()

    def list_installed(self) -> list[InstallRecord]:
        return list(self._packs.values())


class SeedPack:
    """Manages a single seed pack: validation, content loading, and installation."""

    def __init__(self, pack_dir: str | Path):
        self._dir = Path(pack_dir)
        self._manifest: PackManifest | None = None

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def manifest(self) -> PackManifest:
        if not self._manifest:
            self._manifest = PackManifest.load(self._dir)
        return self._manifest

    def validate(self) -> list[str]:
        """Validate the pack structure and manifest."""
        errors: list[str] = []
        manifest_path = self._dir / "manifest.json"
        if not manifest_path.exists():
            errors.append("manifest.json not found")
            return errors

        errors.extend(self.manifest.validate())

        # Check that at least corpus.jsonl exists
        if not (self._dir / "corpus.jsonl").exists():
            errors.append("corpus.jsonl not found")

        return errors

    def install(
        self,
        smart_memory: Any,
        installed_packs: InstalledPacks,
        mode: str = "direct",
        skip_patterns: bool = False,
        skip_entities: bool = False,
        progress_callback: Any | None = None,
    ) -> dict[str, int]:
        """Install the seed pack into SmartMemory.

        Returns:
            Dict with counts: {"corpus": N, "patterns": N, "entities": N}
        """
        manifest = self.manifest

        # Idempotent: skip if same version already installed
        if installed_packs.is_installed(manifest.name, manifest.version):
            logger.info("Pack %s v%s already installed, skipping", manifest.name, manifest.version)
            return {"corpus": 0, "patterns": 0, "entities": 0}

        counts: dict[str, int] = {"corpus": 0, "patterns": 0, "entities": 0}

        # 1. Import entities.jsonl → grounding store
        entities_path = self._dir / "entities.jsonl"
        if entities_path.exists() and not skip_entities:
            counts["entities"] = self._import_entities(smart_memory, entities_path)

        # 2. Import patterns.jsonl → PatternManager
        patterns_path = self._dir / "patterns.jsonl"
        if patterns_path.exists() and not skip_patterns:
            counts["patterns"] = self._import_patterns(smart_memory, patterns_path)

        # 3. Import corpus.jsonl → memories
        corpus_path = self._dir / "corpus.jsonl"
        if corpus_path.exists():
            from smartmemory.corpus.importer import CorpusImporter

            importer = CorpusImporter(smart_memory=smart_memory, mode=mode, batch_size=100)
            stats = importer.run(corpus_path, progress_callback=progress_callback)
            counts["corpus"] = stats.imported

        installed_packs.record_install(manifest.name, manifest.version)
        logger.info(
            "Installed pack %s v%s: %d corpus, %d patterns, %d entities",
            manifest.name,
            manifest.version,
            counts["corpus"],
            counts["patterns"],
            counts["entities"],
        )
        return counts

    def _import_entities(self, smart_memory: Any, path: Path) -> int:
        """Load PublicEntity JSONL into the grounding store."""
        count = 0
        if hasattr(smart_memory, "_public_knowledge_store"):
            store = smart_memory._public_knowledge_store
            if store and hasattr(store, "load_snapshot"):
                store.load_snapshot(str(path))
                # Count lines for reporting
                with open(path) as f:
                    count = sum(1 for line in f if line.strip())
        return count

    def _import_patterns(self, smart_memory: Any, path: Path) -> int:
        """Load EntityRuler patterns from JSONL."""
        count = 0
        patterns: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    name = data.get("name", "")
                    label = data.get("label", "")
                    if name and label:
                        patterns[name] = label
                        count += 1
                except json.JSONDecodeError:
                    continue

        if patterns and hasattr(smart_memory, "pattern_manager"):
            pm = smart_memory.pattern_manager
            if pm:
                pm.add_patterns(patterns, source="seed-pack", initial_count=2)

        return count
