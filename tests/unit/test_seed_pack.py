"""Unit tests for seed pack system."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from smartmemory.corpus.format import CorpusRecord
from smartmemory.corpus.pack import InstalledPacks, PackManifest, SeedPack
from smartmemory.corpus.writer import CorpusWriter


def _create_pack(tmp_path: Path, name: str = "test-pack", version: str = "1.0.0") -> Path:
    """Create a minimal seed pack directory."""
    pack_dir = tmp_path / name
    pack_dir.mkdir()

    # Manifest
    manifest = {
        "name": name,
        "version": version,
        "description": "Test pack",
        "domain": "test",
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest))

    # Corpus JSONL
    with CorpusWriter(pack_dir / "corpus.jsonl", source=name, domain="test") as writer:
        writer.write_record(CorpusRecord(content="Test memory one"))
        writer.write_record(CorpusRecord(content="Test memory two"))

    return pack_dir


class TestPackManifest:
    def test_load(self, tmp_path):
        pack_dir = _create_pack(tmp_path)
        manifest = PackManifest.load(pack_dir)
        assert manifest.name == "test-pack"
        assert manifest.version == "1.0.0"

    def test_validate_missing_name(self):
        m = PackManifest(name="", version="1.0")
        assert "name is required" in m.validate()

    def test_validate_missing_version(self):
        m = PackManifest(name="x", version="")
        assert "version is required" in m.validate()

    def test_round_trip(self):
        m = PackManifest(name="test", version="1.0", domain="tech")
        d = m.to_dict()
        m2 = PackManifest.from_dict(d)
        assert m2.name == "test"
        assert m2.domain == "tech"


class TestInstalledPacks:
    def test_record_and_check(self, tmp_path):
        installed = InstalledPacks(tmp_path)
        assert not installed.is_installed("test", "1.0")

        installed.record_install("test", "1.0")
        assert installed.is_installed("test", "1.0")
        assert not installed.is_installed("test", "2.0")

    def test_persistence(self, tmp_path):
        installed1 = InstalledPacks(tmp_path)
        installed1.record_install("test", "1.0")

        installed2 = InstalledPacks(tmp_path)
        assert installed2.is_installed("test", "1.0")

    def test_list_installed(self, tmp_path):
        installed = InstalledPacks(tmp_path)
        installed.record_install("pack-a", "1.0")
        installed.record_install("pack-b", "2.0")
        records = installed.list_installed()
        assert len(records) == 2
        names = {r.name for r in records}
        assert names == {"pack-a", "pack-b"}


class TestSeedPack:
    def test_validate_valid_pack(self, tmp_path):
        pack_dir = _create_pack(tmp_path)
        pack = SeedPack(pack_dir)
        assert pack.validate() == []

    def test_validate_missing_manifest(self, tmp_path):
        pack_dir = tmp_path / "bad-pack"
        pack_dir.mkdir()
        pack = SeedPack(pack_dir)
        errors = pack.validate()
        assert "manifest.json not found" in errors

    def test_validate_missing_corpus(self, tmp_path):
        pack_dir = tmp_path / "no-corpus"
        pack_dir.mkdir()
        (pack_dir / "manifest.json").write_text(json.dumps({"name": "x", "version": "1.0"}))
        pack = SeedPack(pack_dir)
        errors = pack.validate()
        assert "corpus.jsonl not found" in errors

    def test_install_idempotent(self, tmp_path):
        pack_dir = _create_pack(tmp_path)
        pack = SeedPack(pack_dir)
        installed = InstalledPacks(tmp_path / "data")

        sm = MagicMock()
        sm.add.return_value = "item-123"
        sm.graph = MagicMock()

        # First install
        counts = pack.install(sm, installed, mode="direct")
        assert counts["corpus"] == 2

        # Second install — idempotent
        counts2 = pack.install(sm, installed, mode="direct")
        assert counts2["corpus"] == 0

    def test_install_with_patterns(self, tmp_path):
        pack_dir = _create_pack(tmp_path)

        # Add patterns file
        patterns = [
            {"name": "Python", "label": "Language"},
            {"name": "React", "label": "Framework"},
        ]
        with open(pack_dir / "patterns.jsonl", "w") as f:
            for p in patterns:
                f.write(json.dumps(p) + "\n")

        pack = SeedPack(pack_dir)
        installed = InstalledPacks(tmp_path / "data")

        sm = MagicMock()
        sm.add.return_value = "item-123"
        sm.graph = MagicMock()
        sm.pattern_manager = MagicMock()

        counts = pack.install(sm, installed, mode="direct")
        assert counts["patterns"] == 2
        sm.pattern_manager.add_patterns.assert_called_once()
