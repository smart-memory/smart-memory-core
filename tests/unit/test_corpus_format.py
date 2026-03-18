"""Unit tests for corpus JSONL format: header, record, reader, writer."""

import json
from pathlib import Path

import pytest

from smartmemory.corpus.format import (
    CORPUS_TYPE,
    CORPUS_VERSION,
    CorpusEntity,
    CorpusHeader,
    CorpusRecord,
    CorpusRelation,
)
from smartmemory.corpus.reader import CheckpointState, CorpusReader
from smartmemory.corpus.writer import CorpusWriter


# ── CorpusHeader ─────────────────────────────────────────────────────────────


class TestCorpusHeader:
    def test_round_trip(self):
        header = CorpusHeader(source="test-source", domain="tech", item_count=42)
        d = header.to_dict()
        assert d["_type"] == CORPUS_TYPE
        assert d["_version"] == CORPUS_VERSION
        assert d["_source"] == "test-source"
        assert d["_domain"] == "tech"
        assert d["_item_count"] == 42

        parsed = CorpusHeader.parse(header.to_json())
        assert parsed.source == "test-source"
        assert parsed.domain == "tech"
        assert parsed.item_count == 42

    def test_validate_missing_source(self):
        header = CorpusHeader(source="")
        errors = header.validate()
        assert "_source is required" in errors

    def test_parse_wrong_type(self):
        with pytest.raises(ValueError, match="Expected _type"):
            CorpusHeader.parse(json.dumps({"_type": "wrong", "_version": "1.0", "_source": "x"}))

    def test_parse_wrong_version(self):
        with pytest.raises(ValueError, match="Unsupported corpus version"):
            CorpusHeader.parse(json.dumps({"_type": CORPUS_TYPE, "_version": "2.0", "_source": "x"}))

    def test_parse_valid_minor_version(self):
        """1.x versions should be forward-compatible."""
        header = CorpusHeader.parse(
            json.dumps({"_type": CORPUS_TYPE, "_version": "1.1", "_source": "x"})
        )
        assert header.source == "x"

    def test_created_auto_populated(self):
        header = CorpusHeader(source="test")
        d = header.to_dict()
        assert "_created" in d


# ── CorpusRecord ─────────────────────────────────────────────────────────────


class TestCorpusRecord:
    def test_round_trip_minimal(self):
        record = CorpusRecord(content="Hello world")
        d = record.to_dict()
        assert d == {"content": "Hello world"}

        parsed = CorpusRecord.parse(record.to_json())
        assert parsed.content == "Hello world"
        assert parsed.memory_type == "semantic"

    def test_round_trip_full(self):
        record = CorpusRecord(
            content="React 19 introduces use() hook",
            memory_type="episodic",
            entities=[
                CorpusEntity(name="React", type="Technology", qid="Q19399674"),
                CorpusEntity(name="use() hook", type="Concept"),
            ],
            relations=[
                CorpusRelation(source="React", relation="INTRODUCES", target="use() hook"),
            ],
            metadata={"domain": "web-development", "confidence": 0.95},
        )
        j = record.to_json()
        parsed = CorpusRecord.parse(j)
        assert parsed.memory_type == "episodic"
        assert len(parsed.entities) == 2
        assert parsed.entities[0].qid == "Q19399674"
        assert len(parsed.relations) == 1
        assert parsed.relations[0].relation == "INTRODUCES"
        assert parsed.metadata["confidence"] == 0.95

    def test_validate_empty_content(self):
        record = CorpusRecord(content="")
        errors = record.validate()
        assert errors

    def test_validate_whitespace_content(self):
        record = CorpusRecord(content="   ")
        errors = record.validate()
        assert errors

    def test_parse_rejects_header(self):
        with pytest.raises(ValueError, match="Got header record"):
            CorpusRecord.parse(json.dumps({"_type": CORPUS_TYPE, "content": "x"}))

    def test_has_extractions_empty(self):
        record = CorpusRecord(content="x")
        assert not record.has_extractions

    def test_has_extractions_with_entities(self):
        record = CorpusRecord(content="x", entities=[CorpusEntity(name="Foo")])
        assert record.has_extractions

    def test_memory_type_default_not_in_dict(self):
        """Default 'semantic' type omitted from dict for compactness."""
        record = CorpusRecord(content="x")
        assert "memory_type" not in record.to_dict()

    def test_non_default_memory_type_in_dict(self):
        record = CorpusRecord(content="x", memory_type="episodic")
        assert record.to_dict()["memory_type"] == "episodic"


# ── CorpusReader ─────────────────────────────────────────────────────────────


class TestCorpusReader:
    def _write_corpus(self, path: Path, records: list[str], header: dict | None = None) -> Path:
        corpus = path / "test.jsonl"
        h = header or {"_type": CORPUS_TYPE, "_version": CORPUS_VERSION, "_source": "test"}
        lines = [json.dumps(h)] + records
        corpus.write_text("\n".join(lines) + "\n")
        return corpus

    def test_read_header(self, tmp_path):
        corpus = self._write_corpus(tmp_path, [])
        reader = CorpusReader(corpus)
        header = reader.read_header()
        assert header.source == "test"

    def test_iter_records(self, tmp_path):
        records = [
            json.dumps({"content": "first"}),
            json.dumps({"content": "second"}),
            json.dumps({"content": "third"}),
        ]
        corpus = self._write_corpus(tmp_path, records)
        reader = CorpusReader(corpus)
        reader.read_header()
        result = list(reader.iter_records())
        assert len(result) == 3
        assert result[0][1].content == "first"
        assert result[2][1].content == "third"

    def test_skip_lines(self, tmp_path):
        records = [json.dumps({"content": f"item-{i}"}) for i in range(5)]
        corpus = self._write_corpus(tmp_path, records)
        reader = CorpusReader(corpus)
        reader.read_header()
        result = list(reader.iter_records(skip_lines=3))
        assert len(result) == 2
        assert result[0][1].content == "item-3"

    def test_domain_filter(self, tmp_path):
        records = [
            json.dumps({"content": "a", "metadata": {"domain": "tech"}}),
            json.dumps({"content": "b", "metadata": {"domain": "science"}}),
            json.dumps({"content": "c", "metadata": {"domain": "web-tech"}}),
        ]
        corpus = self._write_corpus(tmp_path, records)
        reader = CorpusReader(corpus)
        reader.read_header()
        result = list(reader.iter_records(domain_filter="tech"))
        assert len(result) == 2
        contents = [r[1].content for r in result]
        assert "a" in contents
        assert "c" in contents

    def test_corrupt_records_skipped(self, tmp_path):
        records = [
            json.dumps({"content": "good"}),
            "not valid json {{{",
            json.dumps({"content": "also good"}),
        ]
        corpus = self._write_corpus(tmp_path, records)
        reader = CorpusReader(corpus)
        reader.read_header()
        result = list(reader.iter_records())
        assert len(result) == 2

    def test_count_records(self, tmp_path):
        records = [json.dumps({"content": f"item-{i}"}) for i in range(10)]
        corpus = self._write_corpus(tmp_path, records)
        reader = CorpusReader(corpus)
        assert reader.count_records() == 10

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CorpusReader("/nonexistent/path.jsonl")

    def test_empty_file(self, tmp_path):
        corpus = tmp_path / "empty.jsonl"
        corpus.write_text("")
        reader = CorpusReader(corpus)
        with pytest.raises(ValueError, match="Empty corpus file"):
            reader.read_header()


# ── CorpusWriter ─────────────────────────────────────────────────────────────


class TestCorpusWriter:
    def test_write_and_verify(self, tmp_path):
        out = tmp_path / "output.jsonl"
        with CorpusWriter(out, source="test-export", domain="tech") as writer:
            writer.write_record(CorpusRecord(content="first"))
            writer.write_record(CorpusRecord(content="second"))
            writer.write_record(CorpusRecord(content="third"))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 records

        header = json.loads(lines[0])
        assert header["_type"] == CORPUS_TYPE
        assert header["_source"] == "test-export"
        assert header["_item_count"] == 3
        assert header["_checksum"].startswith("sha256:")

    def test_round_trip_reader_writer(self, tmp_path):
        out = tmp_path / "roundtrip.jsonl"
        records = [
            CorpusRecord(content="Hello", memory_type="episodic"),
            CorpusRecord(
                content="React is a framework",
                entities=[CorpusEntity(name="React", type="Technology")],
                relations=[CorpusRelation(source="React", relation="IS_A", target="framework")],
            ),
        ]
        with CorpusWriter(out, source="roundtrip-test") as writer:
            for r in records:
                writer.write_record(r)

        reader = CorpusReader(out)
        header = reader.read_header()
        assert header.source == "roundtrip-test"
        assert header.item_count == 2

        parsed = list(reader.iter_records())
        assert len(parsed) == 2
        assert parsed[0][1].content == "Hello"
        assert parsed[0][1].memory_type == "episodic"
        assert parsed[1][1].entities[0].name == "React"
        assert parsed[1][1].relations[0].relation == "IS_A"

    def test_count_property(self, tmp_path):
        out = tmp_path / "count.jsonl"
        with CorpusWriter(out, source="test") as writer:
            assert writer.count == 0
            writer.write_record(CorpusRecord(content="one"))
            assert writer.count == 1

    def test_write_without_open_raises(self):
        writer = CorpusWriter("/tmp/nope.jsonl", source="test")
        with pytest.raises(RuntimeError, match="Writer not open"):
            writer.write_record(CorpusRecord(content="x"))


# ── CheckpointState ──────────────────────────────────────────────────────────


class TestCheckpointState:
    def test_save_and_load(self, tmp_path):
        cp_path = tmp_path / "test.checkpoint.json"
        cp = CheckpointState(cp_path)
        cp.last_line = 42
        cp.imported = 40
        cp.errors = 2
        cp.save()

        cp2 = CheckpointState(cp_path)
        assert cp2.load()
        assert cp2.last_line == 42
        assert cp2.imported == 40
        assert cp2.errors == 2

    def test_load_missing(self, tmp_path):
        cp = CheckpointState(tmp_path / "missing.checkpoint.json")
        assert not cp.load()

    def test_mark_complete(self, tmp_path):
        cp_path = tmp_path / "test.checkpoint.json"
        cp = CheckpointState(cp_path)
        cp.save()
        assert cp_path.exists()

        cp.mark_complete()
        assert not cp_path.exists()
        assert (tmp_path / "test.complete.json").exists()
