"""Corpus JSONL format: header + data record dataclasses with validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


CORPUS_TYPE = "smartmemory-corpus"
CORPUS_VERSION = "1.0"


@dataclass
class CorpusEntity:
    """Entity extracted from or annotated on a corpus record."""

    name: str
    type: str = ""
    qid: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.type:
            d["type"] = self.type
        if self.qid:
            d["qid"] = self.qid
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusEntity:
        return cls(name=data["name"], type=data.get("type", ""), qid=data.get("qid", ""))


@dataclass
class CorpusRelation:
    """Relation triple from a corpus record."""

    source: str
    relation: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "relation": self.relation, "target": self.target}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusRelation:
        return cls(source=data["source"], relation=data["relation"], target=data["target"])


@dataclass
class CorpusHeader:
    """First line of a corpus JSONL file — metadata about the corpus.

    Required fields: _type, _version, _source.
    """

    source: str
    domain: str = ""
    created: str = ""
    item_count: int = 0
    checksum: str = ""

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.source:
            errors.append("_source is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "_type": CORPUS_TYPE,
            "_version": CORPUS_VERSION,
            "_source": self.source,
        }
        if self.domain:
            d["_domain"] = self.domain
        if self.created:
            d["_created"] = self.created
        else:
            d["_created"] = datetime.now(timezone.utc).isoformat()
        if self.item_count:
            d["_item_count"] = self.item_count
        if self.checksum:
            d["_checksum"] = self.checksum
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusHeader:
        return cls(
            source=data.get("_source", ""),
            domain=data.get("_domain", ""),
            created=data.get("_created", ""),
            item_count=data.get("_item_count", 0),
            checksum=data.get("_checksum", ""),
        )

    @classmethod
    def parse(cls, line: str) -> CorpusHeader:
        """Parse a JSON line into a CorpusHeader. Raises ValueError on bad format."""
        data = json.loads(line)
        if data.get("_type") != CORPUS_TYPE:
            raise ValueError(f"Expected _type={CORPUS_TYPE!r}, got {data.get('_type')!r}")
        version = data.get("_version", "")
        if not version.startswith("1."):
            raise ValueError(f"Unsupported corpus version: {version!r}")
        header = cls.from_dict(data)
        errors = header.validate()
        if errors:
            raise ValueError(f"Invalid corpus header: {'; '.join(errors)}")
        return header


@dataclass
class CorpusRecord:
    """One data record in a corpus JSONL file (lines 2+).

    Only `content` is required. Entities/relations are optional pre-extracted
    annotations that can bypass LLM extraction in direct mode.
    """

    content: str
    memory_type: str = "semantic"
    entities: list[CorpusEntity] = field(default_factory=list)
    relations: list[CorpusRelation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.content or not self.content.strip():
            errors.append("content is required and must be non-empty")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"content": self.content}
        if self.memory_type != "semantic":
            d["memory_type"] = self.memory_type
        if self.entities:
            d["entities"] = [e.to_dict() for e in self.entities]
        if self.relations:
            d["relations"] = [r.to_dict() for r in self.relations]
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusRecord:
        entities = [CorpusEntity.from_dict(e) for e in data.get("entities", [])]
        relations = [CorpusRelation.from_dict(r) for r in data.get("relations", [])]
        return cls(
            content=data.get("content", ""),
            memory_type=data.get("memory_type", "semantic"),
            entities=entities,
            relations=relations,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def parse(cls, line: str) -> CorpusRecord:
        """Parse a JSON line into a CorpusRecord. Raises ValueError on bad format."""
        data = json.loads(line)
        # Skip header records that ended up in data stream
        if "_type" in data:
            raise ValueError("Got header record where data record expected")
        record = cls.from_dict(data)
        errors = record.validate()
        if errors:
            raise ValueError(f"Invalid corpus record: {'; '.join(errors)}")
        return record

    @property
    def has_extractions(self) -> bool:
        """Whether this record has pre-extracted entities or relations."""
        return bool(self.entities or self.relations)
