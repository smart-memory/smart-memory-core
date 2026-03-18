"""Wikidata mining pipeline: domain sweep, incremental updates, and quota-aware scheduling.

Outputs PublicEntity JSONL for grounding and/or corpus JSONL for import.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from smartmemory.corpus.format import CorpusEntity, CorpusRecord
from smartmemory.corpus.writer import CorpusWriter
from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.sparql_client import WDQSClient
from smartmemory.grounding.type_map import WIKIDATA_TYPE_MAP

logger = logging.getLogger(__name__)


# Expanded domain list (Phase 2 — from 6 to ~15 domains)
EXPANDED_DOMAINS: dict[str, str] = {
    # Existing 6
    "Q9143": "programming_languages",
    "Q271680": "software_frameworks",
    "Q188860": "software_libraries",
    "Q9135": "operating_systems",
    "Q4830453": "companies",
    "Q3918": "universities",
    # New tech
    "Q1639024": "databases",
    "Q131093": "web_frameworks",
    "Q15401930": "protocols",
    "Q1301371": "cloud_platforms",
    # New general knowledge
    "Q515": "cities",
    "Q6256": "countries",
    "Q11424": "films",
    "Q7889": "video_games",
}


@dataclass
class MiningCheckpoint:
    """Tracks mining progress across domains for resume."""

    completed_domains: list[str] = field(default_factory=list)
    partial_domain: str = ""
    partial_offset: int = 0
    total_queries: int = 0
    total_entities: int = 0

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "completed_domains": self.completed_domains,
                    "partial_domain": self.partial_domain,
                    "partial_offset": self.partial_offset,
                    "total_queries": self.total_queries,
                    "total_entities": self.total_entities,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> MiningCheckpoint:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            return cls()


@dataclass
class MiningResult:
    """Result of a mining run."""

    entities: list[PublicEntity] = field(default_factory=list)
    domain_counts: dict[str, int] = field(default_factory=dict)
    total_queries: int = 0
    quota_exhausted: bool = False


class WikidataMiner:
    """Mines Wikidata for entities via SPARQL, with quota awareness and checkpointing.

    Supports three strategies:
    - Domain sweep: query expanded default domains
    - Custom domains: user-provided domain config file
    - Incremental: only entities modified since last harvest
    """

    def __init__(
        self,
        client: WDQSClient | None = None,
        quota_limit: int = 0,
        limit_per_domain: int = 5000,
    ):
        self._client = client or WDQSClient()
        self._quota_limit = quota_limit
        self._limit_per_domain = limit_per_domain
        self._query_count = 0

    def mine_domains(
        self,
        domains: dict[str, str],
        output_dir: str | Path,
        output_format: str = "both",
        incremental_since: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> MiningResult:
        """Mine one or more Wikidata domains.

        Args:
            domains: Dict of {domain_qid: domain_name}.
            output_dir: Directory for output files.
            output_format: "corpus", "snapshot", or "both".
            incremental_since: ISO date string for incremental mode.
            checkpoint_path: Path for mining checkpoint file.

        Returns:
            MiningResult with entities and stats.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = MiningCheckpoint()
        if checkpoint_path:
            checkpoint = MiningCheckpoint.load(checkpoint_path)

        result = MiningResult(total_queries=checkpoint.total_queries)
        all_entities: list[PublicEntity] = []

        for domain_qid, domain_name in domains.items():
            if domain_name in checkpoint.completed_domains:
                logger.info("Skipping completed domain: %s", domain_name)
                continue

            if self._quota_limit and self._query_count >= self._quota_limit:
                logger.warning("Quota limit reached (%d queries)", self._query_count)
                result.quota_exhausted = True
                if checkpoint_path:
                    checkpoint.save(checkpoint_path)
                break

            logger.info("Mining domain: %s (%s)", domain_name, domain_qid)
            try:
                if incremental_since:
                    entities = self._query_incremental(domain_qid, incremental_since)
                else:
                    entities = self._client.query_domain(domain_qid, limit=self._limit_per_domain)
                self._query_count += 1
                result.total_queries += 1

                for entity in entities:
                    entity.domain = domain_name
                    if not entity.entity_type:
                        for iof in entity.instance_of:
                            mapped = WIKIDATA_TYPE_MAP.get(iof)
                            if mapped:
                                entity.entity_type = mapped
                                break

                all_entities.extend(entities)
                result.domain_counts[domain_name] = len(entities)
                logger.info("  → %d entities", len(entities))

                checkpoint.completed_domains.append(domain_name)
                checkpoint.total_queries = result.total_queries
                checkpoint.total_entities = len(all_entities)
                if checkpoint_path:
                    checkpoint.save(checkpoint_path)

            except Exception as e:
                logger.error("Failed to mine domain %s: %s", domain_name, e)
                result.domain_counts[domain_name] = 0
                # Exponential backoff handled by WDQSClient

        # Deduplicate by QID
        seen: set[str] = set()
        for entity in all_entities:
            if entity.qid not in seen:
                seen.add(entity.qid)
                result.entities.append(entity)

        # Write outputs
        if output_format in ("snapshot", "both"):
            self._write_snapshot(result.entities, output_path)

        if output_format in ("corpus", "both"):
            self._write_corpus(result.entities, output_path)

        logger.info(
            "Mining complete: %d unique entities from %d domains (%d queries)",
            len(result.entities),
            len(result.domain_counts),
            result.total_queries,
        )
        return result

    def _query_incremental(self, domain_qid: str, since: str) -> list[PublicEntity]:
        """Query entities modified after a given date."""
        import requests

        sparql = f"""
        SELECT ?item ?itemLabel ?itemDescription ?instanceOf ?altLabel WHERE {{
            ?item wdt:P31/wdt:P279* wd:{domain_qid} .
            ?item schema:dateModified ?mod .
            FILTER(?mod > "{since}"^^xsd:dateTime)
            OPTIONAL {{ ?item wdt:P31 ?instanceOf . }}
            OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) = "en") }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT {self._limit_per_domain}
        """

        self._client._rate_limit()
        response = requests.get(
            self._client.ENDPOINT,
            params={"query": sparql, "format": "json"},
            timeout=30.0,
            headers={"User-Agent": "SmartMemory/1.0 (incremental-mine)"},
        )
        response.raise_for_status()
        return self._client._parse_results(response.json())

    def _write_snapshot(self, entities: list[PublicEntity], output_dir: Path) -> None:
        """Write PublicEntity JSONL for grounding store."""
        path = output_dir / "mined_entities.jsonl"
        with open(path, "w") as f:
            for entity in entities:
                f.write(
                    json.dumps(
                        {
                            "qid": entity.qid,
                            "label": entity.label,
                            "aliases": entity.aliases,
                            "description": entity.description,
                            "entity_type": entity.entity_type,
                            "instance_of": entity.instance_of,
                            "domain": entity.domain,
                            "confidence": entity.confidence,
                        }
                    )
                    + "\n"
                )
        logger.info("Wrote %d entities to %s", len(entities), path)

    def _write_corpus(self, entities: list[PublicEntity], output_dir: Path) -> None:
        """Write corpus JSONL (Phase 1 format) from mined entities."""
        path = output_dir / "mined_corpus.jsonl"
        with CorpusWriter(path, source="wikidata-mine", domain="general") as writer:
            for entity in entities:
                content = entity.label
                if entity.description:
                    content = f"{entity.label}: {entity.description}"

                record = CorpusRecord(
                    content=content,
                    memory_type="semantic",
                    entities=[
                        CorpusEntity(
                            name=entity.label,
                            type=entity.entity_type,
                            qid=entity.qid,
                        )
                    ],
                    metadata={
                        "source_corpus": "wikidata",
                        "domain": entity.domain,
                        "qid": entity.qid,
                    },
                )
                writer.write_record(record)
        logger.info("Wrote corpus to %s", path)


def load_domain_config(path: str | Path) -> dict[str, str]:
    """Load a domain config JSON file.

    Format: {"domain_name": {"qid": "Q123", "limit": 2000, "type": "Technology"}, ...}
    Returns {qid: name} dict suitable for WikidataMiner.mine_domains().
    """
    data = json.loads(Path(path).read_text())
    domains: dict[str, str] = {}
    for name, config in data.items():
        qid = config.get("qid", "")
        if qid:
            domains[qid] = name
    return domains
