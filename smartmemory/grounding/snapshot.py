"""Snapshot pipeline — build-time CLI tool for Wikidata entity snapshot.

Queries WDQS for entities in specified domains, normalizes types via
WIKIDATA_TYPE_MAP, and exports to JSONL + pre-built SQLite database.

Usage:
    python -m smartmemory.grounding.snapshot --output smartmemory/data/public_knowledge/
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from smartmemory.grounding.models import PublicEntity
from smartmemory.grounding.sparql_client import WDQSClient
from smartmemory.grounding.sqlite_store import SQLitePublicKnowledgeStore
from smartmemory.grounding.type_map import WIKIDATA_TYPE_MAP

logger = logging.getLogger(__name__)

# Default domains to query — Wikidata superclass QIDs
DEFAULT_DOMAINS = {
    "Q9143": "programming_languages",  # programming language
    "Q271680": "software_frameworks",  # software framework
    "Q188860": "software_libraries",  # software library
    "Q9135": "operating_systems",  # operating system
    "Q4830453": "companies",  # business enterprise
    "Q3918": "universities",  # university
}


def build_snapshot(
    output_dir: str,
    domains: dict[str, str] | None = None,
    limit_per_domain: int = 5000,
) -> dict:
    """Build a public knowledge snapshot from Wikidata SPARQL.

    Args:
        output_dir: Directory to write JSONL and SQLite files.
        domains: Dict of {domain_qid: domain_name}. Defaults to DEFAULT_DOMAINS.
        limit_per_domain: Max entities per domain query.

    Returns:
        Metadata dict with version, counts, and file paths.
    """
    domains = domains or DEFAULT_DOMAINS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = WDQSClient()
    all_entities: list[PublicEntity] = []
    domain_counts: dict[str, int] = {}

    for domain_qid, domain_name in domains.items():
        logger.info("Querying domain %s (%s)...", domain_name, domain_qid)
        try:
            entities = client.query_domain(domain_qid, limit=limit_per_domain)
            # Apply domain label and ensure type mapping
            for entity in entities:
                entity.domain = domain_name
                if not entity.entity_type:
                    for iof in entity.instance_of:
                        mapped = WIKIDATA_TYPE_MAP.get(iof)
                        if mapped:
                            entity.entity_type = mapped
                            break
            all_entities.extend(entities)
            domain_counts[domain_name] = len(entities)
            logger.info("  → %d entities", len(entities))
        except Exception as e:
            logger.error("Failed to query domain %s: %s", domain_name, e)
            domain_counts[domain_name] = 0

    # Deduplicate by QID (entities may appear in multiple domains)
    seen_qids: set[str] = set()
    unique_entities: list[PublicEntity] = []
    for entity in all_entities:
        if entity.qid not in seen_qids:
            seen_qids.add(entity.qid)
            unique_entities.append(entity)

    # Export JSONL
    jsonl_path = output_path / "public_entities.jsonl"
    with open(jsonl_path, "w") as f:
        for entity in unique_entities:
            f.write(json.dumps({
                "qid": entity.qid,
                "label": entity.label,
                "aliases": entity.aliases,
                "description": entity.description,
                "entity_type": entity.entity_type,
                "instance_of": entity.instance_of,
                "domain": entity.domain,
                "confidence": entity.confidence,
            }) + "\n")

    # Build pre-built SQLite
    sqlite_path = output_path / "public_entities.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    store = SQLitePublicKnowledgeStore(str(sqlite_path))
    store.load_snapshot(str(jsonl_path))
    store.close()

    # Write metadata
    meta = {
        "version": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_entities": len(unique_entities),
        "domain_counts": domain_counts,
        "files": {
            "jsonl": str(jsonl_path),
            "sqlite": str(sqlite_path),
        },
    }
    meta_path = output_path / "snapshot_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Snapshot complete: %d unique entities across %d domains",
        len(unique_entities),
        len(domains),
    )
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build Wikidata public knowledge snapshot")
    parser.add_argument(
        "--output", "-o",
        default="smartmemory/data/public_knowledge/",
        help="Output directory for snapshot files",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5000,
        help="Max entities per domain query",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    meta = build_snapshot(args.output, limit_per_domain=args.limit)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
