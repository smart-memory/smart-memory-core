"""REBEL dataset converter: HuggingFace → corpus JSONL.

REBEL contains 3.47M Wikipedia sentences with entity-relation triples.
This converter loads via HuggingFace datasets (streaming mode), parses
linearized triplets, and outputs SmartMemory corpus JSONL.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from smartmemory.corpus.format import CorpusEntity, CorpusRecord, CorpusRelation
from smartmemory.corpus.triplet_parser import parse_rebel_triplets
from smartmemory.corpus.writer import CorpusWriter

logger = logging.getLogger(__name__)

# Domain keyword filters for targeted extraction
DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "tech": {
        "programming",
        "software",
        "computer",
        "database",
        "framework",
        "API",
        "algorithm",
        "protocol",
        "server",
        "web",
        "application",
        "operating system",
        "compiler",
        "library",
        "runtime",
        "processor",
        "CPU",
        "GPU",
        "network",
        "internet",
        "cloud",
        "artificial intelligence",
        "machine learning",
        "data structure",
        "encryption",
        "open source",
        "Linux",
        "Windows",
    },
    "science": {
        "physics",
        "chemistry",
        "biology",
        "research",
        "experiment",
        "theory",
        "molecule",
        "atom",
        "cell",
        "gene",
        "species",
        "evolution",
        "quantum",
        "relativity",
        "thermodynamics",
        "ecology",
        "neuroscience",
        "astronomy",
        "mathematics",
        "statistics",
        "hypothesis",
        "laboratory",
    },
}

# Rough type inference from entity names matching Wikipedia conventions
ENTITY_TYPE_HINTS: dict[str, str] = {
    "language": "Language",
    "programming language": "Language",
    "software": "Technology",
    "framework": "Technology",
    "company": "Organization",
    "university": "Organization",
    "city": "Location",
    "country": "Location",
    "person": "Person",
}


class REBELConverter:
    """Converts REBEL dataset samples to SmartMemory corpus JSONL.

    Loads from HuggingFace in streaming mode to avoid downloading
    the full 3.47M-record dataset.
    """

    def __init__(
        self,
        domain: str | None = None,
        limit: int = 0,
        split: str = "train",
    ):
        self._domain = domain
        self._limit = limit
        self._split = split

    def load_samples(self) -> Iterator[dict[str, Any]]:
        """Load REBEL samples from HuggingFace in streaming mode.

        Yields raw dataset rows with 'title', 'context', 'triplets' fields.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets package required for REBEL conversion. Install with: pip install datasets"
            ) from e

        dataset = load_dataset(
            "Babelscape/rebel-dataset",
            split=self._split,
            streaming=True,
        )

        count = 0
        keywords = DOMAIN_KEYWORDS.get(self._domain, set()) if self._domain else set()

        for sample in dataset:
            if self._limit and count >= self._limit:
                break

            if keywords:
                text = f"{sample.get('title', '')} {sample.get('context', '')}".lower()
                if not any(kw.lower() in text for kw in keywords):
                    continue

            yield sample
            count += 1

    def convert_sample(self, sample: dict[str, Any]) -> CorpusRecord | None:
        """Convert a single REBEL sample to a CorpusRecord."""
        context = sample.get("context", "")
        if not context or not context.strip():
            return None

        triplets_text = sample.get("triplets", "")
        parsed = parse_rebel_triplets(triplets_text)

        entities: list[CorpusEntity] = []
        relations: list[CorpusRelation] = []
        seen_entities: set[str] = set()

        for triple in parsed:
            source = triple["source"]
            target = triple["target"]
            relation = triple["relation"]

            if source and source not in seen_entities:
                entities.append(CorpusEntity(name=source, type=_infer_type(source, relation)))
                seen_entities.add(source)

            if target and target not in seen_entities:
                entities.append(CorpusEntity(name=target, type=_infer_type(target, relation)))
                seen_entities.add(target)

            if source and target and relation:
                relations.append(
                    CorpusRelation(
                        source=source,
                        relation=relation.upper().replace(" ", "_"),
                        target=target,
                    )
                )

        return CorpusRecord(
            content=context,
            entities=entities,
            relations=relations,
            metadata={
                "source_corpus": "rebel",
                "domain": self._domain or "general",
                "title": sample.get("title", ""),
            },
        )

    def convert_to_file(
        self,
        output_path: str,
        progress_callback: Any | None = None,
    ) -> int:
        """Convert REBEL samples and write to corpus JSONL.

        Returns:
            Number of records written.
        """
        domain_tag = self._domain or "general"
        with CorpusWriter(output_path, source=f"rebel-{domain_tag}", domain=domain_tag) as writer:
            for sample in self.load_samples():
                record = self.convert_sample(sample)
                if record:
                    writer.write_record(record)
                    if progress_callback:
                        progress_callback(writer.count, record)

        logger.info("Converted %d REBEL samples to %s", writer.count, output_path)
        return writer.count


def _infer_type(entity_name: str, relation: str) -> str:
    """Best-effort entity type inference from REBEL context."""
    name_lower = entity_name.lower()
    rel_lower = relation.lower()

    # Relation-based hints
    if "country" in rel_lower or "nationality" in rel_lower:
        return "Location"
    if "language" in rel_lower:
        return "Language"
    if "author" in rel_lower or "creator" in rel_lower or "founder" in rel_lower:
        return "Person"

    # Name-based hints
    for hint_key, hint_type in ENTITY_TYPE_HINTS.items():
        if hint_key in name_lower:
            return hint_type

    return ""
