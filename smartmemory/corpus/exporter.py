"""Corpus export: dump SmartMemory contents to corpus JSONL."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from smartmemory.corpus.format import CorpusEntity, CorpusRecord, CorpusRelation
from smartmemory.corpus.writer import CorpusWriter

logger = logging.getLogger(__name__)


class CorpusExporter:
    """Exports memories from SmartMemory to corpus JSONL format.

    Iterates the graph, optionally including extracted entities/relations.
    """

    def __init__(
        self,
        smart_memory: Any,
        memory_type: str | None = None,
        include_entities: bool = False,
        limit: int = 0,
    ):
        self._sm = smart_memory
        self._memory_type = memory_type
        self._include_entities = include_entities
        self._limit = limit

    def run(
        self,
        output_path: str | Path,
        source: str = "smartmemory-export",
        domain: str = "",
        progress_callback: Any | None = None,
    ) -> int:
        """Export memories to a corpus JSONL file.

        Args:
            output_path: Where to write the .jsonl file.
            source: Source identifier for the corpus header.
            domain: Domain tag for the corpus header.
            progress_callback: Called with (count, record) after each record.

        Returns:
            Number of records written.
        """
        items = self._fetch_items()

        with CorpusWriter(output_path, source=source, domain=domain) as writer:
            for item in items:
                record = self._item_to_record(item)
                if record:
                    writer.write_record(record)
                    if progress_callback:
                        progress_callback(writer.count, record)

        logger.info("Exported %d records to %s", writer.count, output_path)
        return writer.count

    def _fetch_items(self) -> list[dict[str, Any]]:
        """Fetch memory items from SmartMemory."""
        # Use search with a broad query, or iterate the graph directly
        if hasattr(self._sm, "graph"):
            return self._fetch_from_graph()
        return []

    def _fetch_from_graph(self) -> list[dict[str, Any]]:
        """Fetch items via SmartGraph Cypher query."""
        graph = self._sm.graph
        type_filter = ""
        if self._memory_type:
            type_filter = f" AND n.memory_type = '{self._memory_type}'"
        limit_clause = f" LIMIT {self._limit}" if self._limit else ""

        query = f"MATCH (n:Memory) WHERE n.content IS NOT NULL{type_filter} RETURN n{limit_clause}"
        try:
            result = graph.query(query)
            items = []
            for row in result.result_set:
                node = row[0]
                items.append(
                    {
                        "item_id": node.properties.get("item_id", ""),
                        "content": node.properties.get("content", ""),
                        "memory_type": node.properties.get("memory_type", "semantic"),
                        "metadata": {},
                    }
                )
            return items
        except Exception as e:
            logger.error("Failed to fetch items from graph: %s", e)
            return []

    def _item_to_record(self, item: dict[str, Any]) -> CorpusRecord | None:
        """Convert a memory item dict to a CorpusRecord."""
        content = item.get("content", "")
        if not content or not content.strip():
            return None

        entities: list[CorpusEntity] = []
        relations: list[CorpusRelation] = []

        if self._include_entities and hasattr(self._sm, "graph"):
            entities, relations = self._fetch_extractions(item.get("item_id", ""))

        return CorpusRecord(
            content=content,
            memory_type=item.get("memory_type", "semantic"),
            entities=entities,
            relations=relations,
            metadata=item.get("metadata", {}),
        )

    def _fetch_extractions(self, item_id: str) -> tuple[list[CorpusEntity], list[CorpusRelation]]:
        """Fetch entities and relations linked to a memory item."""
        if not item_id:
            return [], []

        entities: list[CorpusEntity] = []
        relations: list[CorpusRelation] = []
        graph = self._sm.graph

        try:
            # Fetch entities
            eq = (
                f"MATCH (m:Memory {{item_id: '{item_id}'}})-[:HAS_ENTITY]->(e:Entity) "
                f"RETURN e.name, e.entity_type, e.qid"
            )
            result = graph.query(eq)
            for row in result.result_set:
                entities.append(
                    CorpusEntity(
                        name=row[0] or "",
                        type=row[1] or "",
                        qid=row[2] or "",
                    )
                )

            # Fetch relations
            rq = (
                f"MATCH (m:Memory {{item_id: '{item_id}'}})-[:HAS_ENTITY]->(s:Entity)"
                f"-[r]->(t:Entity) "
                f"WHERE type(r) <> 'HAS_ENTITY' "
                f"RETURN s.name, type(r), t.name"
            )
            result = graph.query(rq)
            for row in result.result_set:
                relations.append(
                    CorpusRelation(
                        source=row[0] or "",
                        relation=row[1] or "",
                        target=row[2] or "",
                    )
                )
        except Exception as e:
            logger.warning("Failed to fetch extractions for %s: %s", item_id, e)

        return entities, relations
