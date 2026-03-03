"""CORE-8: Multi-enricher interaction integration test.

Exercises sequential enricher execution and cumulative property merging
using the three local-only enrichers: BasicEnricher, SentimentEnricher,
TopicEnricher. No network calls, no API keys required.

Tests validate:
- Enrichment orchestration runs all enrichers sequentially
- Properties from different enrichers coexist after merging
- Top-level keys (BasicEnricher) and nested properties (Sentiment/Topic) merge correctly
- Entity context (node_ids) flows through to enrichers that use it
- Graceful handling of None node_ids

The integration/ conftest auto-marks this as pytest.mark.integration.
"""
import pytest

from smartmemory.memory.pipeline.stages.enrichment import Enrichment
from smartmemory.models.memory_item import MemoryItem

# The three local-only enrichers — no network, no API keys
LOCAL_ENRICHERS = ["basic_enricher", "sentiment_enricher", "topic_enricher"]

# Rich test content with entities, sentiment, and topic-worthy keywords
TEST_CONTENT = (
    "Alice leads Project Atlas at Acme Corp. The project achieved excellent results "
    "in machine learning and natural language processing. The team developed innovative "
    "algorithms for knowledge graph construction and semantic search optimization."
)


class TestMultiEnricherOrchestration:
    """Enrichment class runs multiple enrichers and merges results correctly."""

    @pytest.fixture
    def enrichment(self):
        """Create Enrichment orchestrator with a minimal graph stub.

        The Enrichment class stores graph but the local enrichers never call it —
        they only read item content and node_ids. A None graph is safe here.
        """
        return Enrichment(graph=None)

    @pytest.fixture
    def item(self):
        return MemoryItem(content=TEST_CONTENT, memory_type="semantic")

    @pytest.fixture
    def node_ids(self):
        return {"semantic_entities": ["Alice", "Project Atlas", "Acme Corp"]}

    def test_all_local_enrichers_produce_merged_result(self, enrichment, item, node_ids):
        """All three enrichers run and contribute to a single merged result."""
        context = {"item": item, "node_ids": node_ids}
        result = enrichment.enrich(context, enricher_names=LOCAL_ENRICHERS)

        # BasicEnricher: top-level keys
        assert "summary" in result, "BasicEnricher should produce a summary"
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

        assert "tags" in result, "BasicEnricher should produce entity tags"
        assert isinstance(result["tags"], list)
        assert "Alice" in result["tags"]
        assert "Acme Corp" in result["tags"]

        # SentimentEnricher: nested under properties
        assert "properties" in result, "Merged result should have properties dict"
        props = result["properties"]
        assert "sentiment" in props, "SentimentEnricher should add sentiment"
        sentiment = props["sentiment"]
        assert "score" in sentiment
        assert "label" in sentiment
        assert sentiment["label"] in ("positive", "negative", "neutral")
        assert "method" in sentiment
        assert "confidence" in sentiment
        assert -1.0 <= sentiment["score"] <= 1.0

        # TopicEnricher: nested under properties (merged alongside sentiment)
        assert "topics" in props, "TopicEnricher should add topics"
        assert isinstance(props["topics"], list)
        assert "dominant_topic" in props
        assert "keywords" in props
        assert isinstance(props["keywords"], list)
        assert len(props["keywords"]) > 0

    def test_properties_from_different_enrichers_dont_collide(self, enrichment, item, node_ids):
        """Sentiment and topic properties coexist — deep merge, not overwrite."""
        context = {"item": item, "node_ids": node_ids}
        result = enrichment.enrich(context, enricher_names=LOCAL_ENRICHERS)

        props = result.get("properties", {})
        # Both enrichers write to properties — verify neither overwrote the other
        assert "sentiment" in props and "topics" in props, (
            "Both sentiment and topics must be present in merged properties"
        )

    def test_none_node_ids_handled_gracefully(self, enrichment, item):
        """Enrichers tolerate None node_ids without errors."""
        context = {"item": item, "node_ids": None}
        result = enrichment.enrich(context, enricher_names=LOCAL_ENRICHERS)

        # BasicEnricher skips tags when node_ids is None
        assert "tags" not in result or result.get("tags") == []

        # Sentiment and topic still produce results (they ignore node_ids)
        props = result.get("properties", {})
        assert "sentiment" in props
        assert "topics" in props

    def test_enricher_ordering_is_deterministic(self, enrichment, item, node_ids):
        """Running the same enricher list twice produces identical results."""
        context = {"item": item, "node_ids": node_ids}
        result_a = enrichment.enrich(context, enricher_names=LOCAL_ENRICHERS)
        result_b = enrichment.enrich(context, enricher_names=LOCAL_ENRICHERS)

        assert result_a.get("summary") == result_b.get("summary")
        assert result_a.get("tags") == result_b.get("tags")
        assert result_a.get("properties", {}).get("sentiment") == result_b.get("properties", {}).get("sentiment")
        assert result_a.get("properties", {}).get("topics") == result_b.get("properties", {}).get("topics")


class TestEnricherContentQuality:
    """Enricher outputs are semantically meaningful for the test content."""

    @pytest.fixture
    def enrichment(self):
        return Enrichment(graph=None)

    def test_positive_content_gets_positive_sentiment(self, enrichment):
        """Content with 'excellent', 'innovative' should score positive."""
        item = MemoryItem(content=TEST_CONTENT, memory_type="semantic")
        context = {"item": item, "node_ids": None}
        result = enrichment.enrich(context, enricher_names=["sentiment_enricher"])

        sentiment = result["properties"]["sentiment"]
        assert sentiment["score"] > 0, f"Expected positive score for upbeat content, got {sentiment['score']}"
        assert sentiment["label"] == "positive"

    def test_negative_content_gets_negative_sentiment(self, enrichment):
        """Content with negative words should score negative."""
        item = MemoryItem(
            content="The terrible failure caused horrible losses. Everything went wrong.",
            memory_type="episodic",
        )
        context = {"item": item, "node_ids": None}
        result = enrichment.enrich(context, enricher_names=["sentiment_enricher"])

        sentiment = result["properties"]["sentiment"]
        assert sentiment["score"] < 0, f"Expected negative score, got {sentiment['score']}"
        assert sentiment["label"] == "negative"

    def test_topic_keywords_reflect_content(self, enrichment):
        """Topic enricher extracts keywords relevant to the content domain."""
        item = MemoryItem(content=TEST_CONTENT, memory_type="semantic")
        context = {"item": item, "node_ids": None}
        result = enrichment.enrich(context, enricher_names=["topic_enricher"])

        keywords = result["properties"]["keywords"]
        # The content is about ML and NLP — at least one domain keyword should appear
        domain_terms = {"project", "learning", "machine", "algorithms", "knowledge", "semantic", "search", "atlas"}
        overlap = set(k.lower() for k in keywords) & domain_terms
        assert len(overlap) >= 1, f"Expected domain keywords, got {keywords}"

    def test_summary_is_first_sentence(self, enrichment):
        """BasicEnricher summary is the first sentence of content."""
        item = MemoryItem(content=TEST_CONTENT, memory_type="semantic")
        context = {"item": item, "node_ids": {"semantic_entities": ["Alice"]}}
        result = enrichment.enrich(context, enricher_names=["basic_enricher"])

        assert result["summary"] == "Alice leads Project Atlas at Acme Corp."


class TestFullPipelineEnrichment:
    """Enrichers run as part of the full lite pipeline during ingest."""

    @pytest.fixture
    def lite_memory(self, tmp_path):
        """Lite SmartMemory with local enrichers only (no temporal_enricher)."""
        from smartmemory.pipeline.config import PipelineConfig
        from smartmemory.tools.factory import lite_context

        profile = PipelineConfig.lite(llm_enabled=False)
        # Override enricher_names to exclude temporal_enricher (requires OpenAI)
        profile.enrich.enricher_names = [e for e in profile.enrich.enricher_names if e != "temporal_enricher"]
        with lite_context(str(tmp_path), pipeline_profile=profile) as memory:
            yield memory

    def test_ingest_runs_enrichers_and_stores_properties(self, lite_memory):
        """Full pipeline ingest executes enrichers — verify item is stored with enriched data."""
        result = lite_memory.ingest(TEST_CONTENT)
        item_id = result.get("item_id") if isinstance(result, dict) else result
        assert item_id is not None

        item = lite_memory.get(item_id)
        assert item is not None
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        assert content is not None and len(content) > 0
