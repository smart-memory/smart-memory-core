"""
LinkExpansionEnricher - Expands URLs in memory items into rich graph structures.

Fetches URL content, extracts metadata (title, description, OG tags), and optionally
uses LLM for summarization and entity extraction. Creates WebResource nodes linked
to Entity child nodes.
"""

from __future__ import annotations

import hashlib
import json as json_module
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from smartmemory.integration.llm.prompts.prompts_loader import (
    apply_placeholders,
    get_prompt_value,
)
from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata

logger = logging.getLogger(__name__)

# URL pattern - matches http/https URLs
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')  # noqa: E501


@dataclass
class LinkExpansionEnricherConfig(MemoryBaseModel):
    """Configuration for LinkExpansionEnricher."""

    # LLM settings
    enable_llm: bool = False
    model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Fetch settings
    timeout_seconds: int = 10
    max_urls_per_item: int = 5
    user_agent: str = "SmartMemory/0.2.7"

    # Content extraction
    max_content_length: int = 50000

    # Prompt template for LLM summarization
    prompt_template_key: str = "enrichers.link_expansion.prompt_template"


@dataclass
class LinkExpansionEnricherRequest(StageRequest):
    """Request object for LinkExpansionEnricher service layer."""

    enable_llm: bool = False
    model_name: str = "gpt-4o-mini"
    timeout_seconds: int = 10
    max_urls_per_item: int = 5
    context: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None


class LinkExpansionEnricher(EnricherPlugin):
    """
    Enricher that expands URLs in memory items into rich graph structures.

    Fetches URL content, extracts metadata (title, description, OG tags),
    and optionally uses LLM for summarization and entity extraction.
    Creates WebResource nodes linked to Entity child nodes via MENTIONS edges.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="link_expansion_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="Expands URLs into rich graph structures with metadata and entities",
            plugin_type="enricher",
            dependencies=["httpx>=0.24.0", "beautifulsoup4>=4.12.0"],
            min_smartmemory_version="0.2.7",
            requires_network=True,
            requires_llm=False,  # Optional - only when enable_llm=True
        )

    def __init__(self, config: LinkExpansionEnricherConfig | None = None):
        """Initialize the enricher with optional configuration.

        Args:
            config: Configuration for the enricher. If None, uses defaults.

        Raises:
            TypeError: If config is provided but not a LinkExpansionEnricherConfig.
        """
        self.config = config or LinkExpansionEnricherConfig()
        if not isinstance(self.config, LinkExpansionEnricherConfig):
            raise TypeError("LinkExpansionEnricher requires typed config (LinkExpansionEnricherConfig)")

    def _extract_urls(self, item, node_ids: dict[str, Any] | None) -> list[str]:
        """Extract URLs from content, then merge with extraction stage output.

        Args:
            item: The memory item (string or object with content attribute).
            node_ids: Optional dict that may contain 'urls' key.

        Returns:
            list: Deduplicated list of URLs, limited to max_urls_per_item.
        """
        # Get content from item
        if hasattr(item, "content"):
            content = item.content
        else:
            content = str(item)

        # 1. Regex extraction
        urls = set(URL_PATTERN.findall(content))

        # 2. Merge with extraction stage (if present)
        if isinstance(node_ids, dict):
            urls.update(node_ids.get("urls", []))

        # 3. Dedupe and limit
        return list(urls)[: self.config.max_urls_per_item]

    def _fetch_url(self, url: str) -> dict:
        """Fetch URL and return result dict with status.

        Args:
            url: The URL to fetch.

        Returns:
            dict: Result with status, html/error, final_url, content_type.
        """
        try:
            response = httpx.get(
                url,
                timeout=self.config.timeout_seconds,
                headers={"User-Agent": self.config.user_agent},
                follow_redirects=True,
            )
            response.raise_for_status()

            html = response.text[: self.config.max_content_length]
            return {
                "status": "success",
                "html": html,
                "final_url": str(response.url),
                "content_type": response.headers.get("content-type", ""),
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _get_meta(self, soup, name: str) -> str | None:
        """Get meta tag content by property or name.

        Args:
            soup: BeautifulSoup parsed HTML.
            name: Meta property or name to look for.

        Returns:
            Content string or None if not found.
        """
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        return tag.get("content") if tag else None

    def _get_canonical(self, soup) -> str | None:
        """Get canonical URL from link tag.

        Args:
            soup: BeautifulSoup parsed HTML.

        Returns:
            Canonical URL or None if not found.
        """
        link = soup.find("link", rel="canonical")
        return link.get("href") if link else None

    def _extract_metadata(self, html: str, url: str) -> dict:
        """Extract metadata from HTML using heuristics.

        Args:
            html: HTML content string.
            url: Original URL for fallback values.

        Returns:
            dict: Extracted metadata.
        """
        if not HAS_BS4:
            return {
                "title": url,
                "description": None,
                "og_image": None,
                "og_type": None,
                "author": None,
                "published_date": None,
                "domain": urlparse(url).netloc,
                "canonical_url": url,
            }

        soup = BeautifulSoup(html, "html.parser")

        # Title: og:title > twitter:title > <title>
        title = (
            self._get_meta(soup, "og:title")
            or self._get_meta(soup, "twitter:title")
            or (soup.title.string if soup.title else None)
            or url
        )

        # Description: og:description > meta description
        description = self._get_meta(soup, "og:description") or self._get_meta(soup, "description")

        return {
            "title": title[:500] if title else url,
            "description": description[:1000] if description else None,
            "og_image": self._get_meta(soup, "og:image"),
            "og_type": self._get_meta(soup, "og:type"),
            "author": self._get_meta(soup, "author"),
            "published_date": self._get_meta(soup, "article:published_time"),
            "domain": urlparse(url).netloc,
            "canonical_url": self._get_canonical(soup) or url,
        }

    def _parse_jsonld(self, ld: dict) -> list[dict]:
        """Parse JSON-LD structured data to extract entities.

        Args:
            ld: JSON-LD object.

        Returns:
            list: Extracted entity dicts.
        """
        entities = []
        ld_type = ld.get("@type", "")

        # Map JSON-LD types to entity types
        type_map = {
            "Person": "PERSON",
            "Organization": "ORG",
            "Corporation": "ORG",
            "Product": "PRODUCT",
            "Place": "LOCATION",
            "Event": "EVENT",
        }

        # Direct entity
        if ld_type in type_map and ld.get("name"):
            entities.append(
                {
                    "name": ld["name"],
                    "type": type_map[ld_type],
                    "source": "jsonld",
                }
            )

        # Nested author
        author = ld.get("author")
        if isinstance(author, dict) and author.get("name"):
            author_type = type_map.get(author.get("@type", ""), "PERSON")
            entities.append(
                {
                    "name": author["name"],
                    "type": author_type,
                    "source": "jsonld",
                }
            )
        elif isinstance(author, str):
            entities.append(
                {
                    "name": author,
                    "type": "PERSON",
                    "source": "jsonld",
                }
            )

        # Nested publisher
        publisher = ld.get("publisher")
        if isinstance(publisher, dict) and publisher.get("name"):
            entities.append(
                {
                    "name": publisher["name"],
                    "type": "ORG",
                    "source": "jsonld",
                }
            )

        return entities

    def _extract_entities_heuristic(self, html: str, metadata: dict) -> list[dict]:
        """Extract entities from structured data without LLM.

        Args:
            html: HTML content string.
            metadata: Previously extracted metadata.

        Returns:
            list: Extracted entity dicts with name, type, source.
        """
        if not HAS_BS4:
            return []

        soup = BeautifulSoup(html, "html.parser")
        entities = []
        seen_names: set[str] = set()

        # Author from metadata as PERSON
        if metadata.get("author"):
            name = metadata["author"]
            if name not in seen_names:
                entities.append({"name": name, "type": "PERSON", "source": "meta"})
                seen_names.add(name)

        # JSON-LD structured data (Schema.org)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                script_content = script.string
                if script_content is None:
                    continue
                ld = json_module.loads(script_content)
                for entity in self._parse_jsonld(ld):
                    if entity["name"] not in seen_names:
                        entities.append(entity)
                        seen_names.add(entity["name"])
            except (json_module.JSONDecodeError, TypeError, AttributeError):
                pass

        return entities

    def _analyze_with_llm(self, html: str, metadata: dict) -> dict:
        """Analyze content with LLM for summary and entities.

        Args:
            html: HTML content (will extract text).
            metadata: Previously extracted metadata.

        Returns:
            dict: LLM analysis results with summary, entities, topics.
        """
        if not HAS_OPENAI:
            logger.warning("OpenAI not available for LLM analysis")
            return {}

        if not HAS_BS4:
            return {}

        # Extract text from HTML
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)[:8000]

        # Get prompt template
        template = get_prompt_value(self.config.prompt_template_key)
        if not template:
            logger.warning(f"Prompt template not found: {self.config.prompt_template_key}")
            return {}

        prompt = apply_placeholders(
            template,
            {"CONTENT": text, "TITLE": metadata.get("title", "")},
        )

        try:
            # Configure API key if provided
            if self.config.openai_api_key:
                openai.api_key = self.config.openai_api_key

            response = openai.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            # Track token usage (CFS-1b)
            self._track_usage(response)

            message_content = response.choices[0].message.content
            if message_content is None:
                return {}
            result: dict[str, Any] = json_module.loads(message_content)
            return result
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return {}

    def _track_usage(self, response) -> None:
        """Record token usage from OpenAI response (CFS-1b)."""
        try:
            from smartmemory.plugins.enrichers.usage_tracking import record_enricher_usage

            if hasattr(response, "usage") and response.usage:
                record_enricher_usage(
                    enricher_name="link_expansion_enricher",
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(response.usage, "completion_tokens", 0),
                    model=self.config.model_name,
                )
        except ImportError:
            pass

    def enrich(self, item, node_ids=None) -> dict:
        """Enrich a memory item by expanding URLs into graph structures.

        Args:
            item: The memory item to enrich (string or object with content).
            node_ids: Optional dict of node IDs from extraction stage.

        Returns:
            dict: Enrichment results with web_resources, provenance_candidates, tags.
        """
        urls = self._extract_urls(item, node_ids)
        memory_id = getattr(item, 'item_id', None)
        with trace_span("pipeline.enrich.link_expansion_enricher", {"memory_id": memory_id, "url_count": len(urls)}):
            if not urls:
                return {
                    "web_resources": [],
                    "provenance_candidates": [],
                    "tags": [],
                }

            results = []
            all_entities: list[dict] = []
            provenance_candidates = []

            for url in urls:
                # Generate node ID
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                node_id = f"webresource:{url_hash}"

                # Fetch URL
                fetch_result = self._fetch_url(url)

                if fetch_result["status"] == "failed":
                    results.append(
                        {
                            "url": url,
                            "node_id": node_id,
                            "status": "failed",
                            "error": fetch_result.get("error"),
                            "error_type": fetch_result.get("error_type"),
                        }
                    )
                    provenance_candidates.append(node_id)
                    continue

                # Extract metadata (heuristic - always runs)
                metadata = self._extract_metadata(fetch_result["html"], url)
                metadata["fetched_at"] = datetime.now(timezone.utc).isoformat()

                # Extract entities (heuristic - always runs)
                entities = self._extract_entities_heuristic(fetch_result["html"], metadata)

                # Build result
                resource = {
                    "url": url,
                    "node_id": node_id,
                    "status": "success",
                    "metadata": metadata,
                    "extracted_entities": entities,
                    "summary": None,  # LLM only
                }

                # LLM analysis (optional)
                if self.config.enable_llm:
                    llm_result = self._analyze_with_llm(fetch_result["html"], metadata)
                    if llm_result:
                        resource["summary"] = llm_result.get("summary")
                        # Add LLM entities
                        for entity in llm_result.get("entities", []):
                            if isinstance(entity, dict) and entity.get("name"):
                                entities.append(
                                    {
                                        "name": entity["name"],
                                        "type": entity.get("type", "TOPIC"),
                                        "source": "llm",
                                    }
                                )

                results.append(resource)
                all_entities.extend(entities)
                provenance_candidates.append(node_id)

                # Create graph nodes if graph available
                if hasattr(self, "graph") and self.graph is not None:
                    # Create WebResource node
                    self.graph.add_node(
                        item_id=node_id,
                        properties={
                            **metadata,
                            "url": url,
                            "type": "web_resource",
                        },
                    )

                    # Create Entity nodes and MENTIONS edges
                    for entity in entities:
                        entity_id = f"entity:{entity['name'].lower().replace(' ', '_')}"
                        self.graph.add_node(
                            item_id=entity_id,
                            properties={
                                **entity,
                                "type": "extracted_entity",
                            },
                        )
                        self.graph.add_edge(node_id, entity_id, "MENTIONS")

            # Deduplicate entity names for tags
            tags = list(set(e["name"] for e in all_entities))

        return {
            "web_resources": results,
            "provenance_candidates": provenance_candidates,
            "tags": tags,
        }
