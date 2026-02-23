import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata

# Minimal English stopword list to avoid external deps
_STOPWORDS = set(
    '''a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours  ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up very
    was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's with won't would wouldn't
    you you'd you'll you're you've your yours yourself yourselves'''.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")


@dataclass
class TopicEnricherConfig(MemoryBaseModel):
    enabled: bool = True
    max_topics: int = 5


@dataclass
class TopicEnricherRequest(StageRequest):
    enabled: bool = True
    max_topics: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class TopicEnricher(EnricherPlugin):
    """
    Topic enrichment plugin using a simple frequency-based heuristic.
    - No external dependencies.
    - Produces a list of topics with normalized scores and a dominant topic label.

    Returns enrichment result with properties:
      topics: [ { label: str, score: float } ],
      dominant_topic: str,
      keywords: [str],
      topic_method: 'heuristic',
      topic_version: '0.1'
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="topic_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="Frequency-based topic extraction with keyword analysis",
            plugin_type="enricher",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def __init__(self, config: Optional[TopicEnricherConfig] = None):
        self.config = config or TopicEnricherConfig()
        if not isinstance(self.config, TopicEnricherConfig):
            raise TypeError("TopicEnricher requires a typed config (TopicEnricherConfig)")

    def _extract_keywords(self, text: str, max_topics: int) -> List[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
        terms = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
        if not terms:
            return []
        counts = Counter(terms)
        # Return top max_topics labels
        return [w for w, _ in counts.most_common(max_topics)]

    def enrich(self, item, node_ids=None) -> Dict[str, Any]:
        if not isinstance(self.config, TopicEnricherConfig):
            raise TypeError("TopicEnricher requires a typed config (TopicEnricherConfig)")
        if not self.config.enabled:
            return {}
        max_topics = int(self.config.max_topics)
        memory_id = getattr(item, 'item_id', None)
        with trace_span("pipeline.enrich.topic_enricher", {"memory_id": memory_id, "max_topics": max_topics}):
            try:
                content = getattr(item, 'content', str(item))
            except Exception:
                logging.exception("Error getting content from item")
                content = str(item)

            keywords = self._extract_keywords(content, max_topics)
            if not keywords:
                return {
                    'properties': {
                        'topics': [],
                        'dominant_topic': None,
                        'keywords': [],
                        'topic_method': 'heuristic',
                        'topic_version': '0.1'
                    }
                }

            # Build scores normalized by rank or frequency
            # Use frequency normalization
            tokens = [t.lower() for t in _TOKEN_RE.findall(content)]
            terms = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
            counts = Counter(terms)
            total = sum(counts[w] for w in keywords) or 1
            topics = [
                {
                    'label': w,
                    'score': float(counts[w] / total)
                }
                for w in keywords
            ]
            # Sort by score desc
            topics.sort(key=lambda x: x['score'], reverse=True)
            dominant = topics[0]['label'] if topics else None

        return {
            'properties': {
                'topics': topics,
                'dominant_topic': dominant,
                'keywords': keywords,
                'topic_method': 'heuristic',
                'topic_version': '0.1'
            }
        }
