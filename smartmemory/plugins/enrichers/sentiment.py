import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata

try:
    # Optional dependency; used if available
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    _HAS_VADER = True
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore
    _HAS_VADER = False


@dataclass
class SentimentEnricherConfig(MemoryBaseModel):
    enabled: bool = True
    backend: str = 'auto'  # 'auto' | 'vader' | 'heuristic'


@dataclass
class SentimentEnricherRequest(StageRequest):
    enabled: bool = True
    backend: str = 'auto'
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class SentimentEnricher(EnricherPlugin):
    """
    Sentiment enrichment plugin.
    - Tries vaderSentiment if installed.
    - Falls back to a lightweight lexicon-based heuristic with no external deps.

    Returns enrichment result with properties.sentiment = {
      score: float in [-1, 1],
      label: 'negative'|'neutral'|'positive',
      method: 'vader'|'heuristic',
      confidence: float in [0, 1],
      version: str
    }
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="sentiment_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="Sentiment analysis enrichment using VADER or heuristic fallback",
            plugin_type="enricher",
            dependencies=["vaderSentiment>=3.3.2"],  # Optional dependency
            min_smartmemory_version="0.1.0"
        )

    def __init__(self, config: Optional[SentimentEnricherConfig] = None):
        self.config = config or SentimentEnricherConfig()
        if not isinstance(self.config, SentimentEnricherConfig):
            raise TypeError("SentimentEnricher requires a typed config (SentimentEnricherConfig)")
        if _HAS_VADER:
            try:
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                self._vader = None
        else:
            self._vader = None

    def _cfg(self) -> Dict[str, Any]:
        # Backwards-compatible adapter if needed elsewhere; use typed config here
        return {'enabled': self.config.enabled, 'backend': self.config.backend}

    def _label_from_score(self, score: float) -> str:
        # Common thresholds similar to VADER defaults
        if score >= 0.05:
            return 'positive'
        if score <= -0.05:
            return 'negative'
        return 'neutral'

    def _heuristic_sentiment(self, text: str) -> Dict[str, Any]:
        # Minimal keyword-based sentiment without external deps
        pos_words = {
            'good', 'great', 'excellent', 'positive', 'fortunate', 'correct', 'superior', 'happy', 'love', 'like', 'awesome', 'amazing', 'win', 'success', 'benefit', 'best'
        }
        neg_words = {
            'bad', 'terrible', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior', 'sad', 'hate', 'dislike', 'awful', 'horrible', 'lose', 'failure', 'risk', 'worst'
        }
        tokens = [t.strip(".,!?;:""'()[]{}-").lower() for t in text.split()]
        pos = sum(1 for t in tokens if t in pos_words)
        neg = sum(1 for t in tokens if t in neg_words)
        total = pos + neg
        if total == 0:
            score = 0.0
        else:
            score = (pos - neg) / max(total, 1)
            score = max(-1.0, min(1.0, score))
        label = self._label_from_score(score)
        confidence = min(1.0, abs(score))
        return {
            'score': float(score),
            'label': label,
            'method': 'heuristic',
            'confidence': float(confidence),
            'version': '0.1'
        }

    def _vader_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        if not self._vader:
            return None
        try:
            scores = self._vader.polarity_scores(text)
            compound = float(scores.get('compound', 0.0))
            label = self._label_from_score(compound)
            confidence = min(1.0, abs(compound))
            return {
                'score': compound,
                'label': label,
                'method': 'vader',
                'confidence': float(confidence),
                'version': '3.3.2'  # vaderSentiment version baseline
            }
        except Exception:
            logging.exception("SentimentEnricher: failed to compute vader sentiment")
            return None

    def enrich(self, item, node_ids=None) -> Dict[str, Any]:
        # Fail fast on typed config and check enabled
        if not isinstance(self.config, SentimentEnricherConfig):
            raise TypeError("SentimentEnricher requires a typed config (SentimentEnricherConfig)")
        if not self.config.enabled:
            return {}
        backend = (self.config.backend or 'auto').lower()
        memory_id = getattr(item, 'item_id', None)
        with trace_span("pipeline.enrich.sentiment_enricher", {"memory_id": memory_id, "backend": backend}):
            try:
                content = getattr(item, 'content', str(item))
            except Exception:
                logging.exception("SentimentEnricher: failed to extract content from item")
                content = str(item)

            sentiment: Optional[Dict[str, Any]] = None
            if backend in ('auto', 'vader') and self._vader:
                sentiment = self._vader_sentiment(content)
            if sentiment is None:
                # Fallback or when backend=='heuristic'
                sentiment = self._heuristic_sentiment(content)

        return {
            'properties': {
                'sentiment': sentiment
            }
        }
