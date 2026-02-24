"""Coreference resolution stage for preprocessing.

Resolves pronouns and vague references to explicit entity names before extraction.
Uses fastcoref for high-quality coreference resolution.

Example:
    "Apple announced quarterly results. The company exceeded expectations."
    → "Apple announced quarterly results. Apple exceeded expectations."
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smartmemory.models.base import MemoryBaseModel

logger = logging.getLogger(__name__)

# Lazy load models to avoid slow imports
_nlp = None
_coref_model = None


def _detect_device() -> str:
    """Detect best available device for inference."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_coref_model(device: str = "auto"):
    """Get or initialize fastcoref model.

    Args:
        device: Device to use ('auto', 'mps', 'cuda', 'cpu')

    Returns:
        FCoref model instance or None if unavailable
    """
    global _coref_model

    if _coref_model is None:
        try:
            from fastcoref import FCoref

            if device == "auto":
                device = _detect_device()

            _coref_model = FCoref(device=device)
            logger.info(f"Loaded fastcoref model (device={device})")
        except ImportError:
            logger.warning(
                "fastcoref not installed - coreference resolution disabled. "
                "Install with: pip install fastcoref"
            )
            _coref_model = False  # Mark as unavailable
        except Exception as e:
            logger.warning(f"Failed to load fastcoref: {e}")
            _coref_model = False

    return _coref_model if _coref_model else None


@dataclass
class CoreferenceResult(MemoryBaseModel):
    """Result of coreference resolution."""

    original_text: str
    """Original input text."""

    resolved_text: str
    """Text with coreferences resolved."""

    chains: List[Dict[str, Any]] = field(default_factory=list)
    """Coreference chains found: [{"mentions": [...], "head": "Apple"}, ...]"""

    replacements_made: int = 0
    """Number of replacements made."""

    skipped: bool = False
    """Whether resolution was skipped (e.g., text too short)."""

    skip_reason: Optional[str] = None
    """Reason for skipping, if applicable."""


@dataclass
class CoreferenceChain:
    """A single coreference chain."""

    mentions: List[str]
    """All mentions in this chain."""

    head: str
    """The most informative mention (used for replacements)."""

    entity_type: Optional[str] = None
    """Entity type if detected (PERSON, ORG, etc.)."""


class CoreferenceStage:
    """Pipeline stage for coreference resolution.

    Resolves pronouns and vague references to explicit entity names.
    This improves entity extraction quality by making implicit references explicit.

    Usage:
        stage = CoreferenceStage(device="mps")
        result = stage.run("Apple announced... The company...")
        print(result.resolved_text)  # "Apple announced... Apple..."
    """

    def __init__(
        self,
        resolver: str = "fastcoref",
        device: str = "auto",
        min_text_length: int = 50,
    ):
        """Initialize the coreference stage.

        Args:
            resolver: Resolver to use ('fastcoref' or 'spacy')
            device: Device for inference ('auto', 'mps', 'cuda', 'cpu')
            min_text_length: Skip texts shorter than this
        """
        self.resolver = resolver
        self.device = device
        self.min_text_length = min_text_length
        self._model = None  # Lazy load

    def _ensure_model(self):
        """Ensure the coreference model is loaded."""
        if self._model is None:
            if self.resolver == "fastcoref":
                self._model = get_coref_model(self.device)
            else:
                logger.warning(f"Unknown resolver: {self.resolver}")

    def run(self, text: str, config: Optional[Any] = None) -> CoreferenceResult:
        """Run coreference resolution on text.

        Args:
            text: Input text to process
            config: Optional CoreferenceConfig override

        Returns:
            CoreferenceResult with resolved text and chain information
        """
        # Apply config overrides if provided
        if config is not None:
            if hasattr(config, "enabled") and not config.enabled:
                return CoreferenceResult(
                    original_text=text,
                    resolved_text=text,
                    skipped=True,
                    skip_reason="Coreference resolution disabled",
                )
            if hasattr(config, "min_text_length"):
                self.min_text_length = config.min_text_length

        # Skip short texts
        if not text or len(text) < self.min_text_length:
            return CoreferenceResult(
                original_text=text,
                resolved_text=text,
                skipped=True,
                skip_reason=f"Text too short ({len(text) if text else 0} < {self.min_text_length})",
            )

        self._ensure_model()

        if self._model is None:
            return CoreferenceResult(
                original_text=text,
                resolved_text=text,
                skipped=True,
                skip_reason="Coreference model not available",
            )

        try:
            # Run fastcoref
            results = self._model.predict(texts=[text])
            if not results:
                return CoreferenceResult(
                    original_text=text,
                    resolved_text=text,
                )

            result = results[0]
            clusters = result.get_clusters()

            if not clusters:
                return CoreferenceResult(
                    original_text=text,
                    resolved_text=text,
                )

            # Process clusters and build replacements
            chains = []
            replacements = []

            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                # Find the most informative mention (head)
                head = self._find_head_mention(cluster)
                if head is None:
                    continue

                chains.append({
                    "mentions": list(cluster),
                    "head": head,
                })

                # Build replacements for other mentions
                for mention in cluster:
                    if mention == head or mention.lower() == head.lower():
                        continue

                    # Find position of mention in text
                    start = text.find(mention)
                    if start == -1:
                        continue

                    end = start + len(mention)
                    replacements.append({
                        "start": start,
                        "end": end,
                        "original": mention,
                        "replacement": head,
                    })

            # Apply replacements in reverse order to preserve positions
            resolved = text
            for r in sorted(replacements, key=lambda x: -x["start"]):
                resolved = resolved[: r["start"]] + r["replacement"] + resolved[r["end"] :]

            return CoreferenceResult(
                original_text=text,
                resolved_text=resolved,
                chains=chains,
                replacements_made=len(replacements),
            )

        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}")
            return CoreferenceResult(
                original_text=text,
                resolved_text=text,
                skipped=True,
                skip_reason=f"Error: {e}",
            )

    def _find_head_mention(self, mentions: List[str]) -> Optional[str]:
        """Find the most informative mention in a cluster."""
        best = None
        best_score = -1

        for mention in mentions:
            score = self._score_mention(mention)
            if score > best_score:
                best_score = score
                best = mention

        return best

    def _score_mention(self, mention: str) -> int:
        """Score a mention by informativeness."""
        if not mention:
            return -100

        score = 0

        # Prefer proper nouns and named entities (capitalized)
        if mention[0].isupper():
            score += 10

        # Prefer longer mentions (more specific)
        score += len(mention.split())

        # Penalize pronouns heavily
        pronouns = {
            "he", "she", "it", "they", "him", "her", "them",
            "his", "its", "their", "who", "whom", "whose",
        }
        if mention.lower() in pronouns:
            score -= 20

        # Penalize generic references
        generic = {
            "the company", "the firm", "the organization", "the group",
            "the person", "the man", "the woman", "the team",
        }
        if mention.lower() in generic:
            score -= 5

        return score
