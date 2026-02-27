"""PatternManager — hot-reloadable dictionary of learned entity patterns.

Design: dictionary scan (O(n) in text length), not spaCy EntityRuler pipe rebuild.
Patterns are stored as EntityPattern nodes in the ontology graph and cached in-memory
as a dict[str, str] of name.lower() → entity_type.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict

from smartmemory.ontology.promotion import COMMON_WORD_BLOCKLIST

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph

logger = logging.getLogger(__name__)


class PatternManager:
    """Manage learned entity patterns with hot-reload via Redis pub/sub.

    Usage::

        pm = PatternManager(ontology_graph, workspace_id="acme")
        patterns = pm.get_patterns()  # {"python": "Technology", "react": "Technology"}

    EntityRulerStage scans tokens against this dict after running spaCy NER.
    """

    RELOAD_CHANNEL_PREFIX = "smartmemory:patterns:reload"

    def __init__(
        self,
        ontology_graph: OntologyGraph,
        workspace_id: str = "default",
        redis_client=None,
    ):
        self._ontology = ontology_graph
        self._workspace_id = workspace_id
        self._redis = redis_client
        self._patterns: Dict[str, str] = {}
        self._version = 0
        self._lock = threading.Lock()
        self._subscriber_thread: threading.Thread | None = None

        # Initial load
        self._build_patterns()

        # Start subscriber if Redis is available
        if self._redis is not None:
            self._subscribe_for_reload()

    def get_patterns(self) -> Dict[str, str]:
        """Return cached name→type dict."""
        with self._lock:
            return dict(self._patterns)

    def add_patterns(self, patterns: Dict[str, str]) -> int:
        """Merge additional name→type patterns into the in-memory cache.

        Filters short names and common word blocklist entries before accepting.
        Persists accepted patterns to the ontology graph (best-effort) so they
        survive restarts.

        Args:
            patterns: Mapping of entity name to entity type label.

        Returns:
            Number of net-new patterns accepted (duplicates excluded).
        """
        accepted = 0
        for name, label in patterns.items():
            if not name or not label:
                continue
            key = name.lower().strip()
            if key in COMMON_WORD_BLOCKLIST or len(key) < 2:
                continue
            with self._lock:
                if key not in self._patterns:
                    self._patterns[key] = label
                    self._version += 1
                    accepted += 1
                else:
                    continue  # already cached — skip persistence too
            # Persist to ontology graph best-effort (outside lock to avoid I/O under lock)
            try:
                self._ontology.add_entity_pattern(
                    name=key,
                    label=label,
                    confidence=0.9,
                    workspace_id=self._workspace_id,
                    source="code_index",
                )
            except Exception as exc:
                logger.debug("Failed to persist code pattern '%s': %s", key, exc)
        return accepted

    def _build_patterns(self) -> None:
        """Load EntityPattern nodes from ontology graph, filtered by blocklist."""
        try:
            raw = self._ontology.get_entity_patterns(self._workspace_id)
            new_patterns: Dict[str, str] = {}
            for p in raw:
                name = p.get("name", "").lower().strip()
                label = p.get("label", "")
                if not name or not label:
                    continue
                if name in COMMON_WORD_BLOCKLIST:
                    continue
                if len(name) < 2:
                    continue
                new_patterns[name] = label
            with self._lock:
                self._patterns = new_patterns
                self._version += 1
            logger.debug("Loaded %d entity patterns for workspace '%s'", len(new_patterns), self._workspace_id)
        except Exception as e:
            logger.warning("Failed to build entity patterns: %s", e)

    def reload(self) -> None:
        """Force reload patterns from graph."""
        self._build_patterns()

    def notify_reload(self, workspace_id: str | None = None) -> None:
        """Publish reload notification via Redis pub/sub."""
        if self._redis is None:
            return
        ws = workspace_id or self._workspace_id
        channel = f"{self.RELOAD_CHANNEL_PREFIX}:{ws}"
        try:
            self._redis.publish(channel, "reload")
        except Exception as e:
            logger.debug("Failed to publish reload notification: %s", e)

    def _subscribe_for_reload(self) -> None:
        """Background thread subscribing to reload channel."""
        channel = f"{self.RELOAD_CHANNEL_PREFIX}:{self._workspace_id}"

        def _listener():
            try:
                pubsub = self._redis.pubsub()
                pubsub.subscribe(channel)
                for message in pubsub.listen():
                    if message.get("type") == "message":
                        self._build_patterns()
            except Exception as e:
                logger.debug("Pattern reload subscriber stopped: %s", e)

        self._subscriber_thread = threading.Thread(target=_listener, daemon=True, name="pattern-reload")
        self._subscriber_thread.start()

    @property
    def version(self) -> int:
        """Current pattern version (incremented on each reload)."""
        return self._version

    @property
    def pattern_count(self) -> int:
        """Number of active patterns."""
        with self._lock:
            return len(self._patterns)
