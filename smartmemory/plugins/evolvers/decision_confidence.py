"""Decision Confidence Evolver.

Background job that applies evidence-based reinforcement/contradiction and
confidence decay to decisions. Follows the same pattern as
OpinionReinforcementEvolver — matching recent episodic/semantic evidence
against active decisions.

Operations:
1. Get all active decisions
2. Get recent evidence (episodic + semantic + opinion memories)
3. Match evidence to decisions (keyword + semantic similarity)
4. Reinforce or contradict via the Decision model
5. Apply decay to decisions with no recent evidence or activity
6. Retract decisions that fall below confidence threshold
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.decision import Decision
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

_logger = logging.getLogger(__name__)

CONTRADICTION_SIGNALS = ("not", "never", "stopped", "changed", "no longer", "unlike")


@dataclass
class DecisionConfidenceConfig(MemoryBaseModel):
    """Configuration for decision confidence evolution."""

    min_confidence_threshold: float = 0.1
    decay_after_days: int = 30
    decay_rate: float = 0.05
    enable_decay: bool = True
    lookback_days: int = 7
    similarity_threshold: float = 0.7
    enable_reinforcement: bool = True


class DecisionConfidenceEvolver(EvolverPlugin):
    """Apply evidence-based reinforcement/contradiction and confidence decay to decisions.

    Operations:
    1. Get all active decisions
    2. Find recent evidence matching each decision
    3. Reinforce (supporting) or contradict (opposing) via Decision model
    4. Apply decay to decisions with no recent activity
    5. Retract decisions that fall below confidence threshold
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="decision_confidence",
            version="2.0.0",
            author="SmartMemory Team",
            description="Evidence-based reinforcement, contradiction, decay, and retraction for decisions",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["decision", "confidence", "decay", "reinforcement"],
        )

    def __init__(self, config: DecisionConfidenceConfig | None = None):
        self.config = config or DecisionConfidenceConfig()

    def evolve(self, memory: Any, logger: Any = None) -> None:
        """Apply evidence matching, reinforcement/contradiction, decay, and retraction.

        Args:
            memory: SmartMemory instance.
            logger: Optional logger override.
        """
        log = logger or _logger
        cfg = self.config
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.decision_confidence", {"memory_id": memory_id, "lookback_days": cfg.lookback_days}):
            decisions = self._get_active_decisions(memory)
            if not decisions:
                log.info("No active decisions found for evolution")
                return

            log.info(f"Processing {len(decisions)} active decisions")

            # Fetch evidence once for the whole batch
            evidence: list[MemoryItem] = []
            if cfg.enable_reinforcement:
                evidence = self._get_recent_evidence(memory, cfg.lookback_days)
                log.info(f"Found {len(evidence)} recent evidence items")

            reinforced = 0
            contradicted = 0
            decayed = 0
            retracted = 0

            for item in decisions:
                meta = item.metadata
                if meta.get("status") != "active":
                    continue

                decision = Decision.from_dict(meta)
                changed = False

                # Evidence matching
                if cfg.enable_reinforcement and evidence:
                    supporting, contradicting = self._find_matching_evidence(
                        decision, item, evidence, cfg.similarity_threshold
                    )

                    if supporting:
                        for ev in supporting:
                            decision.reinforce(ev.item_id)
                        reinforced += 1
                        changed = True
                        log.debug(f"Reinforced decision {decision.decision_id} with {len(supporting)} evidence")

                    if contradicting:
                        for ev in contradicting:
                            decision.contradict(ev.item_id)
                        contradicted += 1
                        changed = True
                        log.debug(f"Contradicted decision {decision.decision_id} with {len(contradicting)} evidence")

                # Decay if no evidence matched and decision is stale
                if cfg.enable_decay and not changed:
                    last_activity = self._get_last_activity(meta)
                    if self._should_decay(last_activity, cfg.decay_after_days):
                        decision.confidence = max(0.0, decision.confidence - cfg.decay_rate)
                        decision.updated_at = datetime.now(timezone.utc)
                        changed = True
                        decayed += 1

                # Retract if below threshold
                if decision.confidence < cfg.min_confidence_threshold:
                    self._retract_decision(memory, decision)
                    retracted += 1
                    continue

                # Persist changes
                if changed:
                    self._persist_decision(memory, decision)

            log.info(
                f"Decision evolution complete: {reinforced} reinforced, "
                f"{contradicted} contradicted, {decayed} decayed, {retracted} retracted"
            )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_active_decisions(self, memory: Any) -> list:
        """Get all active decision memories."""
        try:
            results = memory.search(query="*", memory_type="decision", top_k=500)
            return results if results else []
        except Exception:
            try:
                return memory.search(query="decision", memory_type="decision", top_k=500) or []
            except Exception as e:
                _logger.warning(f"Failed to search for decisions: {e}")
                return []

    def _get_recent_evidence(self, memory: Any, days: int) -> list[MemoryItem]:
        """Get recent episodic + semantic + opinion memories as evidence.

        Args:
            memory: SmartMemory instance.
            days: Look back this many days for evidence (used when episodic API available).
        """
        evidence: list[MemoryItem] = []
        fetched_episodic = False

        # Try the episodic-specific API first (supports time filtering)
        if days and hasattr(memory, "episodic") and hasattr(memory.episodic, "get_events_since"):
            try:
                results = memory.episodic.get_events_since(days=days)
                if results:
                    evidence.extend(results)
                    fetched_episodic = True
            except Exception:
                pass

        # Fetch each evidence type via standard search (memory_type singular, top_k)
        # Skip episodic if already fetched via the dedicated API above
        for mt in ("episodic", "semantic", "opinion"):
            if mt == "episodic" and fetched_episodic:
                continue
            try:
                results = memory.search(query="*", memory_type=mt, top_k=500)
                if results:
                    evidence.extend(results)
            except Exception:
                try:
                    results = memory.search(query=mt, memory_type=mt, top_k=500) or []
                    if results:
                        evidence.extend(results)
                except Exception as e:
                    _logger.debug(f"Failed to fetch {mt} evidence: {e}")
        return evidence

    # ------------------------------------------------------------------
    # Evidence matching
    # ------------------------------------------------------------------

    def _find_matching_evidence(
        self,
        decision: Decision,
        decision_item: MemoryItem,
        evidence_items: list[MemoryItem],
        threshold: float,
    ) -> tuple[list[MemoryItem], list[MemoryItem]]:
        """Find evidence that supports or contradicts a decision.

        Matches on:
        - Keyword: decision content, domain, and tags against evidence content
        - Semantic: cosine similarity of embeddings above threshold

        Args:
            decision: Reconstructed Decision model (owns field names).
            decision_item: Raw MemoryItem (provides embedding for semantic match).
            evidence_items: Candidate evidence to match against.
            threshold: Minimum cosine similarity for semantic match.

        Returns:
            Tuple of (supporting, contradicting) evidence lists.
        """
        supporting: list[MemoryItem] = []
        contradicting: list[MemoryItem] = []

        decision_content = (decision.content or "").lower()
        decision_domain = (decision.domain or "").lower()
        decision_tags = [t.lower() for t in (decision.tags or [])]

        for ev in evidence_items:
            ev_content = (ev.content or "").lower()
            if not ev_content:
                continue

            matched = False

            # Keyword match: decision content words in evidence
            if decision_content and decision_content in ev_content:
                matched = True

            # Domain match
            if not matched and decision_domain and decision_domain in ev_content:
                matched = True

            # Tag match
            if not matched and decision_tags:
                if any(tag in ev_content for tag in decision_tags):
                    matched = True

            # Semantic match via embeddings
            if not matched and decision_item.embedding and ev.embedding:
                similarity = MemoryItem.cosine_similarity(decision_item.embedding, ev.embedding)
                if similarity >= threshold:
                    matched = True

            if matched:
                is_contradiction = any(signal in ev_content for signal in CONTRADICTION_SIGNALS)
                if is_contradiction:
                    contradicting.append(ev)
                else:
                    supporting.append(ev)

        return supporting, contradicting

    # ------------------------------------------------------------------
    # Staleness helpers
    # ------------------------------------------------------------------

    def _get_last_activity(self, meta: dict) -> datetime | None:
        """Get the most recent activity timestamp from decision metadata."""
        timestamps: list[datetime] = []
        for field in ("last_reinforced_at", "last_contradicted_at", "updated_at"):
            val = meta.get(field)
            if val:
                if isinstance(val, str):
                    try:
                        timestamps.append(datetime.fromisoformat(val))
                    except (ValueError, TypeError):
                        continue
                elif isinstance(val, datetime):
                    timestamps.append(val)
        return max(timestamps) if timestamps else None

    def _should_decay(self, last_activity: datetime | None, decay_after_days: int) -> bool:
        """Check if a decision should decay based on time since last activity."""
        if last_activity is None:
            return True
        days_since = (datetime.now(timezone.utc) - last_activity).days
        return days_since > decay_after_days

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_decision(self, memory: Any, decision: Decision) -> None:
        """Persist full Decision state back to storage."""
        try:
            memory.update_properties(decision.decision_id, decision.to_dict())
        except Exception as e:
            _logger.warning(f"Failed to update decision {decision.decision_id}: {e}")

    def _retract_decision(self, memory: Any, decision: Decision) -> None:
        """Retract a decision that fell below threshold."""
        try:
            decision.status = "retracted"
            decision.updated_at = datetime.now(timezone.utc)
            decision.context_snapshot = decision.context_snapshot or {}
            decision.context_snapshot["retraction_reason"] = "confidence_below_threshold"
            memory.update_properties(decision.decision_id, decision.to_dict())
        except Exception as e:
            _logger.warning(f"Failed to retract decision {decision.decision_id}: {e}")
