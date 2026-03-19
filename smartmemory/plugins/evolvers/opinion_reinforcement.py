"""
Opinion Reinforcement Evolver 

Background job that updates opinion confidence scores based on new evidence.
When new evidence supports an opinion → increase confidence.
When new evidence contradicts → decrease confidence.


Similar to EpisodicDecayEvolver but for opinions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.models.memory_item import MemoryItem
from smartmemory.models.opinion import OpinionMetadata
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

_module_logger = logging.getLogger(__name__)


@dataclass
class OpinionReinforcementConfig(MemoryBaseModel):
    """Configuration for opinion reinforcement."""
    min_confidence_threshold: float = 0.2  # Archive opinions below this
    lookback_days: int = 7  # How far back to look for new evidence
    similarity_threshold: float = 0.7  # Threshold for evidence matching

    # Reinforcement/contradiction factors
    reinforcement_boost: float = 0.1  # How much to increase confidence
    contradiction_penalty: float = 0.15  # How much to decrease confidence

    # Decay for stale opinions
    enable_decay: bool = True
    decay_after_days: int = 30  # Start decaying after this many days without reinforcement
    decay_rate: float = 0.05  # How much to decay per cycle


@dataclass
class OpinionReinforcementRequest(StageRequest):
    lookback_days: int = 7
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class OpinionReinforcementEvolver(EvolverPlugin):
    """
    Updates opinion confidence scores based on new evidence.

    Operations (batch):
    1. Find new episodic memories since last run
    2. Match against existing opinions
    3. Reinforce or contradict based on semantic similarity
    4. Archive opinions that fall below threshold
    5. Apply decay to stale opinions

    Incremental (CORE-EVO-LIVE-1):
    Handles new-evidence reinforcement path only (steps 2-3).
    Decay (step 4) and archive (step 5) run on idle catch-up.
    """

    # CORE-EVO-LIVE-1: Trigger on new episodic/semantic evidence
    TRIGGERS = {("episodic", "add"), ("semantic", "add")}

    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="opinion_reinforcement",
            version="1.0.0",
            author="SmartMemory Team",
            description="Updates opinion confidence based on new evidence ",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
            tags=["reinforcement", "opinion", "confidence"]
        )

    def __init__(self, config: Optional[OpinionReinforcementConfig] = None):
        self.config = config or OpinionReinforcementConfig()

    def evolve(self, memory, log=None, logger=None):
        """
        Main evolution method - reinforces/contradicts opinions based on evidence.

        Args:
            memory: SmartMemory instance
            log: Optional logger (legacy parameter name)
            logger: Optional logger (used by evolution/cycle.py)
        """
        log = log or logger or _module_logger
        cfg = self.config
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.opinion_reinforcement", {"memory_id": memory_id, "lookback_days": cfg.lookback_days}):
            # 1. Get all existing opinions
            opinions = self._get_all_opinions(memory)
            if not opinions:
                log.info("No opinions found for reinforcement")
                return

            log.info(f"Processing {len(opinions)} opinions for reinforcement")

            # 2. Get recent evidence (episodic memories)
            recent_evidence = self._get_recent_evidence(memory, cfg.lookback_days)
            log.info(f"Found {len(recent_evidence)} recent evidence items")

            # 3. Process each opinion
            reinforced = 0
            contradicted = 0
            archived = 0
            decayed = 0

            for opinion in opinions:
                opinion_meta = self._get_opinion_metadata(opinion)
                if not opinion_meta:
                    continue

                # Find matching evidence
                supporting, contradicting = self._find_matching_evidence(
                    opinion, recent_evidence, cfg.similarity_threshold
                )

                # Apply reinforcement
                if supporting:
                    for evidence in supporting:
                        opinion_meta.reinforce(evidence.item_id)
                    reinforced += 1
                    log.debug(f"Reinforced opinion {opinion.item_id} with {len(supporting)} evidence")

                # Apply contradiction
                if contradicting:
                    for evidence in contradicting:
                        opinion_meta.contradict(evidence.item_id)
                    contradicted += 1
                    log.debug(f"Contradicted opinion {opinion.item_id} with {len(contradicting)} evidence")

                # Apply decay if no recent activity
                if cfg.enable_decay and not supporting and not contradicting:
                    if self._should_decay(opinion_meta, cfg.decay_after_days):
                        opinion_meta.confidence = max(0.0, opinion_meta.confidence - cfg.decay_rate)
                        decayed += 1

                # Check if should archive
                if opinion_meta.confidence < cfg.min_confidence_threshold:
                    self._archive_opinion(memory, opinion)
                    archived += 1
                    log.info(f"Archived opinion {opinion.item_id} (confidence: {opinion_meta.confidence:.2f})")
                else:
                    # Update the opinion
                    self._update_opinion(memory, opinion, opinion_meta)

            log.info(
                f"Opinion reinforcement complete: "
                f"{reinforced} reinforced, {contradicted} contradicted, "
                f"{decayed} decayed, {archived} archived"
            )

    def _get_all_opinions(self, memory) -> List[MemoryItem]:
        """Get all opinion memories."""
        try:
            if hasattr(memory, 'search'):
                results = memory.search(
                    query="*",
                    memory_types=['opinion'],
                    limit=500,
                )
                return results if results else []
            return []
        except Exception as e:
            _module_logger.error(f"Failed to get opinions: {e}")
            return []

    def _get_recent_evidence(self, memory, days: int) -> List[MemoryItem]:
        """Get recent episodic memories as evidence."""
        try:
            if hasattr(memory, 'episodic') and hasattr(memory.episodic, 'get_events_since'):
                return memory.episodic.get_events_since(days=days)
            
            if hasattr(memory, 'search'):
                results = memory.search(
                    query="*",
                    memory_types=['episodic'],
                    limit=500,
                )
                return results if results else []
            return []
        except Exception as e:
            _module_logger.error(f"Failed to get recent evidence: {e}")
            return []

    def _get_opinion_metadata(self, opinion: MemoryItem) -> Optional[OpinionMetadata]:
        """Extract OpinionMetadata from a MemoryItem."""
        try:
            if 'confidence' in opinion.metadata:
                return OpinionMetadata.from_dict(opinion.metadata)
            return None
        except Exception as e:
            _module_logger.error(f"Failed to parse opinion metadata: {e}")
            return None

    def _find_matching_evidence(
        self, 
        opinion: MemoryItem, 
        evidence_items: List[MemoryItem],
        threshold: float
    ) -> tuple[List[MemoryItem], List[MemoryItem]]:
        """
        Find evidence that supports or contradicts an opinion.
        
        Returns (supporting, contradicting) lists.
        """
        supporting = []
        contradicting = []
        
        opinion_subject = opinion.metadata.get('subject', '')
        opinion_content = opinion.content.lower() if opinion.content else ''
        
        for evidence in evidence_items:
            evidence_content = evidence.content.lower() if evidence.content else ''
            
            # Simple matching: check if evidence mentions the opinion subject
            if opinion_subject and opinion_subject.lower() in evidence_content:
                # Check for contradiction signals
                contradiction_signals = ['not', 'never', 'stopped', 'changed', 'no longer', 'unlike']
                is_contradiction = any(signal in evidence_content for signal in contradiction_signals)
                
                if is_contradiction:
                    contradicting.append(evidence)
                else:
                    supporting.append(evidence)
            
            # Check semantic similarity if embeddings available
            if opinion.embedding and evidence.embedding:
                similarity = MemoryItem.cosine_similarity(opinion.embedding, evidence.embedding)
                if similarity >= threshold:
                    # High similarity = supporting (unless contradiction signals present)
                    contradiction_signals = ['not', 'never', 'stopped', 'changed', 'no longer']
                    is_contradiction = any(signal in evidence_content for signal in contradiction_signals)
                    
                    if is_contradiction:
                        contradicting.append(evidence)
                    elif evidence not in supporting:
                        supporting.append(evidence)
        
        return supporting, contradicting

    def _should_decay(self, opinion_meta: OpinionMetadata, decay_after_days: int) -> bool:
        """Check if an opinion should decay due to staleness."""
        last_activity = opinion_meta.last_reinforced_at or opinion_meta.formed_at
        if not last_activity:
            return True
        
        days_since_activity = (datetime.now(timezone.utc) - last_activity).days
        return days_since_activity > decay_after_days

    def _update_opinion(self, memory, opinion: MemoryItem, meta: OpinionMetadata):
        """Update an opinion with new metadata."""
        try:
            opinion.metadata.update(meta.to_dict())
            if hasattr(memory, 'update'):
                memory.update(opinion)
        except Exception as e:
            _module_logger.error(f"Failed to update opinion: {e}")

    def _archive_opinion(self, memory, opinion: MemoryItem):
        """Archive an opinion (soft delete)."""
        try:
            opinion.metadata['archived'] = True
            opinion.metadata['archive_reason'] = 'confidence_below_threshold'
            opinion.metadata['archive_timestamp'] = datetime.now(timezone.utc).isoformat()

            if hasattr(memory, 'update'):
                memory.update(opinion)
        except Exception as e:
            _module_logger.error(f"Failed to archive opinion: {e}")

    # ── CORE-EVO-LIVE-1: Incremental evolution ────────────────────────────

    def evolve_incremental(self, ctx) -> list:
        """Handle new-evidence reinforcement path only.

        When a new episodic or semantic item is added, check if it contains
        evidence for or against existing opinions. Apply reinforcement or
        contradiction. Does NOT run decay or archive — those are time-based
        and run on idle catch-up.
        """
        from smartmemory.evolution.events import EvolutionAction

        new_item_props = ctx.get_item(ctx.event.item_id)
        if not new_item_props:
            return []

        new_content = (new_item_props.get("content") or "").lower()
        if not new_content:
            return []

        # Find all opinion nodes in the workspace
        opinions = ctx.search(memory_type="opinion")
        if not opinions:
            return []

        cfg = self.config
        actions: list = []

        for opinion in opinions:
            opinion_subject = (opinion.get("subject") or opinion.get("metadata", {}).get("subject", "")).lower()
            if not opinion_subject:
                continue

            # Check if new evidence mentions the opinion's subject
            if opinion_subject not in new_content:
                continue

            opinion_id = opinion.get("item_id", "")
            confidence = float(opinion.get("confidence", opinion.get("metadata", {}).get("confidence", 0.5)))

            # Detect contradiction signals
            contradiction_signals = ["not", "never", "stopped", "changed", "no longer", "unlike"]
            is_contradiction = any(signal in new_content for signal in contradiction_signals)

            if is_contradiction:
                new_confidence = max(0.0, confidence - cfg.contradiction_penalty)
            else:
                new_confidence = min(1.0, confidence + cfg.reinforcement_boost)

            if new_confidence != confidence:
                actions.append(EvolutionAction(
                    operation="update_property",
                    target_id=opinion_id,
                    properties={"confidence": new_confidence},
                ))

        return actions
