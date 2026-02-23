from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EvolverPlugin, PluginMetadata

if TYPE_CHECKING:
    from smartmemory.evolution.tracker import EvolutionTracker


@dataclass
class WorkingToProceduralConfig(MemoryBaseModel):
    k: int = 5  # minimum pattern count to promote
    track_evolution: bool = True  # Whether to track evolution events


@dataclass
class WorkingToProceduralRequest(StageRequest):
    k: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class WorkingToProceduralEvolver(EvolverPlugin):
    """
    Evolves repeated skill/tool patterns in working memory to procedural memory (macro creation).

    When configured with an evolution_tracker, records creation and refinement events
    for procedures promoted from working memory.
    """

    def __init__(
        self,
        config: Optional[WorkingToProceduralConfig] = None,
        evolution_tracker: Optional["EvolutionTracker"] = None,
    ):
        """Initialize the evolver.

        Args:
            config: Configuration for the evolver
            evolution_tracker: Optional tracker for recording evolution events
        """
        super().__init__()
        self.config = config or WorkingToProceduralConfig()
        self._evolution_tracker = evolution_tracker

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="working_to_procedural",
            version="1.1.0",
            author="SmartMemory Team",
            description="Promotes repeated skill patterns from working to procedural memory",
            plugin_type="evolver",
            dependencies=[],
            min_smartmemory_version="0.1.0",
        )

    def evolve(self, memory, logger=None):
        """Evolve working memory patterns into procedural memory.

        Detects repeated skill patterns in working memory and promotes them
        to procedural memory as macros. When an evolution_tracker is configured,
        records creation events for new procedures.

        Args:
            memory: SmartMemory instance
            logger: Optional logger for status messages
        """
        cfg = self.config
        if not hasattr(cfg, "k"):
            raise TypeError(
                "WorkingToProceduralEvolver requires a typed config with a 'k' attribute. "
                "Please provide WorkingToProceduralConfig or a compatible typed config."
            )

        k = int(cfg.k)
        memory_id = getattr(memory, 'item_id', None)
        with trace_span("pipeline.evolve.working_to_procedural", {"memory_id": memory_id, "k": k}):
            patterns = memory.working.detect_skill_patterns(min_count=k)

            # Get workspace context from memory's scope provider if available
            workspace_id = ""
            user_id = ""
            if hasattr(memory, "scope_provider") and memory.scope_provider:
                write_ctx = memory.scope_provider.get_write_context()
                workspace_id = write_ctx.get("workspace_id", "")
                user_id = write_ctx.get("user_id", "")

            for pattern in patterns:
                # Check if this pattern already exists as a procedure
                existing_procedure = self._find_existing_procedure(memory, pattern)

                if existing_procedure:
                    # Refinement case: update existing procedure
                    self._handle_refinement(
                        memory=memory,
                        existing_procedure=existing_procedure,
                        pattern=pattern,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        pattern_count=k,
                        logger=logger,
                    )
                else:
                    # Creation case: new procedure from working memory
                    self._handle_creation(
                        memory=memory,
                        pattern=pattern,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        pattern_count=k,
                        logger=logger,
                    )

    def _find_existing_procedure(self, memory, pattern) -> Optional[Dict[str, Any]]:
        """Find an existing procedure that matches the pattern.

        Args:
            memory: SmartMemory instance
            pattern: The pattern to search for

        Returns:
            The existing procedure item or None
        """
        try:
            # Search for similar procedures
            pattern_str = str(pattern)
            results = memory.procedural.search(pattern_str, top_k=1)
            if results:
                # Check similarity threshold (basic check - could be improved)
                result = results[0]
                existing_content = getattr(result, "content", "") or ""
                if existing_content and self._is_similar_pattern(existing_content, pattern_str):
                    return {
                        "item_id": getattr(result, "item_id", None),
                        "content": existing_content,
                        "metadata": getattr(result, "metadata", {}),
                    }
        except Exception:
            pass
        return None

    def _is_similar_pattern(self, existing: str, new: str, threshold: float = 0.8) -> bool:
        """Check if two patterns are similar enough to be considered the same.

        Args:
            existing: Existing procedure content
            new: New pattern content
            threshold: Similarity threshold (0-1)

        Returns:
            True if patterns are similar
        """
        # Simple word overlap check - could use embeddings for better matching
        existing_words = set(existing.lower().split())
        new_words = set(new.lower().split())

        if not existing_words or not new_words:
            return False

        intersection = existing_words & new_words
        union = existing_words | new_words

        jaccard = len(intersection) / len(union) if union else 0
        return jaccard >= threshold

    def _handle_creation(
        self,
        memory,
        pattern,
        workspace_id: str,
        user_id: str,
        pattern_count: int,
        logger=None,
    ):
        """Handle creation of a new procedure from a working memory pattern.

        Args:
            memory: SmartMemory instance
            pattern: The pattern being promoted
            workspace_id: Workspace ID for tenant isolation
            user_id: User ID
            pattern_count: Number of times the pattern was observed
            logger: Optional logger
        """
        # Create the procedural macro
        result = memory.procedural.add_macro(pattern)

        if logger:
            logger.info(f"Promoted working skill pattern to procedural: {pattern}")

        # Track evolution event if tracker is configured
        if self._evolution_tracker and self.config.track_evolution and result:
            try:
                procedure_id = self._extract_procedure_id(result, pattern)
                if procedure_id and workspace_id:
                    self._evolution_tracker.track_creation(
                        procedure_id=procedure_id,
                        content=str(pattern),
                        metadata=self._extract_pattern_metadata(pattern),
                        source={
                            "type": "working_memory",
                            "source_items": self._get_source_items(pattern),
                            "pattern_count": pattern_count,
                        },
                        workspace_id=workspace_id,
                        user_id=user_id,
                        confidence=0.0,  # Initial confidence is 0
                        match_stats=None,
                    )
                    if logger:
                        logger.debug(f"Tracked creation event for procedure {procedure_id}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to track creation event: {e}")

    def _handle_refinement(
        self,
        memory,
        existing_procedure: Dict[str, Any],
        pattern,
        workspace_id: str,
        user_id: str,
        pattern_count: int,
        logger=None,
    ):
        """Handle refinement of an existing procedure.

        Args:
            memory: SmartMemory instance
            existing_procedure: The existing procedure data
            pattern: The new pattern data
            workspace_id: Workspace ID
            user_id: User ID
            pattern_count: Number of times the pattern was observed
            logger: Optional logger
        """
        # For now, we don't actually update the procedure content
        # This would require procedural memory to support updates
        # Just track the refinement event

        procedure_id = existing_procedure.get("item_id")

        if logger:
            logger.info(f"Detected refinement pattern for procedure {procedure_id}")

        # Track refinement event if tracker is configured
        if self._evolution_tracker and self.config.track_evolution and procedure_id and workspace_id:
            try:
                self._evolution_tracker.track_refinement(
                    procedure_id=procedure_id,
                    new_content=str(pattern),
                    new_metadata=self._extract_pattern_metadata(pattern),
                    source={
                        "type": "working_memory",
                        "source_items": self._get_source_items(pattern),
                        "pattern_count": pattern_count,
                    },
                    workspace_id=workspace_id,
                    user_id=user_id,
                    confidence=0.0,
                    match_stats=None,
                )
                if logger:
                    logger.debug(f"Tracked refinement event for procedure {procedure_id}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to track refinement event: {e}")

    def _extract_procedure_id(self, result, pattern) -> Optional[str]:
        """Extract procedure ID from the add_macro result.

        Args:
            result: Result from add_macro call
            pattern: The pattern that was added

        Returns:
            The procedure ID or None
        """
        if isinstance(result, dict):
            return result.get("item_id")
        if hasattr(result, "item_id"):
            return result.item_id
        # Fallback: generate from pattern
        return None

    def _extract_pattern_metadata(self, pattern) -> Dict[str, Any]:
        """Extract metadata from a pattern.

        Args:
            pattern: The pattern to extract metadata from

        Returns:
            Metadata dict with name, description, skills, tools, steps
        """
        if isinstance(pattern, dict):
            return {
                "name": pattern.get("name", ""),
                "description": pattern.get("description", ""),
                "skills": pattern.get("skills", []),
                "tools": pattern.get("tools", []),
                "steps": pattern.get("steps", []),
            }
        # Default extraction for string patterns
        pattern_str = str(pattern)
        return {
            "name": pattern_str[:50] if len(pattern_str) > 50 else pattern_str,
            "description": "",
            "skills": [],
            "tools": [],
            "steps": [],
        }

    def _get_source_items(self, pattern) -> List[str]:
        """Get source item IDs from a pattern.

        Args:
            pattern: The pattern

        Returns:
            List of source item IDs
        """
        if isinstance(pattern, dict):
            return pattern.get("source_items", [])
        return []
