"""
Unit tests for Assertion Challenger.
"""

import pytest


pytestmark = pytest.mark.unit
from unittest.mock import Mock

from smartmemory.reasoning.challenger import (
    AssertionChallenger,
    ChallengeResult,
    Conflict,
    ConflictType,
    ResolutionStrategy,
    should_challenge,
)
from smartmemory.models.memory_item import MemoryItem


@pytest.fixture
def mock_smart_memory():
    """Create a mock SmartMemory instance."""
    sm = Mock()
    sm.search = Mock(return_value=[])
    sm.get = Mock(return_value=None)
    sm.update = Mock()
    return sm


@pytest.fixture
def challenger(mock_smart_memory):
    """Create an AssertionChallenger with mocked SmartMemory."""
    return AssertionChallenger(
        mock_smart_memory,
        use_llm=False  # Use heuristics for testing
    )


class TestAssertionChallenger:
    """Test AssertionChallenger functionality."""
    
    def test_initialization(self, mock_smart_memory):
        """Test challenger initialization."""
        challenger = AssertionChallenger(mock_smart_memory)
        
        assert challenger.sm == mock_smart_memory
        assert challenger.similarity_threshold == 0.6
        assert challenger.max_related_facts == 10
        assert challenger.use_llm is True
    
    def test_challenge_no_related_facts(self, challenger, mock_smart_memory):
        """Test challenging when no related facts exist."""
        mock_smart_memory.search.return_value = []
        
        result = challenger.challenge("The sky is blue")
        
        assert isinstance(result, ChallengeResult)
        assert result.has_conflicts is False
        assert len(result.conflicts) == 0
        assert result.overall_confidence == 1.0
    
    def test_challenge_with_related_but_no_conflict(self, challenger, mock_smart_memory):
        """Test challenging with related facts but no contradiction."""
        existing_fact = MemoryItem(
            item_id="fact_1",
            content="Paris is the capital of France",
            memory_type="semantic"
        )
        mock_smart_memory.search.return_value = [existing_fact]
        
        result = challenger.challenge("France is a country in Europe")
        
        assert result.has_conflicts is False
        assert len(result.related_facts) == 1
    
    def test_detect_direct_negation(self, challenger, mock_smart_memory):
        """Test detection of direct negation contradictions."""
        existing_fact = MemoryItem(
            item_id="fact_1",
            content="Python is a programming language",
            memory_type="semantic"
        )
        mock_smart_memory.search.return_value = [existing_fact]
        
        result = challenger.challenge("Python is not a programming language")
        
        assert result.has_conflicts is True
        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == ConflictType.DIRECT_CONTRADICTION
    
    def test_detect_numeric_mismatch(self, challenger, mock_smart_memory):
        """Test detection of numeric value conflicts."""
        existing_fact = MemoryItem(
            item_id="fact_1",
            content="The population of Tokyo is 14 million",
            memory_type="semantic"
        )
        mock_smart_memory.search.return_value = [existing_fact]
        
        result = challenger.challenge("The population of Tokyo is 37 million")
        
        assert result.has_conflicts is True
        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == ConflictType.NUMERIC_MISMATCH
    
    def test_confidence_calculation_no_conflicts(self, challenger):
        """Test confidence is 1.0 when no conflicts."""
        confidence = challenger._calculate_confidence([])
        assert confidence == 1.0
    
    def test_confidence_calculation_with_conflicts(self, challenger, mock_smart_memory):
        """Test confidence decreases with conflicts."""
        existing_fact = MemoryItem(
            item_id="fact_1",
            content="The answer is 42",
            memory_type="semantic"
        )
        
        conflict = Conflict(
            existing_item=existing_fact,
            existing_fact="The answer is 42",
            new_fact="The answer is 24",
            conflict_type=ConflictType.NUMERIC_MISMATCH,
            confidence=0.8,
            explanation="Numbers differ",
            suggested_resolution=ResolutionStrategy.DEFER
        )
        
        confidence = challenger._calculate_confidence([conflict])
        assert confidence < 1.0
    
    def test_apply_confidence_decay(self, challenger, mock_smart_memory):
        """Test applying confidence decay to a fact."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Some fact",
            memory_type="semantic",
            metadata={"confidence": 1.0}
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.apply_confidence_decay("fact_1", decay_factor=0.2)
        
        assert result is True
        assert existing_item.metadata["confidence"] == 0.8
        assert existing_item.metadata["challenged"] is True
        mock_smart_memory.update.assert_called_once()
    
    def test_apply_confidence_decay_item_not_found(self, challenger, mock_smart_memory):
        """Test confidence decay when item doesn't exist."""
        mock_smart_memory.get.return_value = None
        
        result = challenger.apply_confidence_decay("nonexistent")
        
        assert result is False


class TestConflict:
    """Test Conflict dataclass."""
    
    def test_conflict_to_dict(self):
        """Test conflict serialization."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Existing fact",
            memory_type="semantic"
        )
        
        conflict = Conflict(
            existing_item=existing_item,
            existing_fact="Existing fact",
            new_fact="New fact",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.9,
            explanation="Direct contradiction detected",
            suggested_resolution=ResolutionStrategy.DEFER
        )
        
        result = conflict.to_dict()
        
        assert result["existing_item_id"] == "fact_1"
        assert result["conflict_type"] == "direct_contradiction"
        assert result["confidence"] == 0.9
        assert result["suggested_resolution"] == "defer"


class TestChallengeResult:
    """Test ChallengeResult dataclass."""
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = ChallengeResult(
            new_assertion="Test assertion",
            has_conflicts=False,
            conflicts=[],
            related_facts=[],
            overall_confidence=1.0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["new_assertion"] == "Test assertion"
        assert result_dict["has_conflicts"] is False
        assert result_dict["overall_confidence"] == 1.0


class TestResolutionStrategies:
    """Test conflict resolution strategies."""
    
    def test_resolve_keep_existing(self, challenger, mock_smart_memory):
        """Test KEEP_EXISTING resolution."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Existing fact",
            memory_type="semantic",
            metadata={}
        )
        
        conflict = Conflict(
            existing_item=existing_item,
            existing_fact="Existing fact",
            new_fact="New fact",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.9,
            explanation="Test",
            suggested_resolution=ResolutionStrategy.KEEP_EXISTING
        )
        
        result = challenger.resolve_conflict(conflict)
        
        assert result["strategy"] == "keep_existing"
        assert "Rejected new assertion" in result["actions_taken"]
    
    def test_resolve_defer(self, challenger, mock_smart_memory):
        """Test DEFER resolution flags for review."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Existing fact",
            memory_type="semantic",
            metadata={}
        )
        
        conflict = Conflict(
            existing_item=existing_item,
            existing_fact="Existing fact",
            new_fact="New fact",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.9,
            explanation="Test",
            suggested_resolution=ResolutionStrategy.DEFER
        )
        
        result = challenger.resolve_conflict(conflict)
        
        assert result["strategy"] == "defer"
        assert existing_item.metadata.get("needs_review") is True
        mock_smart_memory.update.assert_called()


class TestHeuristicDetection:
    """Test heuristic-based contradiction detection."""
    
    def test_negation_is_not(self, challenger, mock_smart_memory):
        """Test 'is not' vs 'is' detection."""
        existing = MemoryItem(
            item_id="f1",
            content="Water is wet",
            memory_type="semantic"
        )
        mock_smart_memory.search.return_value = [existing]
        
        result = challenger.challenge("Water is not wet")
        
        assert result.has_conflicts is True
    
    def test_negation_cannot(self, challenger, mock_smart_memory):
        """Test 'cannot' vs 'can' detection."""
        existing = MemoryItem(
            item_id="f1",
            content="Birds can fly",
            memory_type="semantic"
        )
        mock_smart_memory.search.return_value = [existing]
        
        result = challenger.challenge("Birds cannot fly")
        
        assert result.has_conflicts is True
    
    def test_no_false_positive_different_subjects(self, challenger, mock_smart_memory):
        """Test that different subjects don't trigger false positives."""
        existing = MemoryItem(
            item_id="f1",
            content="Cats are mammals",
            memory_type="semantic"
        )
        mock_smart_memory.search.return_value = [existing]
        
        result = challenger.challenge("Dogs are mammals")
        
        # Should not conflict - different subjects, same predicate
        assert result.has_conflicts is False


class TestSmartTriggering:
    """Test should_challenge() smart triggering logic."""
    
    def test_factual_claim_is_the(self):
        """Test 'X is the Y' pattern triggers challenge."""
        assert should_challenge("Paris is the capital of France") is True
    
    def test_factual_claim_numeric(self):
        """Test numeric claims trigger challenge."""
        assert should_challenge("The population is 14 million") is True
    
    def test_factual_claim_capital_of(self):
        """Test 'capital of' pattern triggers challenge."""
        assert should_challenge("The capital of Germany is Berlin") is True
    
    def test_factual_claim_absolutes(self):
        """Test absolute claims trigger challenge."""
        assert should_challenge("All mammals are warm-blooded") is True
        assert should_challenge("Birds never swim") is True
    
    def test_skip_opinions(self):
        """Test opinions are skipped."""
        assert should_challenge("I think Python is great") is False
        assert should_challenge("I believe this is correct") is False
    
    def test_skip_questions(self):
        """Test questions are skipped."""
        assert should_challenge("What is the capital of France?") is False
    
    def test_skip_greetings(self):
        """Test greetings are skipped."""
        assert should_challenge("Hello, how are you?") is False
        assert should_challenge("Thanks for your help") is False
    
    def test_skip_temporal_personal(self):
        """Test temporal/personal statements are skipped."""
        assert should_challenge("I'm going to the store today") is False
        assert should_challenge("Currently working on a project") is False
    
    def test_skip_short_content(self):
        """Test very short content is skipped."""
        assert should_challenge("Hello") is False
        assert should_challenge("Yes") is False
    
    def test_non_semantic_memory_type(self):
        """Test non-semantic memory types are skipped."""
        assert should_challenge("Paris is the capital", memory_type="episodic") is False
        assert should_challenge("Paris is the capital", memory_type="working") is False
        assert should_challenge("Paris is the capital", memory_type="procedural") is False
    
    def test_semantic_memory_type(self):
        """Test semantic memory type is challenged."""
        assert should_challenge("Paris is the capital of France", memory_type="semantic") is True
    
    def test_metadata_skip_flag(self):
        """Test skip_challenge metadata flag."""
        assert should_challenge(
            "Paris is the capital of France",
            metadata={"skip_challenge": True}
        ) is False
    
    def test_metadata_force_flag(self):
        """Test force_challenge metadata flag."""
        assert should_challenge(
            "Hello there",  # Would normally be skipped
            metadata={"force_challenge": True}
        ) is True
    
    def test_metadata_trusted_source(self):
        """Test trusted_source metadata flag."""
        assert should_challenge(
            "Paris is the capital of France",
            metadata={"trusted_source": True}
        ) is False
    
    def test_untrusted_source(self):
        """Test untrusted sources trigger challenge."""
        assert should_challenge(
            "Some longer content here that is not obviously factual",
            source="llm_generated"
        ) is True
        assert should_challenge(
            "Some longer content here that is not obviously factual",
            source="user_input"
        ) is True


class TestConfidenceDecay:
    """Test confidence decay tracking."""
    
    def test_apply_confidence_decay_basic(self, challenger, mock_smart_memory):
        """Test basic confidence decay application."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Paris is the capital of France",
            memory_type="semantic",
            metadata={"confidence": 1.0}
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.apply_confidence_decay("fact_1", decay_factor=0.2)
        
        assert result is True
        assert existing_item.metadata["confidence"] == 0.8
        assert existing_item.metadata["challenged"] is True
        assert existing_item.metadata["challenge_count"] == 1
        assert "last_challenged_at" in existing_item.metadata
        mock_smart_memory.update.assert_called_once_with(existing_item)
    
    def test_apply_confidence_decay_with_reason(self, challenger, mock_smart_memory):
        """Test confidence decay with reason tracking."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Paris is the capital of France",
            memory_type="semantic",
            metadata={"confidence": 0.9}
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.apply_confidence_decay(
            "fact_1", 
            decay_factor=0.1,
            reason="auto_resolved:wikipedia",
            conflicting_fact="Paris is the capital of Germany"
        )
        
        assert result is True
        assert existing_item.metadata["confidence"] == 0.8
        assert "confidence_history" in existing_item.metadata
        assert len(existing_item.metadata["confidence_history"]) == 1
        
        event = existing_item.metadata["confidence_history"][0]
        assert event["old_confidence"] == 0.9
        assert event["new_confidence"] == 0.8
        assert event["reason"] == "auto_resolved:wikipedia"
        assert event["conflicting_fact"] == "Paris is the capital of Germany"
    
    def test_apply_confidence_decay_cumulative(self, challenger, mock_smart_memory):
        """Test cumulative confidence decay."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Paris is the capital of France",
            memory_type="semantic",
            metadata={
                "confidence": 0.7,
                "challenged": True,
                "challenge_count": 2,
                "confidence_history": [
                    {"timestamp": "2024-01-01T00:00:00Z", "old_confidence": 1.0, "new_confidence": 0.9},
                    {"timestamp": "2024-01-02T00:00:00Z", "old_confidence": 0.9, "new_confidence": 0.7}
                ]
            }
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.apply_confidence_decay("fact_1", decay_factor=0.2)
        
        assert result is True
        assert abs(existing_item.metadata["confidence"] - 0.5) < 0.001  # Float tolerance
        assert existing_item.metadata["challenge_count"] == 3
        assert len(existing_item.metadata["confidence_history"]) == 3
    
    def test_apply_confidence_decay_floor(self, challenger, mock_smart_memory):
        """Test confidence doesn't go below 0."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Some fact",
            memory_type="semantic",
            metadata={"confidence": 0.1}
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.apply_confidence_decay("fact_1", decay_factor=0.5)
        
        assert result is True
        assert existing_item.metadata["confidence"] == 0.0
    
    def test_apply_confidence_decay_item_not_found(self, challenger, mock_smart_memory):
        """Test decay when item doesn't exist."""
        mock_smart_memory.get.return_value = None
        
        result = challenger.apply_confidence_decay("nonexistent")
        
        assert result is False
    
    def test_apply_confidence_decay_history_limit(self, challenger, mock_smart_memory):
        """Test confidence history is limited to 20 events."""
        # Create item with 19 history events
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Some fact",
            memory_type="semantic",
            metadata={
                "confidence": 0.5,
                "challenge_count": 19,
                "confidence_history": [
                    {"timestamp": f"2024-01-{i:02d}T00:00:00Z", "old_confidence": 1.0 - i*0.02, "new_confidence": 1.0 - (i+1)*0.02}
                    for i in range(19)
                ]
            }
        )
        mock_smart_memory.get.return_value = existing_item
        
        # Apply decay twice more
        challenger.apply_confidence_decay("fact_1", decay_factor=0.1)
        challenger.apply_confidence_decay("fact_1", decay_factor=0.1)
        
        # Should be capped at 20
        assert len(existing_item.metadata["confidence_history"]) == 20
    
    def test_get_confidence_history(self, challenger, mock_smart_memory):
        """Test retrieving confidence history."""
        history = [
            {"timestamp": "2024-01-01T00:00:00Z", "old_confidence": 1.0, "new_confidence": 0.9},
            {"timestamp": "2024-01-02T00:00:00Z", "old_confidence": 0.9, "new_confidence": 0.7}
        ]
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Some fact",
            memory_type="semantic",
            metadata={"confidence_history": history}
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.get_confidence_history("fact_1")
        
        assert result == history
    
    def test_get_confidence_history_empty(self, challenger, mock_smart_memory):
        """Test retrieving empty confidence history."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Some fact",
            memory_type="semantic",
            metadata={}
        )
        mock_smart_memory.get.return_value = existing_item
        
        result = challenger.get_confidence_history("fact_1")
        
        assert result == []
    
    def test_get_confidence_history_not_found(self, challenger, mock_smart_memory):
        """Test retrieving history for nonexistent item."""
        mock_smart_memory.get.return_value = None
        
        result = challenger.get_confidence_history("nonexistent")
        
        assert result == []
    
    def test_get_low_confidence_items(self, challenger, mock_smart_memory):
        """Test getting low confidence items."""
        items = [
            MemoryItem(item_id="1", content="Fact 1", memory_type="semantic", metadata={"confidence": 0.3}),
            MemoryItem(item_id="2", content="Fact 2", memory_type="semantic", metadata={"confidence": 0.8}),
            MemoryItem(item_id="3", content="Fact 3", memory_type="semantic", metadata={"confidence": 0.2}),
            MemoryItem(item_id="4", content="Fact 4", memory_type="semantic", metadata={}),  # Default 1.0
        ]
        mock_smart_memory.search.return_value = items
        
        result = challenger.get_low_confidence_items(threshold=0.5)
        
        assert len(result) == 2
        # Should be sorted by confidence (lowest first)
        assert result[0].item_id == "3"  # 0.2
        assert result[1].item_id == "1"  # 0.3


class TestAutoResolve:
    """Test auto-resolution functionality."""
    
    def test_auto_resolve_via_recency(self, challenger, mock_smart_memory):
        """Test auto-resolution via recency heuristic."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="The CEO of Twitter is Jack Dorsey",
            memory_type="semantic",
            metadata={"timestamp": "2020-01-01T00:00:00Z"}  # Need timestamp for recency check
        )
        mock_smart_memory.get.return_value = existing_item
        
        conflict = Conflict(
            existing_item=existing_item,
            existing_fact="The CEO of Twitter is Jack Dorsey",
            new_fact="The CEO of Twitter is now Elon Musk",
            conflict_type=ConflictType.TEMPORAL_CONFLICT,
            confidence=0.8,
            explanation="Leadership change",
            suggested_resolution=ResolutionStrategy.DEFER
        )
        
        result = challenger.auto_resolve(conflict, use_wikipedia=False, use_llm=False)
        
        # Should resolve via recency (new fact has "now")
        assert result["auto_resolved"] is True
        assert result["method"] == "recency"
        assert result["resolution"] == ResolutionStrategy.ACCEPT_NEW
    
    def test_auto_resolve_via_grounding(self, challenger, mock_smart_memory):
        """Test auto-resolution via grounding check."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="Paris is the capital of France",
            memory_type="semantic",
            metadata={
                "grounded_to": "https://en.wikipedia.org/wiki/Paris",  # Correct field
                "provenance": {"wikipedia_id": "Paris"}
            }
        )
        mock_smart_memory.get.return_value = existing_item
        
        conflict = Conflict(
            existing_item=existing_item,
            existing_fact="Paris is the capital of France",
            new_fact="Lyon is the capital of France",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.8,
            explanation="Different capitals",
            suggested_resolution=ResolutionStrategy.DEFER
        )
        
        result = challenger.auto_resolve(conflict, use_wikipedia=False, use_llm=False)
        
        # Should resolve via grounding (existing has grounded_to)
        assert result["auto_resolved"] is True
        assert result["method"] == "grounding"
        assert result["resolution"] == ResolutionStrategy.KEEP_EXISTING
    
    def test_auto_resolve_defers_when_no_method_works(self, challenger, mock_smart_memory):
        """Test auto-resolution defers when no method can resolve."""
        existing_item = MemoryItem(
            item_id="fact_1",
            content="The best programming language is Python",
            memory_type="semantic",
            metadata={}
        )
        mock_smart_memory.get.return_value = existing_item
        
        conflict = Conflict(
            existing_item=existing_item,
            existing_fact="The best programming language is Python",
            new_fact="The best programming language is Rust",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.7,
            explanation="Subjective claim",
            suggested_resolution=ResolutionStrategy.DEFER
        )
        
        result = challenger.auto_resolve(conflict, use_wikipedia=False, use_llm=False)
        
        # Should defer - subjective claims can't be auto-resolved
        assert result["auto_resolved"] is False
        # No resolution set when deferring
        assert result["resolution"] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
