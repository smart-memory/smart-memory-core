#!/usr/bin/env python3
"""
Decision Memory System Example

Demonstrates the Decision Memory System in SmartMemory:
1. Creating decisions with confidence tracking
2. Reinforcing and contradicting decisions with evidence
3. Decision lifecycle (active → superseded → retracted)
4. Provenance tracking and conflict detection

Decision types:
- inference: Concluded from evidence (e.g., "User prefers TypeScript")
- preference: User preference detected
- classification: Categorization decision
- choice: Selection between options
- belief: Adopted belief/opinion
- policy: Rule to follow going forward

Run this example:
    PYTHONPATH=. python examples/decision_memory_example.py
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from smartmemory.smart_memory import SmartMemory
    from smartmemory.models.decision import Decision
    from smartmemory.decisions import DecisionManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("This example requires the full SmartMemory system to be available.")
    exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_basic_decision_creation():
    """Demonstrate basic decision creation and storage."""
    print_section("1. Basic Decision Creation")
    
    # Initialize SmartMemory and DecisionManager
    memory = SmartMemory()
    manager = DecisionManager(memory)
    
    # Create a simple inference decision
    decision = manager.create(
        content="The user prefers dark mode interfaces",
        decision_type="preference",
        confidence=0.8,
        source_type="inferred",
        domain="ui_preferences",
        tags=["dark-mode", "user-preference"],
    )
    
    print(f"✅ Created decision: {decision.decision_id}")
    print(f"   Content: {decision.content}")
    print(f"   Type: {decision.decision_type}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Status: {decision.status}")
    print(f"   Is Active: {decision.is_active}")
    
    return memory, manager, decision


def demo_confidence_tracking(memory, manager, decision):
    """Demonstrate confidence reinforcement and contradiction."""
    print_section("2. Confidence Tracking: Reinforce & Contradict")
    
    print(f"Initial confidence: {decision.confidence:.2f}")
    
    # Simulate finding supporting evidence
    print("\n📈 Reinforcing with supporting evidence...")
    for i in range(3):
        evidence_id = f"evidence_support_{i}"
        decision = manager.reinforce(decision.decision_id, evidence_id)
        print(f"   After reinforce #{i+1}: confidence = {decision.confidence:.3f}")
    
    print(f"\n   Reinforcement count: {decision.reinforcement_count}")
    print(f"   Evidence IDs: {decision.evidence_ids}")
    
    # Simulate finding contradicting evidence
    print("\n📉 Contradicting with opposing evidence...")
    contradiction_id = "evidence_contradiction_1"
    decision = manager.contradict(decision.decision_id, contradiction_id)
    print(f"   After contradiction: confidence = {decision.confidence:.3f}")
    print(f"   Contradiction count: {decision.contradiction_count}")
    print(f"   Contradicting IDs: {decision.contradicting_ids}")
    
    # Check stability
    print(f"\n📊 Decision stability: {decision.stability:.2f}")
    print(f"   Net reinforcement: {decision.net_reinforcement}")
    
    return decision


def demo_decision_lifecycle(memory, manager):
    """Demonstrate decision supersession and retraction."""
    print_section("3. Decision Lifecycle: Supersede & Retract")
    
    # Create an initial decision
    old_decision = manager.create(
        content="The project should use JavaScript for the frontend",
        decision_type="choice",
        confidence=0.75,
        domain="tech_stack",
    )
    print(f"Created initial decision: {old_decision.decision_id}")
    print(f"   '{old_decision.content}'")
    
    # Supersede with a new decision
    print("\n🔄 Superseding with updated decision...")
    new_decision = Decision(
        decision_id=Decision.generate_id(),
        content="The project should use TypeScript for the frontend",
        decision_type="choice",
        confidence=0.9,
        domain="tech_stack",
    )
    
    superseded = manager.supersede(
        old_decision_id=old_decision.decision_id,
        new_decision=new_decision,
        reason="TypeScript provides better type safety and tooling"
    )
    
    # Check the old decision status
    old_refreshed = manager.get_decision(old_decision.decision_id)
    print(f"   Old decision status: {old_refreshed.status}")
    print(f"   Old decision superseded by: {old_refreshed.superseded_by}")
    print(f"   New decision: '{superseded.content}'")
    print(f"   New decision confidence: {superseded.confidence:.2f}")
    
    # Demonstrate retraction
    print("\n🗑️ Retracting a decision...")
    retract_decision = manager.create(
        content="The database should be MongoDB",
        decision_type="choice",
        domain="tech_stack",
    )
    print(f"Created decision to retract: {retract_decision.decision_id}")
    
    manager.retract(retract_decision.decision_id, reason="Requirements changed to require SQL support")
    
    retracted = manager.get_decision(retract_decision.decision_id)
    print(f"   After retraction: status = {retracted.status}")
    print(f"   Retraction reason: {retracted.context_snapshot.get('retraction_reason')}")


def demo_decision_types():
    """Demonstrate different decision types."""
    print_section("4. Decision Types")
    
    memory = SmartMemory()
    manager = DecisionManager(memory)
    
    decision_examples = [
        {
            "content": "Based on usage patterns, the user is a backend developer",
            "decision_type": "inference",
            "description": "Concluded from evidence"
        },
        {
            "content": "User prefers detailed code explanations over brief summaries",
            "decision_type": "preference",
            "description": "User preference detected"
        },
        {
            "content": "This code issue is a performance bug, not a logic error",
            "decision_type": "classification",
            "description": "Categorization decision"
        },
        {
            "content": "Use pytest over unittest for this project's test suite",
            "decision_type": "choice",
            "description": "Selection between options"
        },
        {
            "content": "Functional programming leads to more maintainable code",
            "decision_type": "belief",
            "description": "Adopted belief/opinion"
        },
        {
            "content": "Always add type hints to new Python functions",
            "decision_type": "policy",
            "description": "Rule to follow going forward"
        },
    ]
    
    for example in decision_examples:
        decision = manager.create(
            content=example["content"],
            decision_type=example["decision_type"],
            confidence=0.8,
        )
        print(f"📋 {example['decision_type'].upper()} ({example['description']})")
        print(f"   → {decision.content}")
        print()


def demo_conflict_detection():
    """Demonstrate conflict detection between decisions."""
    print_section("5. Conflict Detection")
    
    memory = SmartMemory()
    manager = DecisionManager(memory)
    
    # Create some existing decisions
    decision1 = manager.create(
        content="The frontend framework should be React",
        decision_type="choice",
        domain="tech_stack",
        tags=["frontend", "framework"],
    )
    
    decision2 = manager.create(
        content="TypeScript should be used for all frontend code",
        decision_type="policy",
        domain="tech_stack",
        tags=["frontend", "language"],
    )
    
    print("Existing decisions:")
    print(f"   1. {decision1.content}")
    print(f"   2. {decision2.content}")
    
    # Create a potentially conflicting decision
    print("\n🔍 Checking for conflicts with new decision...")
    new_decision = Decision(
        decision_id=Decision.generate_id(),
        content="The frontend framework should be Vue.js",
        decision_type="choice",
        domain="tech_stack",
    )
    
    conflicts = manager.find_conflicts(new_decision)
    
    if conflicts:
        print(f"   ⚠️ Found {len(conflicts)} potential conflict(s):")
        for conflict in conflicts:
            print(f"      - {conflict.content}")
    else:
        print("   ✅ No conflicts detected")


def demo_pending_decisions():
    """Demonstrate pending decisions with requirements (residuation)."""
    print_section("6. Pending Decisions (Residuation)")
    
    from smartmemory.models.decision import PendingRequirement
    
    # Create a decision that's pending more information
    decision = Decision(
        decision_id=Decision.generate_id(),
        content="The API should use GraphQL",
        decision_type="choice",
        confidence=0.5,
        status="pending",
        domain="api_design",
        pending_requirements=[
            PendingRequirement(
                requirement_id="req_1",
                description="Need confirmation on client requirements",
                requirement_type="confirmation",
            ),
            PendingRequirement(
                requirement_id="req_2",
                description="Need performance benchmarks comparing REST vs GraphQL",
                requirement_type="evidence",
            ),
        ],
    )
    
    print(f"Created pending decision: {decision.decision_id}")
    print(f"   Content: {decision.content}")
    print(f"   Status: {decision.status}")
    print(f"   Is Pending: {decision.is_pending}")
    print(f"   Confidence: {decision.confidence:.2f} (low due to pending requirements)")
    
    print("\n📋 Pending Requirements:")
    for req in decision.pending_requirements:
        status = "✅" if req.resolved else "⏳"
        print(f"   {status} [{req.requirement_type}] {req.description}")
    
    print(f"\n   Has unresolved requirements: {decision.has_unresolved_requirements}")


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            SmartMemory Decision Memory System - Demo                 ║
║                                                                      ║
║  Decisions are first-class memory entities that capture what the    ║
║  system believes, with full provenance and confidence tracking.     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run demonstrations
        memory, manager, decision = demo_basic_decision_creation()
        demo_confidence_tracking(memory, manager, decision)
        demo_decision_lifecycle(memory, manager)
        demo_decision_types()
        demo_conflict_detection()
        demo_pending_decisions()
        
        print_section("Demo Complete!")
        print("""
🎉 You've seen the core features of SmartMemory's Decision Memory System!

Key concepts:
  • Decisions capture conclusions, inferences, and choices
  • Confidence tracking with reinforce/contradict pattern
  • Lifecycle management: active → superseded → retracted
  • Conflict detection between competing decisions
  • Pending decisions with residuation support

Next steps:
  1. Integrate DecisionManager into your agent workflow
  2. Use DecisionQueries for filtered retrieval
  3. Link decisions to ReasoningTraces for full provenance
        """)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
