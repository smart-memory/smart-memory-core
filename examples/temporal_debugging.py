"""
Debugging with Temporal Queries

Demonstrates using time-travel for debugging AI behavior.

Use Case: Understanding why an AI agent gave a specific answer
by examining what it knew at that point in time.
"""

from smartmemory import SmartMemory, MemoryItem
from datetime import datetime, timedelta
import time


def simulate_ai_learning(memory):
    """Simulate an AI learning facts over time."""
    
    print("\n🤖 Simulating AI Learning Process...")
    print("-" * 70)
    
    timestamps = {}
    
    # Day 1: AI learns basic fact
    print("\n📅 Day 1 (2 days ago): Initial learning")
    day1 = datetime.now() - timedelta(days=2)
    
    item1 = MemoryItem(
        content="Python was created by Guido van Rossum",
        metadata={
            "learned_at": day1.isoformat(),
            "confidence": 0.9,
            "source": "documentation",
            "topic": "python_history"
        }
    )
    item1_id = memory.add(item1)
    item1.item_id = item1_id
    timestamps['day1'] = day1
    print(f"✓ Learned: {item1.content}")
    print(f"  Confidence: {item1.metadata['confidence']}")
    print(f"  Source: {item1.metadata['source']}")
    
    time.sleep(0.1)
    
    # Day 2: AI learns more details
    print("\n📅 Day 2 (1 day ago): Additional learning")
    day2 = datetime.now() - timedelta(days=1)
    
    item2 = MemoryItem(
        content="Python was first released in 1991",
        metadata={
            "learned_at": day2.isoformat(),
            "confidence": 0.95,
            "source": "wikipedia",
            "topic": "python_history"
        }
    )
    item2_id = memory.add(item2)
    item2.item_id = item2_id
    timestamps['day2'] = day2
    print(f"✓ Learned: {item2.content}")
    print(f"  Confidence: {item2.metadata['confidence']}")
    
    time.sleep(0.1)
    
    # Day 3: AI corrects understanding
    print("\n📅 Day 3 (today): Correction based on new information")
    day3 = datetime.now()
    
    item1_v2 = MemoryItem(
        content="Python was created by Guido van Rossum in 1989, first released in 1991",
        metadata={
            "learned_at": day1.isoformat(),
            "confidence": 1.0,
            "source": "official_docs",
            "topic": "python_history",
            "corrected_at": day3.isoformat(),
            "correction_reason": "Found more accurate source with creation vs release dates",
            "version": 2,
            "original_id": item1_id
        }
    )
    item1_v2_id = memory.add(item1_v2)
    timestamps['day3'] = day3
    print(f"✓ Corrected: {item1_v2.content}")
    print(f"  Confidence: {item1_v2.metadata['confidence']}")
    print(f"  Reason: {item1_v2.metadata['correction_reason']}")
    
    return {
        'item1_id': item1_id,
        'item2_id': item2_id,
        'timestamps': timestamps
    }


def main():
    memory = SmartMemory()
    
    print("=" * 70)
    print("Debugging AI Behavior with Temporal Queries")
    print("Use Case: Understanding AI Knowledge Evolution")
    print("=" * 70)
    
    # Simulate AI learning over time
    learning_data = simulate_ai_learning(memory)
    item1_id = learning_data['item1_id']
    timestamps = learning_data['timestamps']
    
    # Debugging Scenario
    print("\n\n🐛 DEBUGGING SCENARIO")
    print("=" * 70)
    print("\nProblem: AI gave answer 'Python created in 1989' on Day 2")
    print("Question: Was this correct based on what it knew then?")
    print("\nLet's investigate using temporal queries...")
    
    # Debug 1: What did AI know on Day 2?
    print("\n\n1️⃣  What did the AI know on Day 2?")
    print("-" * 70)
    
    day2_time = timestamps['day2'].isoformat()
    print(f"Querying knowledge as of: {day2_time}")
    
    with memory.time_travel(day2_time):
        print("\n✓ Time-traveled to Day 2")
        print("  In a full implementation, search would return Day 2 state")
        print("  For now, we can query specific items:")
        
        # Get the state of item1 on Day 2
        day2_state = memory.graph.get_node(item1_id, as_of_time=day2_time)
        if day2_state:
            print("\n  Knowledge about Python creator on Day 2:")
            print(f"    Content: {getattr(day2_state, 'content', 'N/A')}")
            print(f"    Confidence: {getattr(day2_state, 'metadata', {}).get('confidence', 'N/A')}")
        else:
            print("  (Temporal query returned no result - feature in development)")
    
    print("\n✓ Back to current time")
    
    # Debug 2: When did the correction happen?
    print("\n\n2️⃣  When did the AI correct its understanding?")
    print("-" * 70)
    
    changes = memory.temporal.get_changes(item1_id)
    if changes:
        print(f"✓ Found {len(changes)} change(s) to this memory:")
        for i, change in enumerate(changes, 1):
            print(f"\n  Change #{i}:")
            print(f"    Timestamp: {change.timestamp.isoformat()}")
            print(f"    Type: {change.change_type}")
            print(f"    Fields changed: {', '.join(change.changed_fields)}")
            
            if 'content' in change.changed_fields and change.old_value and change.new_value:
                print("\n    Content Evolution:")
                print(f"      Before: {change.old_value.get('content', 'N/A')[:60]}...")
                print(f"      After:  {change.new_value.get('content', 'N/A')[:60]}...")
            
            if 'metadata.confidence' in change.changed_fields:
                old_conf = change.old_value.get('metadata', {}).get('confidence')
                new_conf = change.new_value.get('metadata', {}).get('confidence')
                print("\n    Confidence Change:")
                print(f"      Before: {old_conf}")
                print(f"      After:  {new_conf}")
    else:
        print("  No detailed changes available (history tracking in progress)")
    
    # Debug 3: Compare knowledge before and after
    print("\n\n3️⃣  Compare knowledge: Day 2 vs Day 3")
    print("-" * 70)
    
    day3_time = timestamps['day3'].isoformat()
    diff = memory.temporal.compare_versions(
        item1_id,
        day2_time,
        day3_time
    )
    
    if 'error' not in diff:
        print("✓ Comparison complete:")
        print(f"  Changed fields: {diff.get('changed_fields', [])}")
        
        if diff.get('modifications'):
            print("\n  Detailed modifications:")
            for field, change in diff['modifications'].items():
                print(f"\n    {field}:")
                if isinstance(change, dict) and 'old' in change and 'new' in change:
                    old_val = str(change['old'])[:60]
                    new_val = str(change['new'])[:60]
                    print(f"      Old: {old_val}...")
                    print(f"      New: {new_val}...")
    else:
        print(f"  Error: {diff['error']}")
    
    # Debug 4: Timeline visualization
    print("\n\n4️⃣  Knowledge Evolution Timeline")
    print("-" * 70)
    
    timeline = memory.temporal.get_timeline(item1_id, granularity='day')
    if timeline:
        print("✓ Timeline of changes:")
        for date, events in sorted(timeline.items()):
            print(f"\n  📅 {date}:")
            for event in events:
                print(f"      • {event['type'].upper()}: {', '.join(event['fields'])}")
                print(f"        at {event['timestamp']}")
    else:
        print("  Timeline data not available")
    
    # Debug 5: Complete history
    print("\n\n5️⃣  Complete Version History")
    print("-" * 70)
    
    history = memory.temporal.get_history(item1_id)
    print(f"✓ Found {len(history)} version(s):")
    
    for i, version in enumerate(reversed(history), 1):
        print(f"\n  Version {i}:")
        print(f"    Content: {version.content}")
        print(f"    Confidence: {version.metadata.get('confidence', 'N/A')}")
        if version.transaction_time_start:
            print(f"    Recorded: {version.transaction_time_start.isoformat()}")
        if 'corrected_at' in version.metadata:
            print(f"    Corrected: {version.metadata['corrected_at']}")
            print(f"    Reason: {version.metadata.get('correction_reason', 'N/A')}")
    
    # Debug 6: Audit trail for debugging
    print("\n\n6️⃣  Audit Trail (Debugging Perspective)")
    print("-" * 70)
    
    trail = memory.temporal.get_audit_trail(item1_id)
    print(f"✓ Audit trail ({len(trail)} events):")
    
    for event in reversed(trail):
        print(f"\n  {event['timestamp']}:")
        print(f"    Action: {event['action']}")
        print(f"    Valid: {event['valid_from']} → {event['valid_to']}")
    
    # Debugging Conclusion
    print("\n\n" + "=" * 70)
    print("🎯 DEBUGGING CONCLUSIONS")
    print("=" * 70)
    
    print("\n✓ Root Cause Analysis:")
    print("  • On Day 2, AI only knew 'created by Guido van Rossum'")
    print("  • It didn't yet know the creation date (1989)")
    print("  • Answer was incomplete but not wrong given knowledge at that time")
    print("  • Day 3 correction added creation date (1989) and release date (1991)")
    
    print("\n✓ Knowledge Evolution:")
    print("  • Day 1: Basic fact (creator name)")
    print("  • Day 2: Additional fact (release year)")
    print("  • Day 3: Refined understanding (creation vs release)")
    
    print("\n✓ Confidence Progression:")
    print("  • Day 1: 0.9 (good confidence)")
    print("  • Day 2: 0.95 (higher confidence)")
    print("  • Day 3: 1.0 (full confidence after correction)")
    
    print("\n✓ Debugging Benefits:")
    print("  • Understand AI behavior at any point in time")
    print("  • Track knowledge evolution")
    print("  • Identify when corrections were made")
    print("  • Analyze confidence changes")
    print("  • Reproduce historical states for testing")
    
    print("\n" + "=" * 70)
    print("💡 Use Cases for Temporal Debugging:")
    print("=" * 70)
    print("\n1. AI Hallucination Analysis")
    print("   → Check what the AI knew when it gave a wrong answer")
    
    print("\n2. A/B Testing")
    print("   → Compare AI performance before/after knowledge updates")
    
    print("\n3. Regression Testing")
    print("   → Ensure new knowledge doesn't break existing capabilities")
    
    print("\n4. Knowledge Quality Assurance")
    print("   → Track confidence scores over time")
    
    print("\n5. Explainable AI")
    print("   → Show users what the AI knew when making decisions")
    
    print("\n🎯 SmartMemory's temporal queries make AI debugging possible!")
    print("=" * 70)


if __name__ == "__main__":
    main()
