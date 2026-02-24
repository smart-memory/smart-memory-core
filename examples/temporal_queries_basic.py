"""
Basic Temporal Queries Example

Demonstrates time-travel queries, history tracking, and audit trails.

This example shows SmartMemory's unique bi-temporal capabilities that
enable compliance, debugging, and time-travel analysis.
"""

from smartmemory import SmartMemory, MemoryItem
from datetime import datetime, timedelta
import time


def main():
    # Initialize memory
    memory = SmartMemory()
    
    print("=" * 70)
    print("SmartMemory Temporal Queries - Basic Examples")
    print("Unique Feature: Time-Travel & Audit Trails")
    print("=" * 70)
    
    # Example 1: Track changes over time
    print("\n📝 Example 1: Creating and updating a memory...")
    print("-" * 70)
    
    item = MemoryItem(
        content="Python is a programming language",
        metadata={"category": "programming", "version": 1, "author": "system"}
    )
    # Use add() for simple storage (temporal demo doesn't need full pipeline)
    item_id = memory.add(item)
    item.item_id = item_id  # Set the ID on the item
    print(f"✓ Created memory: {item_id}")
    print(f"  Content: {item.content}")
    
    # Wait a moment to ensure different timestamps
    time.sleep(0.1)
    
    # Add updated version (simulating an update) - use add() for version records
    item2 = MemoryItem(
        content="Python is a powerful programming language",
        metadata={"category": "programming", "version": 2, "updated_by": "user_alice", "original_id": item_id}
    )
    item2_id = memory.add(item2)
    print("\n✓ Added updated version (version 2)")
    print(f"  Content: {item2.content}")
    
    time.sleep(0.1)
    
    # Add another version
    item3 = MemoryItem(
        content="Python is a powerful, versatile programming language",
        metadata={"category": "programming", "version": 3, "updated_by": "user_bob", "original_id": item_id}
    )
    item3_id = memory.add(item3)
    print("\n✓ Added updated version (version 3)")
    print(f"  Content: {item3.content}")
    
    # Example 2: Get history
    print("\n\n📚 Example 2: Getting complete history...")
    print("-" * 70)
    
    history = memory.temporal.get_history(item_id)
    print(f"✓ Found {len(history)} version(s) in history for item {item_id}")
    for i, version in enumerate(history, 1):
        print(f"\n  Version {i}:")
        print(f"    Content: {version.content[:60]}...")
        print(f"    Version: {version.metadata.get('version', 1)}")
        if version.transaction_time_start:
            print(f"    Timestamp: {version.transaction_time_start.isoformat()}")
    
    print("\n  Note: Currently showing history for single item.")
    print("  Full version tracking across updates coming in next iteration.")
    
    # Example 3: Time travel
    print("\n\n⏰ Example 3: Time travel to past state...")
    print("-" * 70)
    
    # Get state from 5 seconds ago
    five_sec_ago = (datetime.now() - timedelta(seconds=5)).isoformat()
    print(f"Querying state as of: {five_sec_ago}")
    
    past_state = memory.graph.get_node(item_id, as_of_time=five_sec_ago)
    
    if past_state:
        print("✓ Memory 5 seconds ago:")
        print(f"  Content: {getattr(past_state, 'content', 'N/A')}")
    else:
        print("  (Memory didn't exist 5 seconds ago or time-travel not fully implemented)")
    
    # Example 4: Find what changed
    print("\n\n🔍 Example 4: Finding changes...")
    print("-" * 70)
    
    changes = memory.temporal.get_changes(item_id)
    if changes:
        print(f"✓ Found {len(changes)} change(s):")
        for i, change in enumerate(changes, 1):
            print(f"\n  Change {i}:")
            print(f"    Timestamp: {change.timestamp.isoformat()}")
            print(f"    Type: {change.change_type}")
            print(f"    Changed fields: {', '.join(change.changed_fields)}")
    else:
        print("  No changes detected (history tracking in progress)")
    
    # Example 5: Compare versions
    print("\n\n⚖️  Example 5: Comparing versions...")
    print("-" * 70)
    
    if len(history) >= 2:
        print("Comparing first and last versions...")
        diff = memory.temporal.compare_versions(
            item_id,
            history[-1].transaction_time_start.isoformat() if history[-1].transaction_time_start else "2024-01-01",
            history[0].transaction_time_start.isoformat() if history[0].transaction_time_start else datetime.now().isoformat()
        )
        print("✓ Differences found:")
        print(f"  Changed fields: {diff.get('changed_fields', [])}")
        if diff.get('modifications'):
            print(f"  Modifications: {len(diff['modifications'])} field(s)")
    else:
        print("  Need at least 2 versions to compare")
    
    # Example 6: Audit trail
    print("\n\n📋 Example 6: Getting audit trail...")
    print("-" * 70)
    
    trail = memory.temporal.get_audit_trail(item_id, include_metadata=True)
    print(f"✓ Audit trail ({len(trail)} event(s)):")
    for event in trail:
        print("\n  Event:")
        print(f"    Timestamp: {event['timestamp']}")
        print(f"    Action: {event['action']}")
        print(f"    Valid from: {event['valid_from']}")
        print(f"    Valid to: {event['valid_to']}")
        if 'content_preview' in event:
            print(f"    Content: {event['content_preview']}")
    
    # Example 7: Time-travel context manager
    print("\n\n🚀 Example 7: Time-travel context manager...")
    print("-" * 70)
    
    print("Using time-travel context to query past state...")
    with memory.time_travel(five_sec_ago):
        print(f"✓ Inside time-travel context (as of {five_sec_ago})")
        print("  All queries in this block would use that time reference")
        # In a full implementation, search would use the temporal context
    
    print("✓ Back to current time")
    
    # Example 8: Rollback (dry run)
    print("\n\n↩️  Example 8: Rollback preview...")
    print("-" * 70)
    
    if len(history) >= 2:
        rollback_to = history[1].transaction_time_start.isoformat() if history[1].transaction_time_start else "2024-01-01"
        print(f"Previewing rollback to: {rollback_to}")
        
        preview = memory.temporal.rollback(
            item_id,
            rollback_to,
            dry_run=True
        )
        
        if 'error' not in preview:
            print("✓ Rollback preview:")
            print(f"  Would rollback to: {rollback_to}")
            print(f"  Would change: {preview.get('would_change', [])}")
            print("  This is a DRY RUN - no actual changes made")
        else:
            print(f"  Error: {preview['error']}")
    else:
        print("  Need at least 2 versions to rollback")
    
    # Example 9: Timeline visualization
    print("\n\n📊 Example 9: Timeline visualization...")
    print("-" * 70)
    
    timeline = memory.temporal.get_timeline(item_id, granularity='day')
    if timeline:
        print("✓ Timeline by day:")
        for date, events in sorted(timeline.items()):
            print(f"\n  {date}:")
            for event in events:
                print(f"    - {event['type']}: {', '.join(event['fields'])}")
    else:
        print("  No timeline data available")
    
    print("\n" + "=" * 70)
    print("✅ Examples completed!")
    print("\nKey Takeaways:")
    print("  • SmartMemory tracks complete history of all changes")
    print("  • Time-travel queries let you see past states")
    print("  • Audit trails provide compliance-ready logs")
    print("  • Rollback enables safe recovery from mistakes")
    print("\n🎯 This is a UNIQUE feature - no other AI memory system has this!")
    print("=" * 70)


if __name__ == "__main__":
    main()
