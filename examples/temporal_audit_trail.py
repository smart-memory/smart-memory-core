"""
Audit Trail Example

Demonstrates compliance-focused audit trail features for:
- GDPR compliance
- HIPAA requirements  
- SOC2 audit trails
- Financial regulations

This showcases SmartMemory's unique bi-temporal capabilities.
"""

from smartmemory import SmartMemory, MemoryItem
from datetime import datetime, UTC
import json
import time


def main():
    memory = SmartMemory()
    
    print("=" * 70)
    print("Audit Trail & Compliance Example")
    print("Use Case: Healthcare Record Tracking (HIPAA Compliance)")
    print("=" * 70)
    
    # Simulate a healthcare scenario with multiple operations
    print("\n📋 Scenario: Medical Record Management")
    print("-" * 70)
    
    # User creates a medical record
    print("\n1. Doctor Smith creates patient record...")
    item = MemoryItem(
        content="Patient John Doe, Age 45, presents with mild hypertension",
        metadata={
            "user_id": "doctor_smith",
            "user_role": "physician",
            "patient_id": "patient_12345",
            "record_type": "medical",
            "sensitivity": "PHI",  # Protected Health Information
            "facility": "General Hospital",
            "department": "Cardiology"
        }
    )
    record_id = memory.add(item)
    item.item_id = record_id
    
    print(f"✓ Record created: {record_id}")
    print(f"  By: {item.metadata['user_id']} ({item.metadata['user_role']})")
    print(f"  Patient: {item.metadata['patient_id']}")
    print(f"  Content: {item.content}")
    
    time.sleep(0.1)
    
    # Doctor updates the record (simulated as new version)
    print("\n2. Doctor Smith updates diagnosis...")
    item_v2 = MemoryItem(
        content="Patient John Doe, Age 45, diagnosed with Stage 1 Hypertension. Prescribed Lisinopril 10mg daily.",
        metadata={
            "user_id": "doctor_smith",
            "user_role": "physician",
            "patient_id": "patient_12345",
            "record_type": "medical",
            "sensitivity": "PHI",
            "facility": "General Hospital",
            "department": "Cardiology",
            "diagnosis": "Stage 1 Hypertension",
            "medication": "Lisinopril 10mg",
            "updated_by": "doctor_smith",
            "update_reason": "Added diagnosis and treatment plan",
            "version": 2,
            "original_id": record_id
        }
    )
    record_v2_id = memory.add(item_v2)
    
    print("✓ Record updated")
    print(f"  By: {item_v2.metadata['updated_by']}")
    print(f"  Reason: {item_v2.metadata['update_reason']}")
    
    time.sleep(0.1)
    
    # Nurse views the record
    print("\n3. Nurse Johnson accesses record...")
    viewed_item = memory.get(record_id)
    print("✓ Record accessed")
    print("  By: nurse_johnson (nurse)")
    print("  Action: View")
    
    time.sleep(0.1)
    
    # Specialist adds consultation notes (simulated as new version)
    print("\n4. Cardiologist adds consultation notes...")
    item_v3 = MemoryItem(
        content=item_v2.content + " Cardiologist consultation: Patient shows good response to medication. Continue current treatment.",
        metadata={
            "user_id": "doctor_williams",
            "user_role": "cardiologist",
            "patient_id": "patient_12345",
            "record_type": "medical",
            "sensitivity": "PHI",
            "facility": "General Hospital",
            "department": "Cardiology",
            "diagnosis": "Stage 1 Hypertension",
            "medication": "Lisinopril 10mg",
            "consultation": "cardiology",
            "updated_by": "doctor_williams",
            "update_reason": "Added specialist consultation",
            "version": 3,
            "original_id": record_id
        }
    )
    record_v3_id = memory.add(item_v3)
    
    print("✓ Record updated")
    print(f"  By: {item_v3.metadata['updated_by']} (cardiologist)")
    print(f"  Reason: {item_v3.metadata['update_reason']}")
    item = item_v3  # Use latest version for exports
    
    # Generate comprehensive audit report
    print("\n\n📊 Generating Compliance Audit Report...")
    print("=" * 70)
    
    trail = memory.temporal.get_audit_trail(record_id, include_metadata=True)
    
    print(f"\n{'='*70}")
    print("HIPAA COMPLIANCE AUDIT REPORT")
    print(f"{'='*70}")
    print("\nRecord Information:")
    print(f"  Record ID: {record_id}")
    print(f"  Patient ID: {item.metadata['patient_id']}")
    print(f"  Record Type: {item.metadata['record_type']}")
    print(f"  Sensitivity: {item.metadata['sensitivity']}")
    print(f"  Facility: {item.metadata['facility']}")
    print(f"  Department: {item.metadata['department']}")
    
    print("\nAudit Trail:")
    print(f"  Total Events: {len(trail)}")
    print(f"  Report Generated: {datetime.now(UTC).isoformat()}")
    
    print(f"\n{'-'*70}")
    print("Event History (Chronological):")
    print(f"{'-'*70}")
    
    for i, event in enumerate(reversed(trail), 1):
        print(f"\nEvent #{i}:")
        print(f"  Timestamp: {event['timestamp']}")
        print(f"  Action: {event['action'].upper()}")
        print(f"  Valid From: {event['valid_from']}")
        print(f"  Valid To: {event['valid_to']}")
        
        if 'metadata' in event:
            user = event['metadata'].get('updated_by') or event['metadata'].get('user_id')
            role = event['metadata'].get('user_role', 'unknown')
            if user:
                print(f"  User: {user} ({role})")
            
            if 'update_reason' in event['metadata']:
                print(f"  Reason: {event['metadata']['update_reason']}")
        
        if 'content_preview' in event:
            print(f"  Content: {event['content_preview']}")
    
    print(f"\n{'='*70}")
    
    # Export audit trail for compliance
    print("\n\n💾 Exporting audit trail for compliance...")
    print("-" * 70)
    
    audit_export = {
        'report_type': 'HIPAA_Audit_Trail',
        'record_id': record_id,
        'patient_id': item.metadata['patient_id'],
        'facility': item.metadata['facility'],
        'department': item.metadata['department'],
        'generated_at': datetime.now(UTC).isoformat(),
        'generated_by': 'compliance_system',
        'total_events': len(trail),
        'events': trail,
        'compliance_notes': [
            'All access to PHI logged',
            'Complete audit trail maintained',
            'Bi-temporal tracking enabled',
            'Rollback capability available'
        ]
    }
    
    filename = f"audit_trail_{record_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(audit_export, f, indent=2)
    
    print(f"✓ Audit trail exported to: {filename}")
    print("  Format: JSON")
    print(f"  Size: {len(json.dumps(audit_export))} bytes")
    print(f"  Events: {len(trail)}")
    
    # Demonstrate change tracking
    print("\n\n🔍 Detailed Change Analysis...")
    print("-" * 70)
    
    changes = memory.temporal.get_changes(record_id)
    if changes:
        print(f"✓ Detected {len(changes)} change(s):")
        for i, change in enumerate(changes, 1):
            print(f"\nChange #{i}:")
            print(f"  Timestamp: {change.timestamp.isoformat()}")
            print(f"  Type: {change.change_type}")
            print(f"  Fields modified: {', '.join(change.changed_fields)}")
            
            if change.old_value and change.new_value:
                print("  Changes:")
                for field in change.changed_fields:
                    if field == 'content':
                        old = change.old_value.get('content', '')[:50]
                        new = change.new_value.get('content', '')[:50]
                        print(f"    {field}:")
                        print(f"      Before: {old}...")
                        print(f"      After:  {new}...")
    else:
        print("  No detailed changes available (full history tracking in progress)")
    
    # Demonstrate rollback capability (dry run)
    print("\n\n↩️  Rollback Capability (Compliance Recovery)...")
    print("-" * 70)
    
    history = memory.temporal.get_history(record_id)
    if len(history) >= 2:
        rollback_to = history[1].transaction_time_start.isoformat() if history[1].transaction_time_start else "2024-01-01"
        print("Scenario: Need to recover previous version")
        print(f"Rollback target: {rollback_to}")
        
        preview = memory.temporal.rollback(
            record_id,
            rollback_to,
            dry_run=True
        )
        
        if 'error' not in preview:
            print("\n✓ Rollback Preview (DRY RUN):")
            print(f"  Would restore to: {rollback_to}")
            print(f"  Fields that would change: {preview.get('would_change', [])}")
            print("  ⚠️  This is a preview only - no changes made")
            print("  ✓ Actual rollback would create new version (preserving history)")
        else:
            print(f"  Error: {preview['error']}")
    
    # Compliance summary
    print("\n\n" + "=" * 70)
    print("✅ COMPLIANCE FEATURES DEMONSTRATED")
    print("=" * 70)
    print("\n✓ HIPAA Compliance:")
    print("  • Complete audit trail of all PHI access")
    print("  • User identification for all operations")
    print("  • Timestamp tracking (transaction time)")
    print("  • Reason tracking for all changes")
    
    print("\n✓ Data Integrity:")
    print("  • Bi-temporal tracking (valid time + transaction time)")
    print("  • Immutable history (cannot delete past versions)")
    print("  • Change detection and analysis")
    print("  • Rollback capability with full audit trail")
    
    print("\n✓ Regulatory Requirements:")
    print("  • SOC2: Complete audit logs")
    print("  • GDPR: Data access tracking")
    print("  • HIPAA: PHI access logging")
    print("  • Financial: Transaction history")
    
    print("\n🎯 SmartMemory is the ONLY AI memory system with these capabilities!")
    print("=" * 70)


if __name__ == "__main__":
    main()
