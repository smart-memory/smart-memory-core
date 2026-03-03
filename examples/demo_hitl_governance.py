"""
Demo of Human-In-The-Loop (HITL) Ontology Governance System

This demo shows how humans interact with the intelligent ontology governance
system to review violations, make decisions, and manage ontology evolution.
"""

import shutil
import tempfile
from datetime import datetime, UTC

from smartmemory.ontology.governance import (
    OntologyGovernor, OntologyViolation, ViolationSeverity, GovernanceAction
)
from smartmemory.ontology.hitl.hitl_interface import (
    HITLInterface, NotificationSystem, console_notification_handler
)
from smartmemory.ontology.llm_manager import LLMOntologyManager
from smartmemory.ontology.manager import (
    OntologyManager
)
from smartmemory.ontology.models import EntityTypeDefinition, RelationshipTypeDefinition
from smartmemory.stores.ontology import FileSystemOntologyStorage


# NO MOCKS - REAL SERVICES ONLY


def create_demo_environment():
    """Create demo environment with ontology and violations."""
    print("🔧 Setting up demo environment...")

    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    storage = FileSystemOntologyStorage(temp_dir)
    ontology_manager = OntologyManager(storage)
    llm_manager = LLMOntologyManager(ontology_manager)

    # Real SmartMemory - NO MOCKS
    from smartmemory.smart_memory import SmartMemory
    smart_memory = SmartMemory()

    # Create governance system
    governor = OntologyGovernor(ontology_manager, llm_manager, smart_memory)

    # Create test ontology
    ontology = ontology_manager.create_ontology(
        name="Software Engineering",
        domain="technology",
        description="Demo ontology for software engineering concepts"
    )

    # Add entity types
    framework_type = EntityTypeDefinition(
        name="framework",
        description="Software framework",
        properties={"name": "string", "language": "string", "version": "string"},
        required_properties={"name", "language"},
        parent_types=set(),
        aliases=set(),
        examples=["React", "Django", "Spring"],
        created_by="demo",
        created_at=datetime.now(UTC)
    )

    organization_type = EntityTypeDefinition(
        name="organization",
        description="Software organization",
        properties={"name": "string", "founded": "date", "headquarters": "string"},
        required_properties={"name"},
        parent_types=set(),
        aliases=set(),
        examples=["Google", "Microsoft", "Facebook"],
        created_by="demo",
        created_at=datetime.now(UTC)
    )

    ontology.add_entity_type(framework_type)
    ontology.add_entity_type(organization_type)

    # Add relationship type
    developed_by_rel = RelationshipTypeDefinition(
        name="developed_by",
        description="Framework developed by organization",
        source_types={"framework"},
        target_types={"organization"},
        properties={},
        created_by="demo",
        created_at=datetime.now(UTC)
    )

    ontology.add_relationship_type(developed_by_rel)
    storage.save_ontology(ontology)

    # Create sample violations for demo
    violations = [
        OntologyViolation(
            id="violation_1",
            item_id="item_1",
            ontology_id=ontology.id,
            violation_type="unknown_entity_type",
            severity=ViolationSeverity.HIGH,
            description="Unknown entity type 'programming_language' found in data",
            suggested_fix="Add 'programming_language' entity type to ontology",
            confidence=0.9,
            detected_at=datetime.now(UTC),
            data_context={
                "entity": {"name": "Python", "type": "programming_language"},
                "item_content": "Python is a high-level programming language"
            },
            auto_fixable=False
        ),
        OntologyViolation(
            id="violation_2",
            item_id="item_2",
            ontology_id=ontology.id,
            violation_type="missing_required_property",
            severity=ViolationSeverity.MEDIUM,
            description="Framework 'Vue.js' missing required property 'language'",
            suggested_fix="Add 'language' property with value 'JavaScript'",
            confidence=0.95,
            detected_at=datetime.now(UTC),
            data_context={
                "entity": {"name": "Vue.js", "type": "framework"},
                "missing_property": "language"
            },
            auto_fixable=True
        ),
        OntologyViolation(
            id="violation_3",
            item_id="item_3",
            ontology_id=ontology.id,
            violation_type="unknown_relationship_type",
            severity=ViolationSeverity.LOW,
            description="Unknown relationship type 'competes_with' found",
            suggested_fix="Add 'competes_with' relationship type or map to existing type",
            confidence=0.7,
            detected_at=datetime.now(UTC),
            data_context={
                "relationship": {"type": "competes_with", "source": "React", "target": "Vue.js"}
            },
            auto_fixable=False
        ),
        OntologyViolation(
            id="violation_4",
            item_id="item_4",
            ontology_id=ontology.id,
            violation_type="data_inconsistency",
            severity=ViolationSeverity.CRITICAL,
            description="Critical data inconsistency: organization 'Facebook' has conflicting founding dates",
            suggested_fix="Review and correct founding date information",
            confidence=0.85,
            detected_at=datetime.now(UTC),
            data_context={
                "conflicting_values": {"founded": ["2004", "2003"]},
                "entity": {"name": "Facebook", "type": "organization"}
            },
            auto_fixable=False
        )
    ]

    # Add violations to governor
    for violation in violations:
        governor.violations[violation.id] = violation

    print("✅ Demo environment ready!")
    print(f"   - Created ontology: {ontology.name}")
    print(f"   - Added {len(violations)} sample violations")

    return governor, temp_dir


def demo_basic_violation_display():
    """Demo: Basic violation display and review."""
    print("\n" + "=" * 80)
    print("📋 DEMO 1: Basic Violation Display and Review")
    print("=" * 80)

    governor, temp_dir = create_demo_environment()
    hitl = HITLInterface(governor)

    try:
        # Show governance dashboard
        hitl.show_governance_dashboard()

        # Display violations for review
        hitl.display_violations_for_review(max_violations=3)

        print("\n💡 This shows how violations are presented to humans in a readable format.")
        print("   Humans can see severity, confidence, context, and suggested fixes.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_programmatic_decisions():
    """Demo: Programmatic decision making (simulating human input)."""
    print("\n" + "=" * 80)
    print("⚖️ DEMO 2: Programmatic Decision Making")
    print("=" * 80)

    governor, temp_dir = create_demo_environment()
    hitl = HITLInterface(governor)

    try:
        # Show initial state
        print("📊 Initial state:")
        hitl.show_governance_dashboard()

        # Make some programmatic decisions (simulating human choices)
        violations = list(governor.violations.keys())

        print(f"\n🤖 Simulating human decisions on {len(violations)} violations...")

        # Decision 1: Approve the programming language violation
        if "violation_1" in violations:
            success = governor.apply_governance_decision(
                "violation_1",
                GovernanceAction.EVOLVE_ONTOLOGY,
                "Programming languages are a valid concept in our domain. Evolve ontology to include them.",
                decided_by="human"
            )
            print(f"✅ Decision 1: {'Success' if success else 'Failed'} - Evolve ontology for programming languages")

        # Decision 2: Auto-fix the missing property
        if "violation_2" in violations:
            success = governor.apply_governance_decision(
                "violation_2",
                GovernanceAction.FIX_DATA,
                "Vue.js is indeed a JavaScript framework. Fix the missing language property.",
                decided_by="human"
            )
            print(f"✅ Decision 2: {'Success' if success else 'Failed'} - Fix missing language property")

        # Decision 3: Ignore the low-priority relationship
        if "violation_3" in violations:
            success = governor.apply_governance_decision(
                "violation_3",
                GovernanceAction.IGNORE,
                "Competition relationships are not critical for our use case. Ignore for now.",
                decided_by="human"
            )
            print(f"✅ Decision 3: {'Success' if success else 'Failed'} - Ignore competition relationship")

        # Decision 4: Review critical issue later
        if "violation_4" in violations:
            success = governor.apply_governance_decision(
                "violation_4",
                GovernanceAction.REVIEW_LATER,
                "Critical data inconsistency needs deeper investigation. Schedule for detailed review.",
                decided_by="human"
            )
            print(f"✅ Decision 4: {'Success' if success else 'Failed'} - Schedule critical issue for later review")

        # Show final state
        print("\n📊 Final state after decisions:")
        hitl.show_governance_dashboard()

        print("\n💡 This shows how human decisions are applied and tracked.")
        print("   Each decision is recorded with rationale and attribution.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_batch_processing():
    """Demo: Batch processing of similar violations."""
    print("\n" + "=" * 80)
    print("📦 DEMO 3: Batch Processing of Similar Violations")
    print("=" * 80)

    governor, temp_dir = create_demo_environment()
    hitl = HITLInterface(governor)

    try:
        # Add more violations of the same type for batch processing
        batch_violations = []
        for i in range(3):
            violation = OntologyViolation(
                id=f"batch_violation_{i}",
                item_id=f"batch_item_{i}",
                ontology_id=list(governor.ontology_manager.list_ontologies())[0]['id'],
                violation_type="unknown_entity_type",
                severity=ViolationSeverity.MEDIUM,
                description=f"Unknown entity type 'database_{i}' found in data",
                suggested_fix=f"Add 'database_{i}' entity type to ontology",
                confidence=0.8,
                detected_at=datetime.now(UTC),
                data_context={"entity_type": f"database_{i}"},
                auto_fixable=False
            )
            batch_violations.append(violation)
            governor.violations[violation.id] = violation

        print(f"📊 Added {len(batch_violations)} similar violations for batch processing")

        # Show how batch processing would work (simulated)
        print("\n📦 Simulating batch processing for 'unknown_entity_type' violations...")

        unknown_entity_violations = [
            v for v in governor.violations.values()
            if v.violation_type == "unknown_entity_type"
        ]

        print(f"Found {len(unknown_entity_violations)} violations of type 'unknown_entity_type'")

        # Simulate batch decision
        batch_rationale = "All these unknown entity types are valid database concepts. Approve all for ontology evolution."
        processed = 0

        for violation in unknown_entity_violations:
            success = governor.apply_governance_decision(
                violation.id,
                GovernanceAction.EVOLVE_ONTOLOGY,
                batch_rationale,
                decided_by="human"
            )
            if success:
                processed += 1

        print(f"✅ Batch processed {processed} violations with action: EVOLVE_ONTOLOGY")

        # Show final state
        print("\n📊 State after batch processing:")
        hitl.show_governance_dashboard()

        print("\n💡 This shows how humans can efficiently handle similar violations in batches.")
        print("   Reduces repetitive decision-making for patterns of similar issues.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_notification_system():
    """Demo: Notification system for governance events."""
    print("\n" + "=" * 80)
    print("🔔 DEMO 4: Notification System")
    print("=" * 80)

    governor, temp_dir = create_demo_environment()

    try:
        # Create notification system
        notification_system = NotificationSystem(governor)
        notification_system.add_notification_handler(console_notification_handler)

        # Check for critical violations
        critical_violations = notification_system.check_for_critical_violations()
        print(f"🚨 Found {len(critical_violations)} critical violations")

        # Send alert for critical violation
        if critical_violations:
            notification_system.alert_critical_violation(critical_violations[0])

        # Send daily summary
        print("\n📊 Sending daily governance summary...")
        notification_system.send_daily_summary()

        print("\n💡 This shows how humans are notified about governance events.")
        print("   Critical violations trigger immediate alerts.")
        print("   Daily summaries keep humans informed of overall governance health.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_governance_reporting():
    """Demo: Governance reporting and export."""
    print("\n" + "=" * 80)
    print("📄 DEMO 5: Governance Reporting and Export")
    print("=" * 80)

    governor, temp_dir = create_demo_environment()
    hitl = HITLInterface(governor)

    try:
        # Make some decisions to create history
        violations = list(governor.violations.keys())[:2]
        for i, violation_id in enumerate(violations):
            governor.apply_governance_decision(
                violation_id,
                GovernanceAction.APPROVE if i % 2 == 0 else GovernanceAction.IGNORE,
                f"Demo decision {i + 1} for reporting purposes",
                decided_by="human"
            )

        # Show dashboard
        hitl.show_governance_dashboard()

        # Export governance report
        report_file = hitl.export_governance_report("demo_governance_report.json")

        print("\n💡 This shows how governance activity is tracked and reported.")
        print("   Reports include violations, decisions, and complete history.")
        print(f"   Exported to: {report_file}")

        # Clean up report file
        import os
        if os.path.exists(report_file):
            os.remove(report_file)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all HITL governance demos."""
    print("🎯 HUMAN-IN-THE-LOOP ONTOLOGY GOVERNANCE DEMO")
    print("=" * 80)
    print("This demo shows how humans interact with the intelligent ontology")
    print("governance system to review violations and make decisions.")
    print("=" * 80)

    # Run all demos
    demo_basic_violation_display()
    demo_programmatic_decisions()
    demo_batch_processing()
    demo_notification_system()
    demo_governance_reporting()

    print("\n" + "=" * 80)
    print("🎉 DEMO COMPLETE!")
    print("=" * 80)
    print("\n📋 SUMMARY OF HITL CAPABILITIES:")
    print("✅ Human-readable violation display with context and suggestions")
    print("✅ Interactive decision-making with multiple action types")
    print("✅ Batch processing for efficient handling of similar violations")
    print("✅ Notification system for critical alerts and daily summaries")
    print("✅ Comprehensive reporting and governance history tracking")
    print("✅ Programmatic API for integration with custom interfaces")

    print("\n🚀 INTEGRATION OPTIONS:")
    print("• Command-line interface (shown in this demo)")
    print("• Web dashboard (can be built using the HITLInterface)")
    print("• Email/Slack notifications (handlers provided)")
    print("• REST API endpoints (can wrap the governance methods)")
    print("• Jupyter notebook integration (for data science workflows)")

    print("\n💡 NEXT STEPS:")
    print("• Customize notification handlers for your environment")
    print("• Build web UI using the HITLInterface as backend")
    print("• Set up periodic governance jobs")
    print("• Integrate with your existing workflow tools")


if __name__ == "__main__":
    main()
