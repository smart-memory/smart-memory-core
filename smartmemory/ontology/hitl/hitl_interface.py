"""
Human-In-The-Loop (HITL) interfaces for ontology governance.

Provides user interfaces for humans to review violations, make decisions,
and interact with the intelligent ontology governance system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

from smartmemory.ontology.governance import (
    OntologyGovernor, OntologyViolation, GovernanceAction, ViolationSeverity
)


class HITLInterface:
    """Human-In-The-Loop interface for ontology governance."""

    def __init__(self, governor: OntologyGovernor):
        self.governor = governor
        self.logger = logging.getLogger(__name__)

    def display_violations_for_review(self, max_violations: int = 10) -> None:
        """Display violations in a human-readable format for review."""
        violations = self.governor.get_violations_for_review()

        if not violations:
            print("🎉 No violations found! Your ontology governance is working well.")
            return

        print("\n📋 ONTOLOGY GOVERNANCE REVIEW")
        print("=" * 60)
        print(f"Found {len(violations)} violations requiring human review")
        print(f"Showing top {min(max_violations, len(violations))} by priority\n")

        for i, violation in enumerate(violations[:max_violations], 1):
            self._display_violation(i, violation)
            print("-" * 60)

    def _display_violation(self, index: int, violation: OntologyViolation) -> None:
        """Display a single violation in human-readable format."""
        severity_emoji = {
            ViolationSeverity.CRITICAL: "🚨",
            ViolationSeverity.HIGH: "⚠️",
            ViolationSeverity.MEDIUM: "⚡",
            ViolationSeverity.LOW: "ℹ️"
        }

        print(f"{severity_emoji[violation.severity]} VIOLATION #{index}")
        print(f"ID: {violation.id}")
        print(f"Type: {violation.violation_type}")
        print(f"Severity: {violation.severity.value.upper()}")
        print(f"Confidence: {violation.confidence:.1%}")
        print(f"Auto-fixable: {'✅ Yes' if violation.auto_fixable else '❌ No'}")
        print("\nDescription:")
        print(f"  {violation.description}")
        print("\nSuggested Fix:")
        print(f"  {violation.suggested_fix}")

        if violation.data_context:
            print("\nContext:")
            for key, value in violation.data_context.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")

        print(f"\nDetected: {violation.detected_at.strftime('%Y-%m-%d %H:%M:%S')}")

    def prompt_for_decision(self, violation_id: str) -> Tuple[GovernanceAction, str]:
        """Prompt human for decision on a specific violation."""
        violation = self.governor.violations.get(violation_id)
        if not violation:
            raise ValueError(f"Violation {violation_id} not found")

        print(f"\n🤔 DECISION NEEDED for violation: {violation_id}")
        print(f"Description: {violation.description}")
        print(f"Suggested fix: {violation.suggested_fix}")

        print("\nAvailable actions:")
        actions = [
            ("1", GovernanceAction.APPROVE, "✅ Approve - Accept this as valid"),
            ("2", GovernanceAction.EVOLVE_ONTOLOGY, "🧬 Evolve Ontology - Update ontology to accommodate"),
            ("3", GovernanceAction.FIX_DATA, "🔧 Fix Data - Correct the data to match ontology"),
            ("4", GovernanceAction.REJECT, "❌ Reject - Remove or flag this data"),
            ("5", GovernanceAction.IGNORE, "🙈 Ignore - Mark as acceptable deviation"),
            ("6", GovernanceAction.REVIEW_LATER, "⏰ Review Later - Defer this decision")
        ]

        for key, action, description in actions:
            print(f"  {key}. {description}")

        while True:
            choice = input("\nEnter your choice (1-6): ").strip()

            action_map = {key: action for key, action, _ in actions}
            if choice in action_map:
                selected_action = action_map[choice]
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")

        rationale = input("Enter rationale for your decision: ").strip()
        if not rationale:
            rationale = f"Human decision: {selected_action.value}"

        return selected_action, rationale

    def interactive_review_session(self, max_violations: int = 5) -> Dict[str, Any]:
        """Run an interactive review session for violations."""
        print("\n🎯 INTERACTIVE ONTOLOGY GOVERNANCE SESSION")
        print("=" * 60)

        violations = self.governor.get_violations_for_review()[:max_violations]

        if not violations:
            print("🎉 No violations found! Your ontology governance is working well.")
            return {"violations_reviewed": 0, "decisions_made": 0}

        decisions_made = 0

        for i, violation in enumerate(violations, 1):
            print(f"\n📋 REVIEWING VIOLATION {i}/{len(violations)}")
            self._display_violation(i, violation)

            # Ask if user wants to make a decision
            while True:
                proceed = input("\nMake a decision on this violation? (y/n/q to quit): ").strip().lower()
                if proceed in ['y', 'yes']:
                    try:
                        action, rationale = self.prompt_for_decision(violation.id)
                        success = self.governor.apply_governance_decision(
                            violation.id, action, rationale, decided_by="human"
                        )

                        if success:
                            print(f"✅ Decision applied successfully: {action.value}")
                            decisions_made += 1
                        else:
                            print("❌ Failed to apply decision")

                    except Exception as e:
                        print(f"❌ Error applying decision: {e}")
                    break

                elif proceed in ['n', 'no']:
                    print("⏭️ Skipping this violation")
                    break

                elif proceed in ['q', 'quit']:
                    print("🛑 Ending review session")
                    return {
                        "violations_reviewed": i,
                        "decisions_made": decisions_made,
                        "session_ended_early": True
                    }
                else:
                    print("❌ Please enter 'y' for yes, 'n' for no, or 'q' to quit")

        print("\n🎉 Review session complete!")
        print(f"Violations reviewed: {len(violations)}")
        print(f"Decisions made: {decisions_made}")

        return {
            "violations_reviewed": len(violations),
            "decisions_made": decisions_made,
            "session_ended_early": False
        }

    def batch_decision_interface(self, violation_type: str = None,
                                 severity: ViolationSeverity = None) -> Dict[str, Any]:
        """Interface for making batch decisions on similar violations."""
        violations = self.governor.get_violations_for_review(severity_filter=severity)

        if violation_type:
            violations = [v for v in violations if v.violation_type == violation_type]

        if not violations:
            print("No matching violations found for batch processing.")
            return {"violations_processed": 0}

        print("\n📦 BATCH DECISION INTERFACE")
        print("=" * 60)
        print(f"Found {len(violations)} violations matching criteria")

        if violation_type:
            print(f"Type filter: {violation_type}")
        if severity:
            print(f"Severity filter: {severity.value}")

        # Show summary of violations
        print("\nViolation summary:")
        for i, violation in enumerate(violations[:5], 1):
            print(f"  {i}. {violation.description[:80]}...")

        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more")

        # Get batch decision
        print(f"\nChoose batch action for all {len(violations)} violations:")
        actions = [
            ("1", GovernanceAction.APPROVE, "✅ Approve all"),
            ("2", GovernanceAction.EVOLVE_ONTOLOGY, "🧬 Evolve ontology for all"),
            ("3", GovernanceAction.IGNORE, "🙈 Ignore all"),
            ("4", GovernanceAction.REVIEW_LATER, "⏰ Review all later"),
            ("5", None, "❌ Cancel batch operation")
        ]

        for key, action, description in actions:
            print(f"  {key}. {description}")

        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in ["1", "2", "3", "4"]:
                action_map = {
                    "1": GovernanceAction.APPROVE,
                    "2": GovernanceAction.EVOLVE_ONTOLOGY,
                    "3": GovernanceAction.IGNORE,
                    "4": GovernanceAction.REVIEW_LATER
                }
                selected_action = action_map[choice]
                break
            elif choice == "5":
                print("❌ Batch operation cancelled")
                return {"violations_processed": 0, "cancelled": True}
            else:
                print("❌ Invalid choice. Please enter 1-5.")

        rationale = input("Enter rationale for batch decision: ").strip()
        if not rationale:
            rationale = f"Batch decision: {selected_action.value}"

        # Apply batch decision
        processed = 0
        failed = 0

        print(f"\n⚙️ Applying batch decision to {len(violations)} violations...")

        for violation in violations:
            try:
                success = self.governor.apply_governance_decision(
                    violation.id, selected_action, rationale, decided_by="human"
                )
                if success:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"Failed to process violation {violation.id}: {e}")
                failed += 1

        print("✅ Batch processing complete!")
        print(f"Successfully processed: {processed}")
        print(f"Failed: {failed}")

        return {
            "violations_processed": processed,
            "violations_failed": failed,
            "action_taken": selected_action.value,
            "rationale": rationale
        }

    def show_governance_dashboard(self) -> None:
        """Display governance dashboard with summary statistics."""
        violations = list(self.governor.violations.values())
        decisions = list(self.governor.decisions.values())

        print("\n📊 ONTOLOGY GOVERNANCE DASHBOARD")
        print("=" * 60)

        # Violation summary
        print(f"📋 ACTIVE VIOLATIONS: {len(violations)}")
        if violations:
            severity_counts = {}
            type_counts = {}
            auto_fixable_count = 0

            for v in violations:
                severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1
                type_counts[v.violation_type] = type_counts.get(v.violation_type, 0) + 1
                if v.auto_fixable:
                    auto_fixable_count += 1

            print("  By severity:")
            for severity in ViolationSeverity:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    print(f"    {severity.value.upper()}: {count}")

            print("  By type:")
            for vtype, count in sorted(type_counts.items()):
                print(f"    {vtype}: {count}")

            print(f"  Auto-fixable: {auto_fixable_count}")

        # Decision summary
        print(f"\n⚖️ DECISIONS MADE: {len(decisions)}")
        if decisions:
            action_counts = {}
            human_decisions = 0

            for d in decisions:
                action_counts[d.action] = action_counts.get(d.action, 0) + 1
                if d.decided_by == "human":
                    human_decisions += 1

            print("  By action:")
            for action in GovernanceAction:
                count = action_counts.get(action, 0)
                if count > 0:
                    print(f"    {action.value}: {count}")

            print(f"  Human decisions: {human_decisions}")
            print(f"  Automated decisions: {len(decisions) - human_decisions}")

        # Governance history
        history_count = len(self.governor.governance_history)
        print(f"\n📚 GOVERNANCE HISTORY: {history_count} events")

        if history_count > 0:
            recent_events = self.governor.governance_history[-5:]
            print("  Recent events:")
            for event in recent_events:
                timestamp = event["timestamp"][:19]  # Remove microseconds
                action = event["decision"]["action"]
                decided_by = event["decision"]["decided_by"]
                print(f"    {timestamp}: {action} (by {decided_by})")

    def export_governance_report(self, filename: str = None) -> str:
        """Export governance report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ontology_governance_report_{timestamp}.json"

        report = {
            "generated_at": datetime.now().isoformat(),
            "active_violations": [v.to_dict() for v in self.governor.violations.values()],
            "decisions": [
                {
                    "violation_id": d.violation_id,
                    "action": d.action.value,
                    "rationale": d.rationale,
                    "decided_by": d.decided_by,
                    "decided_at": d.decided_at.isoformat(),
                    "metadata": d.metadata
                }
                for d in self.governor.decisions.values()
            ],
            "governance_history": self.governor.governance_history,
            "summary": {
                "total_violations": len(self.governor.violations),
                "total_decisions": len(self.governor.decisions),
                "total_history_events": len(self.governor.governance_history)
            }
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Governance report exported to: {filename}")
        return filename


class NotificationSystem:
    """Notification system for alerting humans about governance events."""

    def __init__(self, governor: OntologyGovernor):
        self.governor = governor
        self.logger = logging.getLogger(__name__)
        self.notification_handlers = []

    def add_notification_handler(self, handler):
        """Add a notification handler (email, Slack, etc.)."""
        self.notification_handlers.append(handler)

    def check_for_critical_violations(self) -> List[OntologyViolation]:
        """Check for critical violations that need immediate attention."""
        critical_violations = [
            v for v in self.governor.violations.values()
            if v.severity == ViolationSeverity.CRITICAL
        ]
        return critical_violations

    def send_daily_summary(self) -> None:
        """Send daily governance summary."""
        violations = list(self.governor.violations.values())

        if not violations:
            message = "🎉 Daily Ontology Governance Summary: No violations found!"
        else:
            critical_count = len([v for v in violations if v.severity == ViolationSeverity.CRITICAL])
            high_count = len([v for v in violations if v.severity == ViolationSeverity.HIGH])

            message = f"""
📊 Daily Ontology Governance Summary
Total violations: {len(violations)}
Critical: {critical_count}
High priority: {high_count}
Auto-fixable: {len([v for v in violations if v.auto_fixable])}

Review needed: {len(self.governor.get_violations_for_review())}
            """.strip()

        self._send_notification("Daily Governance Summary", message)

    def alert_critical_violation(self, violation: OntologyViolation) -> None:
        """Send immediate alert for critical violation."""
        message = f"""
🚨 CRITICAL ONTOLOGY VIOLATION DETECTED

ID: {violation.id}
Type: {violation.violation_type}
Description: {violation.description}
Confidence: {violation.confidence:.1%}

Immediate review required!
        """.strip()

        self._send_notification("CRITICAL Ontology Violation", message)

    def _send_notification(self, subject: str, message: str) -> None:
        """Send notification through all registered handlers."""
        for handler in self.notification_handlers:
            try:
                handler(subject, message)
            except Exception as e:
                self.logger.error(f"Failed to send notification via {handler}: {e}")

        # Fallback: log the notification
        self.logger.info(f"NOTIFICATION - {subject}: {message}")


# Example notification handlers
def console_notification_handler(subject: str, message: str) -> None:
    """Simple console notification handler."""
    print(f"\n🔔 NOTIFICATION: {subject}")
    print(message)
    print("-" * 40)


def email_notification_handler(subject: str, message: str) -> None:
    """Email notification handler (placeholder - would integrate with email service)."""
    # This would integrate with an actual email service
    print(f"📧 EMAIL NOTIFICATION: {subject}")
    print(f"Would send email with message: {message[:100]}...")


def slack_notification_handler(subject: str, message: str) -> None:
    """Slack notification handler (placeholder - would integrate with Slack API)."""
    # This would integrate with Slack API
    print(f"💬 SLACK NOTIFICATION: {subject}")
    print(f"Would send Slack message: {message[:100]}...")
