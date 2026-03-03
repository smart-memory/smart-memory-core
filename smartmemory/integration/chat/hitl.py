"""
Maya HITL Integration for Ontology Governance

This module integrates the HITL ontology governance system with Maya (AI assistant)
to provide natural language prompts and conversational governance workflows.
Maya asks natural questions to guide users through governance decisions.
"""

import logging
import warnings
from datetime import datetime, UTC
from typing import List, Optional, Tuple

from smartmemory.ontology.governance import (
    OntologyGovernor, OntologyViolation, GovernanceAction, ViolationSeverity
)
from smartmemory.ontology.hitl.hitl_interface import HITLInterface

# Deprecation notice: this module contains Maya-specific conversational UX and
# will be removed from the SmartMemory library in favor of a generic governance
# facade (GovernanceManager) that applications should consume from their own
# UX layers or services.
warnings.warn(
    "smartmemory.integration.chat.hitl (MayaHITLAssistant) is deprecated. "
    "Use smartmemory.ontology.governance_manager.GovernanceManager from your "
    "service/app layer, and host conversational UX within the application.",
    DeprecationWarning,
    stacklevel=2,
)


class MayaHITLAssistant:
    """Maya AI assistant for conversational ontology governance."""

    def __init__(self, governor: OntologyGovernor):
        self.governor = governor
        self.hitl = HITLInterface(governor)
        self.logger = logging.getLogger(__name__)

        # Maya's personality and conversation state
        self.conversation_context = {}
        self.pending_decisions = {}

    def maya_greeting(self) -> str:
        """Maya's greeting when starting governance review."""
        violations = list(self.governor.violations.values())

        if not violations:
            return """
🌟 Hi! I'm Maya, your ontology governance assistant. 

Great news - I don't see any ontology violations that need your attention right now! 
Your data quality looks excellent. 

Would you like me to:
• Run a fresh analysis to check for new issues?
• Show you the governance dashboard?
• Export a governance report?

Just let me know how I can help! ✨
            """.strip()

        critical_count = len([v for v in violations if v.severity == ViolationSeverity.CRITICAL])
        high_count = len([v for v in violations if v.severity == ViolationSeverity.HIGH])

        greeting = f"""
🌟 Hi! I'm Maya, your ontology governance assistant.

I've found {len(violations)} ontology issues that could use your input:
"""

        if critical_count > 0:
            greeting += f"• 🚨 {critical_count} critical issues (need immediate attention)\n"
        if high_count > 0:
            greeting += f"• ⚠️ {high_count} high-priority issues\n"

        auto_fixable = len([v for v in violations if v.auto_fixable])
        if auto_fixable > 0:
            greeting += f"• 🔧 {auto_fixable} issues I can fix automatically\n"

        greeting += """
Don't worry - I'll walk you through each one with clear explanations and suggestions.
Most of these are opportunities to improve your ontology, not problems!

Would you like to start reviewing them together? I'll make it easy! 😊
        """.strip()

        return greeting

    def maya_explain_violation(self, violation: OntologyViolation) -> str:
        """Maya explains a violation in natural, conversational language."""

        # Start with context-appropriate greeting
        severity_intro = {
            ViolationSeverity.CRITICAL: "🚨 This is a critical issue that needs immediate attention:",
            ViolationSeverity.HIGH: "⚠️ Here's an important issue I found:",
            ViolationSeverity.MEDIUM: "⚡ I noticed something that could be improved:",
            ViolationSeverity.LOW: "ℹ️ Here's a minor inconsistency I spotted:"
        }

        explanation = severity_intro[violation.severity] + "\n\n"

        # Explain the violation in natural language
        if violation.violation_type == "unknown_entity_type":
            entity_name = violation.data_context.get('entity') or {}.get('name', 'unknown')
            entity_type = violation.data_context.get('entity') or {}.get('type', 'unknown')

            explanation += f"""
I found an entity called "{entity_name}" that's labeled as type "{entity_type}", 
but this type isn't defined in your current ontology.

This often happens when your data contains new concepts that your ontology 
hasn't caught up with yet - which is totally normal as your knowledge grows!
            """.strip()

        elif violation.violation_type == "missing_required_property":
            entity_name = violation.data_context.get('entity') or {}.get('name', 'unknown')
            missing_prop = violation.data_context.get('missing_property', 'unknown')

            explanation += f"""
The entity "{entity_name}" is missing a required property called "{missing_prop}".

Your ontology expects all entities of this type to have this property, 
but this one doesn't. This might be incomplete data or a data entry issue.
            """.strip()

        elif violation.violation_type == "unknown_relationship_type":
            rel_type = violation.data_context.get('relationship') or {}.get('type', 'unknown')

            explanation += f"""
I found a relationship of type "{rel_type}" that isn't defined in your ontology.

This suggests there's a new way entities are connected that your ontology 
doesn't know about yet. This could be a valuable addition!
            """.strip()

        elif violation.violation_type == "data_inconsistency":
            explanation += """
I found some conflicting information in your data that doesn't match up.

This usually means the same entity has different values for the same property 
in different places, which can cause confusion.
            """.strip()

        else:
            explanation += f"""
I found an issue of type "{violation.violation_type}".

{violation.description}
            """.strip()

        # Add confidence and auto-fix info
        explanation += f"\n\nI'm {violation.confidence:.0%} confident about this assessment."

        if violation.auto_fixable:
            explanation += "\n\n🔧 Good news: I can fix this automatically if you'd like!"

        return explanation

    def maya_suggest_actions(self, violation: OntologyViolation) -> str:
        """Maya suggests actions in conversational language."""

        suggestions = "\n💡 Here are your options:\n\n"

        # Tailor suggestions based on violation type
        if violation.violation_type == "unknown_entity_type":
            suggestions += """
1. **Accept and evolve** - Add this new entity type to your ontology
   (This is usually the best choice when you're discovering new concepts)

2. **Map to existing type** - Maybe this is similar to a type you already have?

3. **Fix the data** - If this entity was mislabeled, I can correct it

4. **Ignore for now** - Keep the data as-is if this exception is okay
            """.strip()

        elif violation.violation_type == "missing_required_property":
            suggestions += """
1. **Add the missing property** - I can fill in a reasonable default value
   (Often the best choice for incomplete data)

2. **Make the property optional** - Update your ontology if this property isn't always needed

3. **Fix the data source** - If this indicates a data quality issue upstream

4. **Remove this entity** - If it's incomplete and shouldn't be kept
            """.strip()

        elif violation.violation_type == "unknown_relationship_type":
            suggestions += """
1. **Add this relationship type** - Expand your ontology to include this connection
   (Great for discovering new ways entities relate)

2. **Map to existing relationship** - Maybe this is similar to one you already have?

3. **Ignore this relationship** - Keep the entities but skip this connection

4. **Review later** - Take time to think about whether this relationship is valuable
            """.strip()

        else:
            suggestions += """
1. **Accept this as valid** - Mark this as an acceptable exception

2. **Evolve your ontology** - Update your rules to accommodate this case

3. **Fix the data** - Correct the data to match your current ontology

4. **Remove or flag** - If this data shouldn't be kept

5. **Ignore** - Keep as-is if this deviation is okay

6. **Review later** - Take more time to decide
            """.strip()

        suggestions += f"\n\n**My suggestion:** {violation.suggested_fix}"

        return suggestions

    def maya_ask_for_decision(self, violation: OntologyViolation) -> str:
        """Maya asks for a decision in natural language."""

        question = self.maya_explain_violation(violation)
        question += "\n\n" + self.maya_suggest_actions(violation)

        question += """

What would you like to do? You can:
• Tell me the number of your choice (1, 2, 3, etc.)
• Describe what you want in your own words
• Ask me questions if you need clarification

I'm here to help make this easy! 😊
        """.strip()

        return question

    def maya_interpret_response(self, user_response: str, violation: OntologyViolation) -> Tuple[Optional[GovernanceAction], str]:
        """Maya interprets user's natural language response."""

        response_lower = user_response.lower().strip()

        # Direct number responses
        if response_lower in ['1', 'one', 'first', 'option 1']:
            if violation.violation_type == "unknown_entity_type":
                return GovernanceAction.EVOLVE_ONTOLOGY, "Add new entity type to ontology"
            elif violation.violation_type == "missing_required_property":
                return GovernanceAction.FIX_DATA, "Add missing property with default value"
            elif violation.violation_type == "unknown_relationship_type":
                return GovernanceAction.EVOLVE_ONTOLOGY, "Add new relationship type to ontology"
            else:
                return GovernanceAction.APPROVE, "Accept as valid exception"

        elif response_lower in ['2', 'two', 'second', 'option 2']:
            if violation.violation_type == "unknown_entity_type":
                return GovernanceAction.FIX_DATA, "Map to existing entity type"
            elif violation.violation_type == "missing_required_property":
                return GovernanceAction.EVOLVE_ONTOLOGY, "Make property optional in ontology"
            elif violation.violation_type == "unknown_relationship_type":
                return GovernanceAction.FIX_DATA, "Map to existing relationship type"
            else:
                return GovernanceAction.EVOLVE_ONTOLOGY, "Update ontology rules"

        # Natural language interpretation
        elif any(word in response_lower for word in ['accept', 'approve', 'okay', 'fine', 'valid', 'good']):
            return GovernanceAction.APPROVE, "User approved this as valid"

        elif any(word in response_lower for word in ['evolve', 'add', 'expand', 'update ontology', 'new type']):
            return GovernanceAction.EVOLVE_ONTOLOGY, "User wants to evolve ontology"

        elif any(word in response_lower for word in ['fix', 'correct', 'repair', 'change data']):
            return GovernanceAction.FIX_DATA, "User wants to fix the data"

        elif any(word in response_lower for word in ['remove', 'delete', 'reject', 'bad data']):
            return GovernanceAction.REJECT, "User wants to remove this data"

        elif any(word in response_lower for word in ['ignore', 'skip', 'leave alone', 'keep as is']):
            return GovernanceAction.IGNORE, "User wants to ignore this violation"

        elif any(word in response_lower for word in ['later', 'think', 'decide later', 'not sure']):
            return GovernanceAction.REVIEW_LATER, "User wants to review this later"

        # If we can't interpret, return None to ask for clarification
        return None, ""

    def maya_confirm_decision(self, action: GovernanceAction, rationale: str, violation: OntologyViolation) -> str:
        """Maya confirms the decision before applying it."""

        action_descriptions = {
            GovernanceAction.APPROVE: "accept this as valid",
            GovernanceAction.EVOLVE_ONTOLOGY: "update your ontology to accommodate this",
            GovernanceAction.FIX_DATA: "fix the data to match your ontology",
            GovernanceAction.REJECT: "remove or flag this data",
            GovernanceAction.IGNORE: "ignore this violation",
            GovernanceAction.REVIEW_LATER: "save this for later review"
        }

        confirmation = f"""
✅ Got it! I'll {action_descriptions[action]}.

**Reasoning:** {rationale}

Is this correct? 
• Say "yes" or "confirm" to proceed
• Say "no" or "change" to choose differently
• Ask me to explain more if you're unsure
        """.strip()

        return confirmation

    def maya_decision_applied(self, action: GovernanceAction, violation: OntologyViolation) -> str:
        """Maya confirms that the decision was applied."""

        success_messages = {
            GovernanceAction.APPROVE: "✅ Perfect! I've marked this as an approved exception.",
            GovernanceAction.EVOLVE_ONTOLOGY: "🧬 Excellent! I've updated your ontology to include this new concept.",
            GovernanceAction.FIX_DATA: "🔧 Done! I've corrected the data to match your ontology.",
            GovernanceAction.REJECT: "🗑️ Removed! I've flagged this data as problematic.",
            GovernanceAction.IGNORE: "🙈 Noted! I'll ignore this type of violation going forward.",
            GovernanceAction.REVIEW_LATER: "⏰ Saved! I'll remind you about this later."
        }

        message = success_messages.get(action, "✅ Decision applied successfully!")

        remaining_violations = len(self.governor.violations)
        if remaining_violations > 0:
            message += f"\n\n{remaining_violations} more issues to review. Ready for the next one? 😊"
        else:
            message += "\n\n🎉 That was the last issue! Your ontology governance is all caught up. Great work!"

        return message

    def maya_ask_clarification(self, user_response: str) -> str:
        """Maya asks for clarification when she can't understand the response."""

        return f"""
I'm not quite sure what you meant by "{user_response}". 

Could you help me understand? You can:
• Choose a number (1, 2, 3, etc.) from the options I listed
• Use words like "accept", "fix", "evolve ontology", "ignore", or "review later"
• Ask me to explain any of the options in more detail

What would you like to do with this issue? 🤔
        """.strip()

    def maya_batch_suggestion(self, similar_violations: List[OntologyViolation]) -> str:
        """Maya suggests batch processing for similar violations."""

        if len(similar_violations) < 3:
            return ""  # Not worth batching

        violation_type = similar_violations[0].violation_type

        suggestion = f"""
💡 Hey! I noticed you have {len(similar_violations)} similar issues of type "{violation_type}".

Would you like me to apply the same decision to all of them at once? 
This could save you time if they're all the same kind of issue.

• Say "yes" or "batch" to handle them together
• Say "no" or "individual" to review each one separately

What do you prefer? 😊
        """.strip()

        return suggestion

    def maya_governance_summary(self) -> str:
        """Maya provides a friendly governance summary."""

        violations = list(self.governor.violations.values())
        decisions = list(self.governor.decisions.values())

        if not violations and not decisions:
            return """
🌟 Your ontology governance is looking great! 

No violations found and no pending decisions. Your data quality 
and ontology are in excellent shape. Keep up the good work! ✨
            """.strip()

        summary = "📊 **Ontology Governance Summary**\n\n"

        if violations:
            critical = len([v for v in violations if v.severity == ViolationSeverity.CRITICAL])
            high = len([v for v in violations if v.severity == ViolationSeverity.HIGH])
            medium = len([v for v in violations if v.severity == ViolationSeverity.MEDIUM])
            low = len([v for v in violations if v.severity == ViolationSeverity.LOW])

            summary += f"**Active Issues:** {len(violations)} total\n"
            if critical > 0:
                summary += f"• 🚨 {critical} critical (need immediate attention)\n"
            if high > 0:
                summary += f"• ⚠️ {high} high priority\n"
            if medium > 0:
                summary += f"• ⚡ {medium} medium priority\n"
            if low > 0:
                summary += f"• ℹ️ {low} low priority\n"

            auto_fixable = len([v for v in violations if v.auto_fixable])
            if auto_fixable > 0:
                summary += f"• 🔧 {auto_fixable} can be fixed automatically\n"

        if decisions:
            today_decisions = len([
                d for d in decisions
                if d.decided_at.date() == datetime.now(UTC).date()
            ])

            summary += "\n**Recent Activity:**\n"
            summary += f"• {len(decisions)} total decisions made\n"
            if today_decisions > 0:
                summary += f"• {today_decisions} decisions made today\n"

        if violations:
            summary += "\nWould you like to start reviewing the issues together? I'm here to help! 😊"
        else:
            summary += "\n🎉 All issues resolved! Your ontology governance is up to date."

        return summary

    def maya_help_message(self) -> str:
        """Maya's help message explaining what she can do."""

        return """
🌟 **Hi! I'm Maya, your ontology governance assistant.**

I help you manage data quality and ontology evolution through friendly conversation.
Here's what I can do for you:

**🔍 Review Issues Together**
• Explain ontology violations in plain English
• Suggest the best actions to take
• Walk you through decisions step by step

**⚡ Make It Easy**
• Handle similar issues in batches
• Apply automatic fixes when I'm confident
• Remember your preferences for similar situations

**📊 Keep You Informed**
• Show governance summaries and dashboards
• Send alerts for critical issues
• Track all decisions and changes

**💬 Natural Conversation**
• Just talk to me normally - no technical jargon required
• Ask questions anytime if you're unsure
• I'll explain everything clearly

**Getting Started:**
• Say "review" to start looking at issues
• Say "summary" for a governance overview
• Say "help" anytime you need guidance

What would you like to do? 😊
        """.strip()


def create_maya_conversation_flow(governor: OntologyGovernor) -> str:
    """Create a complete Maya conversation flow for governance."""

    maya = MayaHITLAssistant(governor)

    # Start with Maya's greeting
    conversation = maya.maya_greeting()

    # If there are violations, show how Maya would handle the first one
    violations = list(governor.violations.values())
    if violations:
        # Sort by severity for demo
        violations.sort(key=lambda v: (v.severity.value, -v.confidence), reverse=True)
        first_violation = violations[0]

        conversation += "\n\n" + "=" * 60 + "\n"
        conversation += "**Maya's Conversation Flow Example:**\n"
        conversation += "=" * 60 + "\n\n"

        # Maya explains the violation
        conversation += "**Maya:** " + maya.maya_ask_for_decision(first_violation)

        # Simulate user response
        conversation += "\n\n**You:** I think we should add this new type to our ontology.\n\n"

        # Maya interprets and confirms
        action, rationale = maya.maya_interpret_response(
            "I think we should add this new type to our ontology",
            first_violation
        )

        if action:
            conversation += "**Maya:** " + maya.maya_confirm_decision(action, rationale, first_violation)
            conversation += "\n\n**You:** Yes, that's correct.\n\n"
            conversation += "**Maya:** " + maya.maya_decision_applied(action, first_violation)

        # Check for batch opportunities
        similar_violations = [
            v for v in violations[1:]
            if v.violation_type == first_violation.violation_type
        ]

        if len(similar_violations) >= 2:
            conversation += "\n\n**Maya:** " + maya.maya_batch_suggestion(similar_violations)

    return conversation
