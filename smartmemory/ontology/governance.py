"""
Intelligent Ontology Governance System

This module provides post-ingestion ontology governance through HITL workflows,
intelligent analysis, and evolution-driven enforcement. Rather than rigid
ingestion-time validation, it enables smart curation and gradual improvement.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Dict, List, Any, Tuple

from smartmemory.models.memory_item import MemoryItem
from smartmemory.ontology.llm_manager import LLMOntologyManager
from smartmemory.ontology.manager import OntologyManager
from smartmemory.ontology.models import Ontology


class GovernanceAction(Enum):
    """Types of governance actions that can be taken."""
    APPROVE = "approve"
    REJECT = "reject"
    EVOLVE_ONTOLOGY = "evolve_ontology"
    FIX_DATA = "fix_data"
    IGNORE = "ignore"
    REVIEW_LATER = "review_later"


class ViolationSeverity(Enum):
    """Severity levels for ontology violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OntologyViolation:
    """Represents a violation of ontology rules in stored data."""
    id: str
    item_id: str
    ontology_id: str
    violation_type: str  # "unknown_entity_type", "invalid_property", "missing_relationship", etc.
    severity: ViolationSeverity
    description: str
    suggested_fix: str
    confidence: float  # 0.0-1.0
    detected_at: datetime
    data_context: Dict[str, Any]
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_id": self.item_id,
            "ontology_id": self.ontology_id,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "data_context": self.data_context,
            "auto_fixable": self.auto_fixable
        }


@dataclass
class GovernanceDecision:
    """Represents a human or AI decision on how to handle a violation."""
    violation_id: str
    action: GovernanceAction
    rationale: str
    decided_by: str  # "human", "ai", "system"
    decided_at: datetime
    metadata: Dict[str, Any]


class OntologyGovernor:
    """Intelligent post-ingestion ontology governance system."""

    def __init__(self, ontology_manager: OntologyManager,
                 llm_manager: LLMOntologyManager = None,
                 smart_memory=None):
        self.ontology_manager = ontology_manager
        self.llm_manager = llm_manager
        self.smart_memory = smart_memory
        self.logger = logging.getLogger(__name__)

        # Governance state
        self.violations: Dict[str, OntologyViolation] = {}
        self.decisions: Dict[str, GovernanceDecision] = {}
        self.governance_history: List[Dict[str, Any]] = []

    def analyze_data_against_ontologies(self, memory_items: List[MemoryItem] = None,
                                        ontology_ids: List[str] = None) -> List[OntologyViolation]:
        """Analyze stored data against ontologies to identify violations."""
        self.logger.info("Starting ontology governance analysis")

        violations = []

        # Get data to analyze
        if memory_items is None:
            memory_items = self._get_recent_memory_items()

        # Get ontologies to check against
        if ontology_ids is None:
            ontology_ids = [ont['id'] for ont in self.ontology_manager.list_ontologies()]

        for ontology_id in ontology_ids:
            ontology = self.ontology_manager.load_ontology(ontology_id)
            if not ontology:
                continue

            self.logger.info(f"Analyzing {len(memory_items)} items against ontology: {ontology.name}")

            for item in memory_items:
                item_violations = self._analyze_item_against_ontology(item, ontology)
                violations.extend(item_violations)

        # Store violations for HITL review
        for violation in violations:
            self.violations[violation.id] = violation

        self.logger.info(f"Found {len(violations)} ontology violations requiring review")
        return violations

    def get_violations_for_review(self, severity_filter: ViolationSeverity = None,
                                  auto_fixable_only: bool = False) -> List[OntologyViolation]:
        """Get violations that need human review."""
        violations = list(self.violations.values())

        if severity_filter:
            violations = [v for v in violations if v.severity == severity_filter]

        if auto_fixable_only:
            violations = [v for v in violations if v.auto_fixable]

        # Sort by severity and confidence
        severity_order = {
            ViolationSeverity.CRITICAL: 4, ViolationSeverity.HIGH: 3,
            ViolationSeverity.MEDIUM: 2, ViolationSeverity.LOW: 1
        }

        violations.sort(key=lambda v: (severity_order[v.severity], v.confidence), reverse=True)
        return violations

    def apply_governance_decision(self, violation_id: str, action: GovernanceAction,
                                  rationale: str, decided_by: str = "human") -> bool:
        """Apply a governance decision to resolve a violation."""
        if violation_id not in self.violations:
            self.logger.error(f"Violation {violation_id} not found")
            return False

        violation = self.violations[violation_id]
        decision = GovernanceDecision(
            violation_id=violation_id,
            action=action,
            rationale=rationale,
            decided_by=decided_by,
            decided_at=datetime.now(UTC),
            metadata={}
        )

        self.logger.info(f"Applying governance decision: {action.value} for violation {violation_id}")

        success = False
        try:
            if action == GovernanceAction.APPROVE:
                success = self._approve_violation(violation)
            elif action == GovernanceAction.EVOLVE_ONTOLOGY:
                success = self._evolve_ontology_for_violation(violation)
            elif action == GovernanceAction.FIX_DATA:
                success = self._fix_data_for_violation(violation)
            elif action == GovernanceAction.REJECT:
                success = self._reject_violation(violation)
            elif action == GovernanceAction.IGNORE:
                success = True  # Just mark as handled
            elif action == GovernanceAction.REVIEW_LATER:
                success = True  # Keep for later review

            if success:
                self.decisions[violation_id] = decision
                if action != GovernanceAction.REVIEW_LATER:
                    # Remove from active violations if resolved
                    del self.violations[violation_id]

                # Record in governance history
                self.governance_history.append({
                    "timestamp": datetime.now(UTC).isoformat(),
                    "violation": violation.to_dict(),
                    "decision": {
                        "action": action.value,
                        "rationale": rationale,
                        "decided_by": decided_by
                    }
                })

        except Exception as e:
            self.logger.error(f"Failed to apply governance decision: {e}")
            success = False

        return success

    def auto_fix_violations(self, confidence_threshold: float = 0.8) -> Tuple[int, int]:
        """Automatically fix high-confidence, auto-fixable violations."""
        auto_fixable = [v for v in self.violations.values()
                        if v.auto_fixable and v.confidence >= confidence_threshold]

        fixed_count = 0
        failed_count = 0

        for violation in auto_fixable:
            success = self.apply_governance_decision(
                violation.id,
                GovernanceAction.FIX_DATA,
                f"Auto-fix (confidence: {violation.confidence:.2f})",
                decided_by="system"
            )

            if success:
                fixed_count += 1
            else:
                failed_count += 1

        self.logger.info(f"Auto-fixed {fixed_count} violations, {failed_count} failed")
        return fixed_count, failed_count

    def suggest_ontology_evolution(self, violation_patterns: List[OntologyViolation] = None) -> Dict[str, Any]:
        """Suggest ontology evolution based on violation patterns."""
        if violation_patterns is None:
            violation_patterns = list(self.violations.values())

        # Group violations by type and ontology
        patterns = {}
        for violation in violation_patterns:
            key = (violation.ontology_id, violation.violation_type)
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(violation)

        suggestions = {}
        for (ontology_id, violation_type), violations in patterns.items():
            if len(violations) >= 3:  # Pattern threshold
                suggestion = self._generate_evolution_suggestion(ontology_id, violation_type, violations)
                suggestions[f"{ontology_id}_{violation_type}"] = suggestion

        return suggestions

    def run_periodic_governance(self, days_back: int = 7) -> Dict[str, Any]:
        """Run periodic governance analysis on recent data."""
        self.logger.info(f"Running periodic governance analysis for last {days_back} days")

        # Get recent memory items
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
        recent_items = self._get_memory_items_since(cutoff_date)

        # Analyze against all ontologies
        violations = self.analyze_data_against_ontologies(recent_items)

        # Auto-fix high-confidence violations
        fixed_count, failed_count = self.auto_fix_violations()

        # Generate evolution suggestions
        evolution_suggestions = self.suggest_ontology_evolution()

        # Prepare summary
        summary = {
            "analysis_date": datetime.now(UTC).isoformat(),
            "items_analyzed": len(recent_items),
            "violations_found": len(violations),
            "auto_fixed": fixed_count,
            "auto_fix_failed": failed_count,
            "pending_review": len(self.violations),
            "evolution_suggestions": len(evolution_suggestions),
            "violation_breakdown": self._get_violation_breakdown(),
            "suggestions": evolution_suggestions
        }

        self.logger.info(f"Periodic governance complete: {summary}")
        return summary

    def _get_recent_memory_items(self, limit: int = 1000) -> List[MemoryItem]:
        """Get recent memory items for analysis."""
        if not self.smart_memory:
            return []

        try:
            # Get recent items from SmartMemory
            items = self.smart_memory.search("*", top_k=limit)
            return items if items else []
        except Exception as e:
            self.logger.warning(f"Failed to get recent memory items: {e}")
            return []

    def _get_memory_items_since(self, cutoff_date: datetime) -> List[MemoryItem]:
        """Get memory items created since a specific date."""
        # This would need to be implemented based on the actual storage backend
        # For now, return recent items
        return self._get_recent_memory_items()

    def _analyze_item_against_ontology(self, item: MemoryItem, ontology: Ontology) -> List[OntologyViolation]:
        """Analyze a single memory item against an ontology."""
        violations = []

        # Extract entities and relationships from the item
        entities = self._extract_entities_from_item(item)
        relationships = self._extract_relationships_from_item(item)

        # Check entity violations
        for entity in entities:
            entity_violations = self._check_entity_violations(entity, ontology, item)
            violations.extend(entity_violations)

        # Check relationship violations
        for relationship in relationships:
            rel_violations = self._check_relationship_violations(relationship, ontology, item)
            violations.extend(rel_violations)

        return violations

    def _extract_entities_from_item(self, item: MemoryItem) -> List[Dict[str, Any]]:
        """Extract entities from a memory item."""
        # This would integrate with the ontology extractor
        # For now, return mock entities based on metadata
        entities = []

        if hasattr(item, 'metadata') and item.metadata:
            # Look for entity information in metadata
            if 'entities' in item.metadata:
                entities = item.metadata['entities']
            elif 'entity_type' in item.metadata:
                entities = [{
                    'name': item.metadata.get('name', 'unknown'),
                    'type': item.metadata['entity_type'],
                    'properties': {k: v for k, v in item.metadata.items()
                                   if k not in ['entity_type', 'name']}
                }]

        return entities

    def _extract_relationships_from_item(self, item: MemoryItem) -> List[Dict[str, Any]]:
        """Extract relationships from a memory item."""
        # This would integrate with the relationship extractor
        relationships = []

        if hasattr(item, 'metadata') and item.metadata:
            if 'relations' in item.metadata:
                relationships = item.metadata['relations']

        return relationships

    def _check_entity_violations(self, entity: Dict[str, Any], ontology: Ontology,
                                 item: MemoryItem) -> List[OntologyViolation]:
        """Check for entity-related ontology violations."""
        violations = []
        entity_type = entity.get('type', '').lower()
        entity_properties = entity.get('properties') or {}

        # Check if entity type exists in ontology
        if entity_type and entity_type not in ontology.entity_types:
            violation = OntologyViolation(
                id=f"unknown_entity_{item.item_id}_{entity_type}",
                item_id=item.item_id,
                ontology_id=ontology.id,
                violation_type="unknown_entity_type",
                severity=ViolationSeverity.MEDIUM,
                description=f"Entity type '{entity_type}' not defined in ontology",
                suggested_fix=f"Add '{entity_type}' to ontology or map to existing type",
                confidence=0.9,
                detected_at=datetime.now(UTC),
                data_context={
                    "entity": entity,
                    "item_content": item.content[:200] if item.content else ""
                },
                auto_fixable=False
            )
            violations.append(violation)

        # Check entity properties if type exists
        elif entity_type in ontology.entity_types:
            entity_def = ontology.entity_types[entity_type]

            # Check for missing required properties
            for req_prop in entity_def.required_properties:
                if req_prop not in entity_properties:
                    violation = OntologyViolation(
                        id=f"missing_prop_{item.item_id}_{entity_type}_{req_prop}",
                        item_id=item.item_id,
                        ontology_id=ontology.id,
                        violation_type="missing_required_property",
                        severity=ViolationSeverity.HIGH,
                        description=f"Missing required property '{req_prop}' for {entity_type}",
                        suggested_fix=f"Add '{req_prop}' property or make it optional",
                        confidence=0.95,
                        detected_at=datetime.now(UTC),
                        data_context={
                            "entity": entity,
                            "required_property": req_prop
                        },
                        auto_fixable=True
                    )
                    violations.append(violation)

        return violations

    def _check_relationship_violations(self, relationship: Dict[str, Any], ontology: Ontology,
                                       item: MemoryItem) -> List[OntologyViolation]:
        """Check for relationship-related ontology violations."""
        violations = []
        rel_type = relationship.get('type', '').lower()

        # Check if relationship type exists in ontology
        if rel_type and rel_type not in ontology.relationship_types:
            violation = OntologyViolation(
                id=f"unknown_rel_{item.item_id}_{rel_type}",
                item_id=item.item_id,
                ontology_id=ontology.id,
                violation_type="unknown_relationship_type",
                severity=ViolationSeverity.MEDIUM,
                description=f"Relationship type '{rel_type}' not defined in ontology",
                suggested_fix=f"Add '{rel_type}' to ontology or map to existing type",
                confidence=0.85,
                detected_at=datetime.now(UTC),
                data_context={
                    "relationship": relationship
                },
                auto_fixable=False
            )
            violations.append(violation)

        return violations

    def _approve_violation(self, violation: OntologyViolation) -> bool:
        """Approve a violation (accept the data as-is)."""
        self.logger.info(f"Approving violation: {violation.description}")
        # Mark as approved in metadata or tracking system
        return True

    def _evolve_ontology_for_violation(self, violation: OntologyViolation) -> bool:
        """Evolve the ontology to accommodate the violation."""
        if not self.llm_manager:
            return False

        try:
            ontology = self.ontology_manager.load_ontology(violation.ontology_id)
            if not ontology:
                return False

            # Use LLM to suggest ontology evolution
            analysis = self.llm_manager.analyze_ontology(violation.ontology_id)
            evolution_plan = self.llm_manager.suggest_ontology_improvements(violation.ontology_id, analysis)

            # Apply evolution automatically for low-risk changes
            success, message = self.llm_manager.evolve_ontology_automatically(
                violation.ontology_id, evolution_plan, auto_approve_low_risk=True
            )

            self.logger.info(f"Ontology evolution for violation: {success} - {message}")
            return success

        except Exception as e:
            self.logger.error(f"Failed to evolve ontology: {e}")
            return False

    def _fix_data_for_violation(self, violation: OntologyViolation) -> bool:
        """Fix the data to comply with the ontology."""
        # This would implement data correction logic
        # For now, just log the action
        self.logger.info(f"Fixing data for violation: {violation.description}")
        return True

    def _reject_violation(self, violation: OntologyViolation) -> bool:
        """Reject the violation (remove or flag the data)."""
        self.logger.info(f"Rejecting violation: {violation.description}")
        # Could implement data removal or flagging logic
        return True

    def _generate_evolution_suggestion(self, ontology_id: str, violation_type: str,
                                       violations: List[OntologyViolation]) -> Dict[str, Any]:
        """Generate ontology evolution suggestion based on violation patterns."""
        return {
            "ontology_id": ontology_id,
            "violation_type": violation_type,
            "violation_count": len(violations),
            "suggested_action": "add_missing_types",
            "confidence": min(0.9, len(violations) * 0.1),
            "rationale": f"Pattern of {len(violations)} similar violations suggests ontology gap"
        }

    def _get_violation_breakdown(self) -> Dict[str, int]:
        """Get breakdown of violations by type and severity."""
        breakdown = {}
        for violation in self.violations.values():
            key = f"{violation.violation_type}_{violation.severity.value}"
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown
