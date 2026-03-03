"""
LLM-Driven Ontology Management System

This module provides AI-driven ontology evolution, enrichment, and refinement capabilities.
The LLM analyzes extraction patterns, identifies gaps, suggests improvements, and evolves
ontologies automatically based on usage patterns and domain knowledge.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, List, Any, Tuple, Optional

from smartmemory.integration.llm.prompts.prompt_provider import get_prompt_value, apply_placeholders
from smartmemory.ontology.manager import (
    OntologyManager
)
from smartmemory.ontology.models import EntityTypeDefinition, RelationshipTypeDefinition, OntologyRule, Ontology


@dataclass
class OntologyAnalysis:
    """Analysis results from LLM ontology evaluation."""
    ontology_id: str
    analysis_date: datetime
    coverage_score: float  # 0.0-1.0
    consistency_score: float  # 0.0-1.0
    completeness_score: float  # 0.0-1.0
    overall_quality: float  # 0.0-1.0
    gaps_identified: List[str]
    improvement_suggestions: List[str]
    new_entity_types: List[Dict[str, Any]]
    new_relationship_types: List[Dict[str, Any]]
    rule_suggestions: List[Dict[str, Any]]
    confidence: float  # 0.0-1.0


@dataclass
class EvolutionPlan:
    """Plan for evolving an ontology based on LLM analysis."""
    ontology_id: str
    created_date: datetime
    priority: str  # "high", "medium", "low"
    changes: List[Dict[str, Any]]
    rationale: str
    estimated_impact: str
    risk_assessment: str
    approval_required: bool


class LLMOntologyManager:
    """AI-driven ontology management and evolution system."""

    def __init__(self, ontology_manager: OntologyManager, llm_client=None):
        self.ontology_manager = ontology_manager
        self.llm_client = llm_client or self._get_default_llm_client()
        self.logger = logging.getLogger(__name__)

    def _get_default_llm_client(self):
        """Get default LLM client for ontology management."""
        try:
            import openai
            return openai.OpenAI()
        except ImportError:
            self.logger.warning("OpenAI client not available. LLM features will be limited.")
            return None

    def analyze_ontology(self, ontology_id: str, extraction_history: List[Dict] = None, template_override: Optional[str] = None) -> OntologyAnalysis:
        """Analyze an ontology for completeness, consistency, and improvement opportunities."""
        self.logger.info(f"Starting LLM analysis of ontology: {ontology_id}")

        ontology = self.ontology_manager.load_ontology(ontology_id)
        if not ontology:
            raise ValueError(f"Ontology {ontology_id} not found")

        # Prepare analysis context
        analysis_context = self._prepare_analysis_context(ontology, extraction_history)

        # Generate LLM analysis
        analysis_prompt = self._create_analysis_prompt(analysis_context, template_override)
        llm_response = self._call_llm(analysis_prompt)

        # Parse and structure the analysis
        analysis = self._parse_analysis_response(llm_response, ontology_id)

        self.logger.info(f"Ontology analysis complete. Quality score: {analysis.overall_quality:.2f}")
        return analysis

    def suggest_ontology_improvements(self, ontology_id: str, analysis: OntologyAnalysis = None, template_override: Optional[str] = None) -> EvolutionPlan:
        """Generate specific improvement suggestions for an ontology."""
        if analysis is None:
            analysis = self.analyze_ontology(ontology_id)

        self.logger.info(f"Generating improvement suggestions for ontology: {ontology_id}")

        # Create improvement prompt
        improvement_prompt = self._create_improvement_prompt(analysis, template_override)
        llm_response = self._call_llm(improvement_prompt)

        # Parse improvement suggestions
        evolution_plan = self._parse_improvement_response(llm_response, ontology_id)

        self.logger.info(f"Generated {len(evolution_plan.changes)} improvement suggestions")
        return evolution_plan

    def evolve_ontology_automatically(self, ontology_id: str, evolution_plan: EvolutionPlan = None,
                                      auto_approve_low_risk: bool = True) -> Tuple[bool, str]:
        """Automatically evolve an ontology based on LLM suggestions."""
        if evolution_plan is None:
            analysis = self.analyze_ontology(ontology_id)
            evolution_plan = self.suggest_ontology_improvements(ontology_id, analysis)

        self.logger.info(f"Starting automatic evolution of ontology: {ontology_id}")

        # Check if auto-approval is allowed
        if evolution_plan.approval_required and not auto_approve_low_risk:
            return False, "Evolution requires manual approval"

        if evolution_plan.risk_assessment == "high" and not auto_approve_low_risk:
            return False, "High-risk changes require manual approval"

        # Apply changes
        try:
            ontology = self.ontology_manager.load_ontology(ontology_id)
            changes_applied = 0

            for change in evolution_plan.changes:
                success = self._apply_change(ontology, change)
                if success:
                    changes_applied += 1

            # Save evolved ontology
            self.ontology_manager.storage.save_ontology(ontology)

            # Add evolution metadata
            evolution_rule = OntologyRule(
                id=f"llm_evolution_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                name="llm_evolution",
                rule_type="evolution",
                description=f"LLM-driven evolution: {evolution_plan.rationale}",
                conditions={},
                actions={"changes_applied": changes_applied, "evolution_plan_id": evolution_plan.ontology_id, "auto_applied": True},
                created_by="llm_manager",
                created_at=datetime.now(UTC)
            )
            ontology.add_rule(evolution_rule)
            self.ontology_manager.storage.save_ontology(ontology)

            self.logger.info(f"Successfully applied {changes_applied}/{len(evolution_plan.changes)} changes")
            return True, f"Applied {changes_applied} improvements automatically"

        except Exception as e:
            self.logger.error(f"Failed to evolve ontology: {e}")
            return False, f"Evolution failed: {str(e)}"

    def enrich_ontology_from_domain_knowledge(self, ontology_id: str, domain: str, template_override: Optional[str] = None) -> bool:
        """Enrich an ontology with domain-specific knowledge from LLM."""
        self.logger.info(f"Enriching ontology {ontology_id} with {domain} domain knowledge")

        ontology = self.ontology_manager.load_ontology(ontology_id)
        if not ontology:
            raise ValueError(f"Ontology {ontology_id} not found")

        # Generate domain enrichment prompt
        enrichment_prompt = self._create_domain_enrichment_prompt(ontology, domain, template_override)
        llm_response = self._call_llm(enrichment_prompt)

        # Parse and apply enrichments
        enrichments = self._parse_enrichment_response(llm_response)

        changes_applied = 0
        for enrichment in enrichments:
            success = self._apply_enrichment(ontology, enrichment)
            if success:
                changes_applied += 1

        # Save enriched ontology
        self.ontology_manager.storage.save_ontology(ontology)

        self.logger.info(f"Applied {changes_applied} domain enrichments")
        return changes_applied > 0

    def validate_ontology_with_llm(self, ontology_id: str, template_override: Optional[str] = None) -> Dict[str, Any]:
        """Use LLM to validate ontology structure and semantics."""
        ontology = self.ontology_manager.load_ontology(ontology_id)
        if not ontology:
            raise ValueError(f"Ontology {ontology_id} not found")

        validation_prompt = self._create_validation_prompt(ontology, template_override)
        llm_response = self._call_llm(validation_prompt)

        validation_results = self._parse_validation_response(llm_response)

        self.logger.info(f"LLM validation complete. Issues found: {len(validation_results.get('issues', []))}")
        return validation_results

    def _prepare_analysis_context(self, ontology: Ontology, extraction_history: List[Dict] = None) -> Dict[str, Any]:
        """Prepare context for ontology analysis."""
        context = {
            "ontology": {
                "name": ontology.name,
                "domain": ontology.domain,
                "entity_types": len(ontology.entity_types),
                "relationship_types": len(ontology.relationship_types),
                "rules": len(ontology.rules)
            },
            "entity_types": [
                {
                    "name": name,
                    "properties": list(et.properties.keys()),
                    "required_properties": list(et.required_properties),
                    "parent_types": list(et.parent_types),
                    "aliases": list(et.aliases)
                }
                for name, et in ontology.entity_types.items()
            ],
            "relationship_types": [
                {
                    "name": name,
                    "source_types": list(rt.source_types),
                    "target_types": list(rt.target_types),
                    "bidirectional": rt.bidirectional,
                    "aliases": list(rt.aliases)
                }
                for name, rt in ontology.relationship_types.items()
            ]
        }

        if extraction_history:
            context["extraction_patterns"] = self._analyze_extraction_patterns(extraction_history)

        return context

    def _analyze_extraction_patterns(self, extraction_history: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in extraction history."""
        entity_types = {}
        relationship_types = {}

        for extraction in extraction_history:
            # Count entity types
            for entity in extraction.get("entities", []):
                entity_type = entity.get("type", "unknown")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            # Count relationship types
            for relation in extraction.get("relations", []):
                rel_type = relation.get("relation_type", "unknown")
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        return {
            "total_extractions": len(extraction_history),
            "entity_type_frequency": entity_types,
            "relationship_type_frequency": relationship_types,
            "most_common_entities": sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10],
            "most_common_relationships": sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def _create_analysis_prompt(self, context: Dict[str, Any], template_override: Optional[str] = None) -> str:
        """Create prompt for ontology analysis using centralized templates.
        Precedence: method override > config; no inline defaults.
        """
        extraction_block = ""
        if context.get('extraction_patterns'):
            extraction_block = f"EXTRACTION PATTERNS: {json.dumps(context.get('extraction_patterns') or {}, indent=2)}"
        tpl = template_override or get_prompt_value('ontology.manager.analysis_template')
        if not tpl:
            raise ValueError("Missing prompt template 'ontology.manager.analysis_template' in prompts.json")
        return apply_placeholders(tpl, {
            'ONTOLOGY_NAME': context['ontology']['name'],
            'ONTOLOGY_DOMAIN': context['ontology']['domain'],
            'ENTITY_TYPES_COUNT': str(context['ontology']['entity_types']),
            'RELATIONSHIP_TYPES_COUNT': str(context['ontology']['relationship_types']),
            'ENTITY_TYPES_JSON': json.dumps(context['entity_types'], indent=2),
            'RELATIONSHIP_TYPES_JSON': json.dumps(context['relationship_types'], indent=2),
            'EXTRACTION_PATTERNS_BLOCK': extraction_block,
        })

    def _create_improvement_prompt(self, analysis: OntologyAnalysis, template_override: Optional[str] = None) -> str:
        """Create prompt for improvement suggestions using centralized templates.
        Precedence: method override > config; no inline defaults.
        """
        tpl = template_override or get_prompt_value('ontology.manager.improvement_template')
        if not tpl:
            raise ValueError("Missing prompt template 'ontology.manager.improvement_template' in prompts.json")
        return apply_placeholders(tpl, {
            'COVERAGE_SCORE': str(analysis.coverage_score),
            'CONSISTENCY_SCORE': str(analysis.consistency_score),
            'COMPLETENESS_SCORE': str(analysis.completeness_score),
            'OVERALL_QUALITY': str(analysis.overall_quality),
            'GAPS_JSON': json.dumps(analysis.gaps_identified, indent=2),
            'IMPROVEMENTS_JSON': json.dumps(analysis.improvement_suggestions, indent=2),
            'NEW_ENTITY_TYPES_JSON': json.dumps(analysis.new_entity_types, indent=2),
            'NEW_RELATIONSHIP_TYPES_JSON': json.dumps(analysis.new_relationship_types, indent=2),
        })

    def _create_domain_enrichment_prompt(self, ontology: Ontology, domain: str, template_override: Optional[str] = None) -> str:
        """Create prompt for domain-specific enrichment using centralized templates.
        Precedence: method override > config; no inline defaults.
        """
        tpl = template_override or get_prompt_value('ontology.manager.domain_enrichment_template')
        if not tpl:
            raise ValueError("Missing prompt template 'ontology.manager.domain_enrichment_template' in prompts.json")
        return apply_placeholders(tpl, {
            'DOMAIN': domain,
            'ONTOLOGY_NAME': ontology.name,
            'ONTOLOGY_DOMAIN': ontology.domain,
            'ENTITY_TYPES_LIST': str(list(ontology.entity_types.keys())),
            'RELATIONSHIP_TYPES_LIST': str(list(ontology.relationship_types.keys())),
        })

    def _create_validation_prompt(self, ontology: Ontology, template_override: Optional[str] = None) -> str:
        """Create prompt for ontology validation using centralized templates.
        Precedence: method override > config; no inline defaults.
        """
        tpl = template_override or get_prompt_value('ontology.manager.validation_template')
        if not tpl:
            raise ValueError("Missing prompt template 'ontology.manager.validation_template' in prompts.json")
        return apply_placeholders(tpl, {
            'ONTOLOGY_NAME': ontology.name,
            'ENTITY_TYPES_LIST': str(list(ontology.entity_types.keys())),
            'RELATIONSHIP_TYPES_LIST': str(list(ontology.relationship_types.keys())),
        })

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        if not self.llm_client:
            # Return mock response for testing
            return self._get_mock_response(prompt)

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert ontology engineer and knowledge management specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._get_mock_response(prompt)

    def _get_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing when LLM is not available."""
        if "analysis" in prompt.lower():
            return json.dumps({
                "coverage_score": 0.75,
                "consistency_score": 0.85,
                "completeness_score": 0.70,
                "overall_quality": 0.77,
                "gaps_identified": ["Missing temporal relationships", "Incomplete property definitions"],
                "improvement_suggestions": ["Add created_at properties", "Define hierarchical relationships"],
                "new_entity_types": [{"name": "event", "properties": {"timestamp": "datetime", "duration": "integer"}}],
                "new_relationship_types": [{"name": "occurs_during", "source_types": ["event"], "target_types": ["time_period"]}],
                "rule_suggestions": [{"name": "temporal_consistency", "type": "validation"}],
                "confidence": 0.80
            })
        elif "improvement" in prompt.lower():
            return json.dumps({
                "priority": "medium",
                "changes": [
                    {"type": "add_entity_type", "data": {"name": "event", "properties": {"timestamp": "datetime"}}},
                    {"type": "add_relationship_type", "data": {"name": "occurs_during", "source_types": ["event"], "target_types": ["time_period"]}}
                ],
                "rationale": "Adding temporal concepts improves domain coverage",
                "estimated_impact": "Moderate improvement in temporal relationship modeling",
                "risk_assessment": "low",
                "approval_required": False
            })
        else:
            return json.dumps({"status": "mock_response", "message": "LLM not available"})

    def _parse_analysis_response(self, response: str, ontology_id: str) -> OntologyAnalysis:
        """Parse LLM analysis response into structured format."""
        try:
            data = json.loads(response)
            return OntologyAnalysis(
                ontology_id=ontology_id,
                analysis_date=datetime.now(UTC),
                coverage_score=data.get("coverage_score", 0.0),
                consistency_score=data.get("consistency_score", 0.0),
                completeness_score=data.get("completeness_score", 0.0),
                overall_quality=data.get("overall_quality", 0.0),
                gaps_identified=data.get("gaps_identified", []),
                improvement_suggestions=data.get("improvement_suggestions", []),
                new_entity_types=data.get("new_entity_types", []),
                new_relationship_types=data.get("new_relationship_types", []),
                rule_suggestions=data.get("rule_suggestions", []),
                confidence=data.get("confidence", 0.0)
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse analysis response: {e}")
            # Return default analysis
            return OntologyAnalysis(
                ontology_id=ontology_id,
                analysis_date=datetime.now(UTC),
                coverage_score=0.5,
                consistency_score=0.5,
                completeness_score=0.5,
                overall_quality=0.5,
                gaps_identified=["Analysis parsing failed"],
                improvement_suggestions=["Retry analysis"],
                new_entity_types=[],
                new_relationship_types=[],
                rule_suggestions=[],
                confidence=0.1
            )

    def _parse_improvement_response(self, response: str, ontology_id: str) -> EvolutionPlan:
        """Parse LLM improvement response into structured format."""
        try:
            data = json.loads(response)
            return EvolutionPlan(
                ontology_id=ontology_id,
                created_date=datetime.now(UTC),
                priority=data.get("priority", "medium"),
                changes=data.get("changes", []),
                rationale=data.get("rationale", ""),
                estimated_impact=data.get("estimated_impact", ""),
                risk_assessment=data.get("risk_assessment", "medium"),
                approval_required=data.get("approval_required", True)
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse improvement response: {e}")
            return EvolutionPlan(
                ontology_id=ontology_id,
                created_date=datetime.now(UTC),
                priority="low",
                changes=[],
                rationale="Parsing failed",
                estimated_impact="Unknown",
                risk_assessment="high",
                approval_required=True
            )

    def _parse_enrichment_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse domain enrichment response."""
        try:
            data = json.loads(response)
            return data.get("enrichments", [])
        except json.JSONDecodeError:
            return []

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse validation response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Failed to parse validation response"}

    def _apply_change(self, ontology: Ontology, change: Dict[str, Any]) -> bool:
        """Apply a specific change to an ontology."""
        try:
            change_type = change.get("type")
            data = change.get("data") or {}

            if change_type == "add_entity_type":
                entity_type = EntityTypeDefinition(
                    name=data["name"],
                    description=data.get("description", f"LLM-generated {data['name']} entity type"),
                    properties=data.get("properties") or {},
                    required_properties=set(data.get("required_properties", [])),
                    parent_types=set(data.get("parent_types", [])),
                    aliases=set(data.get("aliases", [])),
                    examples=data.get("examples", []),
                    created_by="llm_manager",
                    created_at=datetime.now(UTC)
                )
                ontology.add_entity_type(entity_type)
                return True

            elif change_type == "add_relationship_type":
                rel_type = RelationshipTypeDefinition(
                    name=data["name"],
                    description=data.get("description", f"LLM-generated {data['name']} relationship type"),
                    source_types=set(data.get("source_types", [])),
                    target_types=set(data.get("target_types", [])),
                    properties=data.get("properties") or {},
                    bidirectional=data.get("bidirectional", False),
                    aliases=set(data.get("aliases", [])),
                    examples=data.get("examples", []),
                    created_by="llm_manager",
                    created_at=datetime.now(UTC)
                )
                ontology.add_relationship_type(rel_type)
                return True

            elif change_type == "add_rule":
                rule = OntologyRule(
                    id=data.get("id", f"llm_rule_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"),
                    name=data["name"],
                    rule_type=data.get("rule_type", "validation"),
                    description=data.get("description", "LLM-generated rule"),
                    conditions=data.get("conditions") or {},
                    actions=data.get("actions") or {},
                    created_by="llm_manager",
                    created_at=datetime.now(UTC)
                )
                ontology.add_rule(rule)
                return True

            else:
                self.logger.warning(f"Unknown change type: {change_type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to apply change {change}: {e}")
            return False

    def _apply_enrichment(self, ontology: Ontology, enrichment: Dict[str, Any]) -> bool:
        """Apply domain enrichment to ontology."""
        # Similar to _apply_change but for enrichments
        return self._apply_change(ontology, enrichment)
