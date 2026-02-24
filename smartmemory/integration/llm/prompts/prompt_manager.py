"""
Prompt Template Management

Centralized management of prompt templates for LLM operations.
"""

import logging
from typing import Dict, Optional, Any

from smartmemory.integration.llm.prompts.prompt_provider import get_prompt_value, apply_placeholders

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Centralized prompt template management.
    
    Consolidates prompt handling from:
    - LLMOntologyManager prompt creation methods
    - Various scattered prompt templates
    """

    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_ontology_analysis_prompt(self,
                                     context: Dict[str, Any],
                                     template_override: Optional[str] = None) -> str:
        """
        Create ontology analysis prompt.
        
        Consolidates LLMOntologyManager._create_analysis_prompt()
        """
        extraction_block = ""
        if context.get('extraction_patterns'):
            extraction_block = f"EXTRACTION PATTERNS: {context.get('extraction_patterns', {})}"

        template = template_override or get_prompt_value('ontology.manager.analysis_template')
        if not template:
            # Fallback template
            template = """
            Analyze the following ontology for completeness, consistency, and improvement opportunities:
            
            Ontology: {ONTOLOGY_NAME}
            Domain: {ONTOLOGY_DOMAIN}
            Entity Types: {ENTITY_TYPES_COUNT}
            Relationship Types: {RELATIONSHIP_TYPES_COUNT}
            
            Entity Types:
            {ENTITY_TYPES_JSON}
            
            Relationship Types:
            {RELATIONSHIP_TYPES_JSON}
            
            {EXTRACTION_PATTERNS_BLOCK}
            
            Return a JSON object with coverage_score, consistency_score, completeness_score, 
            overall_quality, gaps_identified, improvement_suggestions, new_entity_types, 
            new_relationship_types, rule_suggestions, and confidence.
            """

        return apply_placeholders(template, {
            'ONTOLOGY_NAME': context['ontology']['name'],
            'ONTOLOGY_DOMAIN': context['ontology']['domain'],
            'ENTITY_TYPES_COUNT': str(context['ontology']['entity_types']),
            'RELATIONSHIP_TYPES_COUNT': str(context['ontology']['relationship_types']),
            'ENTITY_TYPES_JSON': str(context['entity_types']),
            'RELATIONSHIP_TYPES_JSON': str(context['relationship_types']),
            'EXTRACTION_PATTERNS_BLOCK': extraction_block,
        })

    def get_improvement_prompt(self,
                               analysis: Dict[str, Any],
                               template_override: Optional[str] = None) -> str:
        """
        Create improvement suggestions prompt.
        
        Consolidates LLMOntologyManager._create_improvement_prompt()
        """
        template = template_override or get_prompt_value('ontology.manager.improvement_template')
        if not template:
            # Fallback template
            template = """
            Based on the following ontology analysis, suggest specific improvements:
            
            Coverage Score: {COVERAGE_SCORE}
            Consistency Score: {CONSISTENCY_SCORE}
            Completeness Score: {COMPLETENESS_SCORE}
            Overall Quality: {OVERALL_QUALITY}
            
            Identified Gaps:
            {GAPS_JSON}
            
            Improvement Suggestions:
            {IMPROVEMENTS_JSON}
            
            New Entity Types:
            {NEW_ENTITY_TYPES_JSON}
            
            New Relationship Types:
            {NEW_RELATIONSHIP_TYPES_JSON}
            
            Return a JSON object with priority, changes, rationale, estimated_impact, 
            risk_assessment, and approval_required.
            """

        return apply_placeholders(template, {
            'COVERAGE_SCORE': str(analysis.get('coverage_score', 0)),
            'CONSISTENCY_SCORE': str(analysis.get('consistency_score', 0)),
            'COMPLETENESS_SCORE': str(analysis.get('completeness_score', 0)),
            'OVERALL_QUALITY': str(analysis.get('overall_quality', 0)),
            'GAPS_JSON': str(analysis.get('gaps_identified', [])),
            'IMPROVEMENTS_JSON': str(analysis.get('improvement_suggestions', [])),
            'NEW_ENTITY_TYPES_JSON': str(analysis.get('new_entity_types', [])),
            'NEW_RELATIONSHIP_TYPES_JSON': str(analysis.get('new_relationship_types', [])),
        })

    def get_domain_enrichment_prompt(self,
                                     ontology: Any,
                                     domain: str,
                                     template_override: Optional[str] = None) -> str:
        """
        Create domain enrichment prompt.
        
        Consolidates LLMOntologyManager._create_domain_enrichment_prompt()
        """
        template = template_override or get_prompt_value('ontology.manager.domain_enrichment_template')
        if not template:
            # Fallback template
            template = """
            Enrich the following ontology with {DOMAIN} domain-specific knowledge:
            
            Ontology: {ONTOLOGY_NAME}
            Current Domain: {ONTOLOGY_DOMAIN}
            Target Domain: {DOMAIN}
            
            Current Entity Types: {ENTITY_TYPES_LIST}
            Current Relationship Types: {RELATIONSHIP_TYPES_LIST}
            
            Suggest domain-specific entity types, relationship types, and properties 
            that would improve coverage of the {DOMAIN} domain.
            
            Return a JSON object with enrichments array containing the suggested additions.
            """

        return apply_placeholders(template, {
            'DOMAIN': domain,
            'ONTOLOGY_NAME': ontology.name,
            'ONTOLOGY_DOMAIN': ontology.domain,
            'ENTITY_TYPES_LIST': str(list(ontology.entity_types.keys())),
            'RELATIONSHIP_TYPES_LIST': str(list(ontology.relationship_types.keys())),
        })

    def get_validation_prompt(self,
                              ontology: Any,
                              template_override: Optional[str] = None) -> str:
        """
        Create ontology validation prompt.
        
        Consolidates LLMOntologyManager._create_validation_prompt()
        """
        template = template_override or get_prompt_value('ontology.manager.validation_template')
        if not template:
            # Fallback template
            template = """
            Validate the structure and semantics of the following ontology:
            
            Ontology: {ONTOLOGY_NAME}
            Entity Types: {ENTITY_TYPES_LIST}
            Relationship Types: {RELATIONSHIP_TYPES_LIST}
            
            Check for:
            - Semantic consistency
            - Structural completeness
            - Logical relationships
            - Naming conventions
            - Domain appropriateness
            
            Return a JSON object with status, issues (array), suggestions (array), 
            and overall_score.
            """

        return apply_placeholders(template, {
            'ONTOLOGY_NAME': ontology.name,
            'ENTITY_TYPES_LIST': str(list(ontology.entity_types.keys())),
            'RELATIONSHIP_TYPES_LIST': str(list(ontology.relationship_types.keys())),
        })

    def get_extraction_prompt(self,
                              text: str,
                              schema: Optional[Dict[str, Any]] = None,
                              template_key: str = "plugins.extractors.llm.system_template") -> str:
        """
        Create entity/relationship extraction prompt.
        
        Uses the prompt provider (prompts.json or DB) as source of truth.
        This method adds dynamic text substitution on top of the base template.
        
        Args:
            text: Text to extract from
            schema: Optional schema to inject
            template_key: Prompt template key (default: LLM extractor template)
        """
        from .prompt_provider import get_prompt_value
        
        # Get base template from provider (prompts.json or DB)
        base_template = get_prompt_value(template_key)
        if not base_template:
            # Fallback to inline prompt if template not found
            base_template = """
        Extract entities and relationships from the following text for a knowledge graph.
        
        Text: {text}
        
        ENTITY CLASSIFICATION:
        Classify each entity accurately using these types:
        - person: People (real or fictional)
        - organization: Companies, institutions, agencies, groups
        - location: Countries, cities, geographic features (mountains, plains, rivers, regions)
        - event: Named events, historical occurrences, meetings
        - product: Tangible objects, products, tools, vehicles
        - work_of_art: Books, songs, movies, paintings, creative works
        - date: Temporal expressions, time periods
        - concept: Abstract ideas, theories, processes, qualities
        
        ENTITY METADATA (optional but valuable):
        For each entity, include additional attributes when relevant:
        - description: Brief explanation of what this entity is
        - aliases: Alternative names or spellings
        - domain: Field or area (e.g., "Computer Science", "Geography", "History")
        - confidence: Your confidence in this classification (0.0-1.0)
        
        RELATIONSHIPS:
        Extract meaningful relationships between entities. Use descriptive predicates like:
        - located_in, part_of, contains
        - created_by, founded_by, invented_by
        - works_for, member_of, belongs_to
        - happened_in, occurred_at
        - related_to, associated_with
        
        OUTPUT FORMAT:
        Return ONLY a JSON object with this structure:
        {{
          "entities": [
            {{
              "name": "entity name",
              "entity_type": "person|organization|location|event|product|work_of_art|date|concept",
              "confidence": 0.95,
              "attrs": {{
                "description": "brief description",
                "domain": "relevant field",
                "aliases": ["alternative name"]
              }}
            }}
          ],
          "triples": [
            {{
              "subject": "entity1",
              "predicate": "relationship_type",
              "object": "entity2",
              "subject_type": "entity_type",
              "object_type": "entity_type"
            }}
          ]
        }}
        
        Do not include markdown fences or commentary.
        If none found, return {{"entities": [], "triples": []}}.
        """
        
        # System template is already complete from prompts.json
        # Just return it as the system message (text substitution happens in LLM extractor)
        return base_template
