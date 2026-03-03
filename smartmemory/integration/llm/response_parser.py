"""
Response Parser for Structured LLM Responses

Handles parsing and validation of LLM responses with Pydantic models.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class StructuredResponse:
    """Structured response from LLM with parsing results."""
    parsed_data: Optional[Dict[str, Any]]
    raw_content: Optional[str]
    model: str
    provider: str
    success: bool
    error: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


class ResponseParser:
    """
    Parser for structured LLM responses.
    
    Consolidates parsing logic from:
    - utils.llm.call_llm() JSON parsing
    - LLMOntologyManager response parsing methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_json_response(self,
                            response_content: str,
                            response_model: Optional[Type[Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from LLM.
        
        Consolidates functionality from:
        - utils.llm.call_llm() JSON parsing
        - LLMOntologyManager._parse_analysis_response()
        - LLMOntologyManager._parse_improvement_response()
        """
        if not response_content or not response_content.strip():
            self.logger.warning("Empty response content")
            return None

        # Clean response content
        cleaned_content = self._clean_json_response(response_content)

        try:
            parsed_data = json.loads(cleaned_content)

            # Validate against response models if provided
            if response_model and hasattr(response_model, '__annotations__'):
                validated_data = self._validate_response_model(parsed_data, response_model)
                return validated_data

            return parsed_data

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response content: {response_content[:500]}...")
            return None
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            return None

    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response by removing markdown fences and extra content."""
        content = content.strip()

        # Remove markdown code fences
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]

        if content.endswith('```'):
            content = content[:-3]

        content = content.strip()

        # Find JSON object boundaries
        start_idx = content.find('{')
        if start_idx == -1:
            start_idx = content.find('[')

        if start_idx != -1:
            # Find matching closing bracket
            bracket_count = 0
            end_idx = -1
            start_char = content[start_idx]
            end_char = '}' if start_char == '{' else ']'

            for i in range(start_idx, len(content)):
                if content[i] == start_char:
                    bracket_count += 1
                elif content[i] == end_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break

            if end_idx != -1:
                content = content[start_idx:end_idx]

        return content

    def _validate_response_model(self,
                                 data: Dict[str, Any],
                                 response_model: Type[Any]) -> Dict[str, Any]:
        """Validate parsed data against response models."""
        if hasattr(response_model, 'model_validate'):
            # Pydantic models
            try:
                validated = response_model.model_validate(data)
                return validated.to_dict()
            except Exception as e:
                self.logger.warning(f"Pydantic validation failed: {e}")
                return data

        elif hasattr(response_model, '__annotations__'):
            # Dataclass or similar
            try:
                # Basic validation - check required fields exist
                annotations = getattr(response_model, '__annotations__', {})
                for field_name, field_type in annotations.items():
                    if field_name not in data:
                        # Check if field has default value
                        if hasattr(response_model, field_name):
                            continue
                        self.logger.warning(f"Missing required field: {field_name}")

                return data
            except Exception as e:
                self.logger.warning(f"Dataclass validation failed: {e}")
                return data

        return data

    def parse_ontology_analysis(self, response: str, ontology_id: str) -> Dict[str, Any]:
        """
        Parse ontology analysis response.
        
        Consolidates LLMOntologyManager._parse_analysis_response()
        """
        parsed_data = self.parse_json_response(response)

        if not parsed_data:
            # Return default analysis structure
            return {
                "ontology_id": ontology_id,
                "analysis_date": datetime.now(UTC).isoformat(),
                "coverage_score": 0.5,
                "consistency_score": 0.5,
                "completeness_score": 0.5,
                "overall_quality": 0.5,
                "gaps_identified": ["Analysis parsing failed"],
                "improvement_suggestions": ["Retry analysis"],
                "new_entity_types": [],
                "new_relationship_types": [],
                "rule_suggestions": [],
                "confidence": 0.1
            }

        # Ensure required fields with defaults
        analysis = {
            "ontology_id": ontology_id,
            "analysis_date": datetime.now(UTC).isoformat(),
            "coverage_score": parsed_data.get("coverage_score", 0.0),
            "consistency_score": parsed_data.get("consistency_score", 0.0),
            "completeness_score": parsed_data.get("completeness_score", 0.0),
            "overall_quality": parsed_data.get("overall_quality", 0.0),
            "gaps_identified": parsed_data.get("gaps_identified", []),
            "improvement_suggestions": parsed_data.get("improvement_suggestions", []),
            "new_entity_types": parsed_data.get("new_entity_types", []),
            "new_relationship_types": parsed_data.get("new_relationship_types", []),
            "rule_suggestions": parsed_data.get("rule_suggestions", []),
            "confidence": parsed_data.get("confidence", 0.0)
        }

        return analysis

    def parse_improvement_plan(self, response: str, ontology_id: str) -> Dict[str, Any]:
        """
        Parse improvement plan response.
        
        Consolidates LLMOntologyManager._parse_improvement_response()
        """
        parsed_data = self.parse_json_response(response)

        if not parsed_data:
            # Return default plan structure
            return {
                "ontology_id": ontology_id,
                "created_date": datetime.now(UTC).isoformat(),
                "priority": "low",
                "changes": [],
                "rationale": "Parsing failed",
                "estimated_impact": "Unknown",
                "risk_assessment": "high",
                "approval_required": True
            }

        # Ensure required fields with defaults
        plan = {
            "ontology_id": ontology_id,
            "created_date": datetime.now(UTC).isoformat(),
            "priority": parsed_data.get("priority", "medium"),
            "changes": parsed_data.get("changes", []),
            "rationale": parsed_data.get("rationale", ""),
            "estimated_impact": parsed_data.get("estimated_impact", ""),
            "risk_assessment": parsed_data.get("risk_assessment", "medium"),
            "approval_required": parsed_data.get("approval_required", True)
        }

        return plan

    def parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse validation response.
        
        Consolidates LLMOntologyManager._parse_validation_response()
        """
        parsed_data = self.parse_json_response(response)

        if not parsed_data:
            return {
                "status": "error",
                "message": "Failed to parse validation response"
            }

        return parsed_data

    def parse_enrichment_response(self, response: str) -> list:
        """
        Parse domain enrichment response.
        
        Consolidates LLMOntologyManager._parse_enrichment_response()
        """
        parsed_data = self.parse_json_response(response)

        if not parsed_data:
            return []

        return parsed_data.get("enrichments", [])
