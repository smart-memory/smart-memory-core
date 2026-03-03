"""
Utility functions for ingestion pipeline.

This module contains sanitization, normalization, and helper functions
used throughout the ingestion pipeline.
"""
import re
from datetime import datetime, UTC
from typing import Optional, Tuple, Any, Dict


def sanitize_relation_type(relation_type: str) -> str:
    """
    Sanitize relationship type by replacing invalid characters.
    
    Args:
        relation_type: Raw relation type string
        
    Returns:
        Sanitized relation type suitable for graph database storage
    """
    if not relation_type:
        return "UNKNOWN"

    # Convert to uppercase and sanitize
    p = str(relation_type).upper()
    p = re.sub(r'[^A-Z0-9]+', '_', p)  # non-allowed → underscore
    p = re.sub(r'_{2,}', '_', p)  # collapse repeats
    p = p.strip('_')[:50]  # trim & cut to 50 chars max

    # Ensure it starts with a letter (FalkorDB requirement)
    if not p or not p[0].isalpha():
        p = f"REL_{p}" if p else "UNKNOWN"

    return p


def normalize_entity_name(entity_name: str) -> str:
    """
    Normalize entity name for consistent storage and retrieval.
    
    Args:
        entity_name: Raw entity name
        
    Returns:
        Normalized entity name
    """
    if not entity_name:
        return ""

    # Basic normalization: strip whitespace and convert to consistent case
    normalized = str(entity_name).strip()

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized


def infer_temporal_metadata(entity: Optional[Dict[str, Any]] = None,
                            enrichment_result: Optional[Dict[str, Any]] = None) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
    """
    Infer temporal metadata (valid_start, valid_end, transaction_time) from entity and enrichment data.
    
    Args:
        entity: Entity data dictionary
        enrichment_result: Enrichment result containing temporal information
        
    Returns:
        Tuple of (valid_start, valid_end, transaction_time)
    """
    valid_start = None
    valid_end = None
    tx_time = datetime.now(UTC)

    # Try to extract temporal information from enrichment
    if enrichment_result:
        temporal_data = enrichment_result.get('temporal', {})
        if isinstance(temporal_data, dict):
            # Extract start/end dates if available
            start_str = temporal_data.get('start_date') or temporal_data.get('valid_start')
            end_str = temporal_data.get('end_date') or temporal_data.get('valid_end')

            if start_str:
                try:
                    valid_start = datetime.fromisoformat(start_str) if isinstance(start_str, str) else start_str
                except (ValueError, TypeError):
                    pass

            if end_str:
                try:
                    valid_end = datetime.fromisoformat(end_str) if isinstance(end_str, str) else end_str
                except (ValueError, TypeError):
                    pass

    # Try to extract from entity metadata
    if entity and isinstance(entity, dict):
        metadata = entity.get('metadata', {})
        if isinstance(metadata, dict):
            created_at = metadata.get('created_at') or metadata.get('timestamp')
            if created_at:
                try:
                    valid_start = datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at
                except (ValueError, TypeError):
                    pass

    return valid_start, valid_end, tx_time


def extract_payload_for_instrumentation(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract payload data for instrumentation from ingestion context.
    
    Args:
        context: Ingestion context dictionary
        
    Returns:
        Payload dictionary for instrumentation
    """
    try:
        return {
            'entities_count': len(context.get('entities', [])),
            'relations_count': len(context.get('relations', [])),
            'item_id': context.get('item_id'),
            'memory_type': context.get('memory_type', 'semantic'),
            'extractor': context.get('extractor', 'unknown'),
            'adapter': context.get('adapter', 'unknown')
        }
    except Exception:
        return {'error': 'Failed to extract payload'}


def validate_triple(triple: Tuple[Any, Any, Any]) -> bool:
    """
    Validate that a triple has the correct structure.
    
    Args:
        triple: Triple tuple (subject, predicate, object)
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(triple, (tuple, list)) or len(triple) != 3:
        return False

    subject, predicate, obj = triple

    # All stages should be non-empty strings or have string representation
    if not all(str(component).strip() for component in [subject, predicate, obj]):
        return False

    return True


def clean_text_content(content: str) -> str:
    """
    Clean and normalize text content for processing.
    
    Args:
        content: Raw text content
        
    Returns:
        Cleaned text content
    """
    if not content:
        return ""

    # Convert to string if not already
    text = str(content)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def safe_get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: Dictionary to traverse
        *keys: Keys to traverse in order
        default: Default value if key path doesn't exist
        
    Returns:
        Value at nested key path or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def merge_metadata(base_metadata: Dict[str, Any], additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge metadata dictionaries with conflict resolution.
    
    Args:
        base_metadata: Base metadata dictionary
        additional_metadata: Additional metadata to merge
        
    Returns:
        Merged metadata dictionary
    """
    if not base_metadata:
        return additional_metadata.copy() if additional_metadata else {}

    if not additional_metadata:
        return base_metadata.copy()

    merged = base_metadata.copy()

    for key, value in additional_metadata.items():
        if key in merged:
            # Handle conflicts - prefer non-empty values
            if not merged[key] and value:
                merged[key] = value
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_metadata(merged[key], value)
            # Otherwise keep the base value
        else:
            merged[key] = value

    return merged
