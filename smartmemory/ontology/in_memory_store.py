"""In-memory ontology store — unblocks OntologyConstrainStage on SQLite.

Implements the minimal interface needed for the pipeline to run:
get_type_status, add_provisional, increment_frequency, promote.
All other OntologyGraph methods are stubbed as no-ops.

Seeded with 14 base entity types on init. State lives in memory —
no persistence across restarts. Full SQLiteOntologyStore (DIST-FULL-LOCAL-1
Phase 2) will add persistence and the remaining 23 methods.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEED_TYPES = [
    "Person", "Organization", "Technology", "Concept", "Event",
    "Location", "Document", "Tool", "Skill", "Decision",
    "Claim", "Action", "Metric", "Process",
]


class InMemoryOntologyStore:
    """Minimal ontology store that unblocks OntologyConstrainStage."""

    def __init__(self, workspace_id: str = "default"):
        self.workspace_id = workspace_id
        self._types: Dict[str, Dict[str, Any]] = {}
        self._relation_types: Dict[str, Dict[str, Any]] = {}
        self.seed_types()

    # ── Entity Type Management (pipeline-critical) ──────────────────

    def seed_types(self, types: Optional[List[str]] = None) -> int:
        types_to_seed = types or SEED_TYPES
        created = 0
        for name in types_to_seed:
            key = name.lower()
            if key not in self._types:
                self._types[key] = {
                    "name": name,
                    "status": "seed",
                    "frequency": 0,
                    "avg_confidence": 0.0,
                }
                created += 1
        return created

    def get_type_status(self, name: str) -> Optional[str]:
        entry = self._types.get(name.lower())
        return entry["status"] if entry else None

    def add_provisional(self, name: str) -> bool:
        key = name.lower()
        if key in self._types:
            return False
        self._types[key] = {
            "name": name,
            "status": "provisional",
            "frequency": 0,
            "avg_confidence": 0.0,
        }
        return True

    def promote(self, name: str) -> bool:
        entry = self._types.get(name.lower())
        if not entry:
            return False
        entry["status"] = "confirmed"
        return True

    def reject(self, name: str) -> bool:
        entry = self._types.get(name.lower())
        if not entry:
            return False
        entry["status"] = "rejected"
        return True

    def increment_frequency(self, name: str, confidence: float) -> None:
        entry = self._types.get(name.lower())
        if not entry:
            return
        old_freq = entry["frequency"]
        old_avg = entry["avg_confidence"]
        entry["frequency"] = old_freq + 1
        # Running average
        entry["avg_confidence"] = (old_avg * old_freq + confidence) / (old_freq + 1)

    def get_frequency(self, name: str) -> int:
        entry = self._types.get(name.lower())
        return entry["frequency"] if entry else 0

    def get_entity_types(self) -> List[Dict[str, Any]]:
        return list(self._types.values())

    def get_all_frequencies(self) -> Dict[str, int]:
        return {v["name"]: v["frequency"] for v in self._types.values()}

    # ── Relation Types (pipeline uses increment + add_provisional) ──

    def increment_relation_frequency(self, name: str, confidence: float) -> None:
        key = name.lower()
        if key not in self._relation_types:
            self._relation_types[key] = {"name": name, "status": "provisional", "frequency": 0}
        self._relation_types[key]["frequency"] += 1

    def add_provisional_relation_type(self, name: str, category: str = "unknown") -> bool:
        key = name.lower()
        if key in self._relation_types:
            return False
        self._relation_types[key] = {"name": name, "status": "provisional", "category": category, "frequency": 0}
        return True

    def get_relation_types(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            return [v for v in self._relation_types.values() if v.get("status") == status]
        return list(self._relation_types.values())

    # ── Stubs (not needed for pipeline, no-op until Phase 2) ────────

    def add_entity_pattern(self, name, label, confidence=0.85, **kwargs) -> bool:
        return False  # JSONLPatternStore handles patterns on local

    def get_entity_patterns(self, workspace_id=None) -> List[Dict]:
        return []  # JSONLPatternStore handles this

    def seed_entity_patterns(self, patterns=None) -> int:
        return 0

    def get_pattern_stats(self) -> Dict[str, int]:
        return {}

    def get_type_assignments(self, entity_name: str) -> List[Dict]:
        return []

    def delete_entity_pattern(self, name, label, workspace_id=None) -> bool:
        return False

    def seed_relation_types(self) -> int:
        return 0

    def seed_type_pair_edges(self) -> int:
        return 0

    def add_novel_relation_label(self, name, **kwargs) -> bool:
        return False

    def get_novel_relation_labels(self, min_frequency=1, status=None) -> List[Dict]:
        return []

    def promote_relation_type(self, name, **kwargs) -> bool:
        return False

    def save_pipeline_config(self, name, config_json, workspace_id=None, description="") -> bool:
        return False

    def get_pipeline_config(self, name, workspace_id=None) -> Optional[Dict]:
        return None

    def list_pipeline_configs(self, workspace_id=None) -> List[Dict]:
        return []

    def delete_pipeline_config(self, name, workspace_id=None) -> bool:
        return False
