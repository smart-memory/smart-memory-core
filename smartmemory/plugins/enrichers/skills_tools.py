import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata


@dataclass
class ExtractSkillsToolsEnricherConfig(MemoryBaseModel):
    enable_skills: bool = True
    enable_tools: bool = True


@dataclass
class ExtractSkillsToolsEnricherRequest(StageRequest):
    enable_skills: bool = True
    enable_tools: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class ExtractSkillsToolsEnricher(EnricherPlugin):
    """
    Enricher: Extracts skills/tools from MemoryItem content and adds to metadata.
    Simple heuristic: looks for keywords like 'use', 'used', 'using', 'skill', 'tool', etc.
    Extend with more advanced extraction as needed.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="extract_skills_tools",
            version="1.0.0",
            author="SmartMemory Team",
            description="Heuristic-based extraction of skills and tools from content",
            plugin_type="enricher",
            dependencies=[],
            min_smartmemory_version="0.1.0"
        )

    def __init__(self, config: Optional[ExtractSkillsToolsEnricherConfig] = None):
        self.config = config or ExtractSkillsToolsEnricherConfig()

    def enrich(self, item, node_ids=None):
        if not isinstance(self.config, ExtractSkillsToolsEnricherConfig):
            raise TypeError("ExtractSkillsToolsEnricher requires a typed config (ExtractSkillsToolsEnricherConfig)")
        memory_id = getattr(item, 'item_id', None)
        with trace_span("pipeline.enrich.extract_skills_tools", {"memory_id": memory_id}):
            content = getattr(item, 'content', '') or ''
            metadata = getattr(item, 'metadata', {})
            # Heuristic: look for 'used <tool>' or 'using <tool>'
            tool_pattern = re.compile(r"(?:use|using|used) ([a-zA-Z0-9_\-]+)", re.IGNORECASE)
            skill_pattern = re.compile(r"skill[s]?: ([a-zA-Z0-9_,\- ]+)", re.IGNORECASE)
            found_tools = tool_pattern.findall(content) if self.config.enable_tools else []
            found_skills = []
            if self.config.enable_skills:
                for match in skill_pattern.findall(content):
                    found_skills.extend([s.strip() for s in match.split(',') if s.strip()])
            # Merge with any existing
            tools = set(metadata.get('tools', [])) | set(found_tools)
            skills = set(metadata.get('skills', [])) | set(found_skills)
            if tools:
                metadata['tools'] = sorted(tools)
            if skills:
                metadata['skills'] = sorted(skills)
            item.metadata = metadata
        return {'skills': list(skills), 'tools': list(tools)}
