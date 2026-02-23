"""
Reasoning Trace Extractor for System 2 Memory 

Extracts chain-of-thought reasoning traces from agent conversations,
enabling retrieval of "why" decisions were made, not just the outcomes.

Two extraction modes:
1. Explicit markup: Parses Thought:/Action:/Observation: markers
2. Implicit detection: Uses LLM to detect reasoning patterns


"""

import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem
from smartmemory.models.reasoning import (
    ReasoningStep, ReasoningTrace, ReasoningEvaluation, TaskContext
)
from smartmemory.observability.tracing import trace_span
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata
from smartmemory.utils.llm import call_llm

logger = logging.getLogger(__name__)


# Explicit markers for reasoning steps (case-insensitive)
EXPLICIT_MARKERS = {
    'thought': [r'(?:^|\n)\s*(?:thought|thinking|i think|let me think):\s*', r'(?:^|\n)\s*💭\s*'],
    'action': [r'(?:^|\n)\s*(?:action|doing|let me):\s*', r'(?:^|\n)\s*🔧\s*'],
    'observation': [r'(?:^|\n)\s*(?:observation|result|output|i see|i notice):\s*', r'(?:^|\n)\s*👁\s*'],
    'decision': [r'(?:^|\n)\s*(?:decision|i will|i\'ll|decided):\s*', r'(?:^|\n)\s*✅\s*'],
    'conclusion': [r'(?:^|\n)\s*(?:conclusion|therefore|thus|in conclusion|finally):\s*', r'(?:^|\n)\s*🎯\s*'],
    'reflection': [r'(?:^|\n)\s*(?:reflection|looking back|in hindsight):\s*', r'(?:^|\n)\s*🔄\s*'],
}

# Implicit reasoning indicators (for LLM detection)
REASONING_INDICATORS = [
    'let me', 'i think', 'first', 'then', 'because', 'therefore',
    'considering', 'analyzing', 'the issue is', 'the problem',
    'step 1', 'step 2', 'approach', 'strategy', 'solution',
]


@dataclass
class ReasoningExtractorConfig(MemoryBaseModel):
    """Configuration for the reasoning extractor."""
    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2000
    
    # Quality thresholds
    min_quality_score: float = 0.4
    min_steps: int = 2  # Minimum steps to consider a valid trace
    
    # Detection settings
    use_llm_detection: bool = True  # Use LLM for implicit detection
    prefer_explicit_markup: bool = True  # Prefer explicit markers when found


class ReasoningExtractor(ExtractorPlugin):
    """
    Extracts reasoning traces from text content.
    
    Returns standard extractor format with reasoning trace in metadata:
    {
        'entities': [],  # No entities extracted (reasoning-focused)
        'relations': [],  # Relations added later via CAUSES edges
        'reasoning_trace': ReasoningTrace,  # The extracted trace
    }
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="reasoning",
            version="1.0.0",
            author="SmartMemory Team",
            description="Extracts chain-of-thought reasoning traces (System 2 memory)",
            plugin_type="extractor",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["reasoning", "chain-of-thought", "system2", "llm"]
        )
    
    def __init__(self, config: Optional[ReasoningExtractorConfig] = None):
        self.cfg = config or ReasoningExtractorConfig()
    
    def extract(self, text: str) -> dict:
        """
        Extract reasoning trace from text.

        Returns:
            dict with 'entities', 'relations', and 'reasoning_trace' keys
        """
        with trace_span("pipeline.extract.reasoning", {"text_length": len(text)}):
            result = {
                'entities': [],
                'relations': [],
                'reasoning_trace': None,
            }

            if not text or len(text.strip()) < 50:
                return result

            # 1. Try explicit markup extraction first
            trace = None
            has_explicit = self._has_explicit_markers(text)

            if has_explicit and self.cfg.prefer_explicit_markup:
                trace = self._extract_explicit(text)

            # 2. Fall back to LLM detection if no explicit markers or extraction failed
            if trace is None and self.cfg.use_llm_detection:
                if self._likely_contains_reasoning(text):
                    trace = self._extract_implicit(text)

            # 3. Evaluate quality
            if trace and trace.steps:
                trace.evaluation = self._evaluate_trace(trace)

                # Only include if passes quality threshold
                if trace.evaluation.should_store:
                    result['reasoning_trace'] = trace
                else:
                    logger.debug(f"Trace rejected: quality={trace.evaluation.quality_score:.2f}")

            return result
    
    def _has_explicit_markers(self, text: str) -> bool:
        """Check if text contains explicit reasoning markers."""
        text_lower = text.lower()
        for step_type, patterns in EXPLICIT_MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
        return False
    
    def _likely_contains_reasoning(self, text: str) -> bool:
        """Quick heuristic check for reasoning content."""
        text_lower = text.lower()
        indicator_count = sum(1 for ind in REASONING_INDICATORS if ind in text_lower)
        return indicator_count >= 2
    
    def _extract_explicit(self, text: str) -> Optional[ReasoningTrace]:
        """Extract reasoning steps from explicit markers."""
        steps = []
        
        # Find all markers and their positions
        markers_found: List[Tuple[int, str, str]] = []  # (position, type, content_start)
        
        for step_type, patterns in EXPLICIT_MARKERS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    markers_found.append((match.end(), step_type, match.group()))
        
        if not markers_found:
            return None
        
        # Sort by position
        markers_found.sort(key=lambda x: x[0])
        
        # Extract content between markers
        for i, (pos, step_type, _) in enumerate(markers_found):
            # Find end of this step's content
            if i + 1 < len(markers_found):
                end_pos = markers_found[i + 1][0] - len(markers_found[i + 1][2])
            else:
                end_pos = len(text)
            
            content = text[pos:end_pos].strip()
            
            # Clean up content (remove trailing markers)
            content = re.sub(r'\n\s*(?:thought|action|observation|decision|conclusion|reflection):\s*$', '', content, flags=re.IGNORECASE)
            
            if content:
                steps.append(ReasoningStep(type=step_type, content=content[:1000]))  # type: ignore  # Truncate long content
        
        if len(steps) < self.cfg.min_steps:
            return None
        
        trace_id = f"trace_{hashlib.sha256(text[:500].encode()).hexdigest()[:12]}"
        
        return ReasoningTrace(
            trace_id=trace_id,
            steps=steps,
            has_explicit_markup=True,
            task_context=self._infer_task_context(text, steps),
        )
    
    def _extract_implicit(self, text: str) -> Optional[ReasoningTrace]:
        """Use LLM to extract implicit reasoning steps."""
        try:
            api_key = os.getenv(self.cfg.api_key_env)
            if not api_key:
                logger.warning(f"No API key found for implicit reasoning extraction")
                return None
            
            prompt = self._build_extraction_prompt(text)
            
            _, response = call_llm(
                user_content=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                model=self.cfg.model_name,
                api_key=api_key,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
            )
            
            if not response:
                return None
            
            return self._parse_llm_response(response, text)
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for LLM extraction."""
        # Truncate very long texts
        truncated = text[:4000] if len(text) > 4000 else text
        
        return f"""Analyze the following text and extract the reasoning steps.

TEXT:
{truncated}

Extract each distinct reasoning step and classify it as one of:
- thought: Internal reasoning or consideration
- action: Action taken or proposed
- observation: Result or observation from an action
- decision: Decision point reached
- conclusion: Final conclusion or answer
- reflection: Meta-reasoning about the process

Return as JSON array of objects with "type" and "content" fields.
Only include actual reasoning steps, not general statements.
If no clear reasoning is present, return an empty array."""
    
    def _parse_llm_response(self, response: str, original_text: str) -> Optional[ReasoningTrace]:
        """Parse LLM response into ReasoningTrace."""
        import json
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                return None
            
            steps_data = json.loads(json_match.group())
            
            if not isinstance(steps_data, list) or len(steps_data) < self.cfg.min_steps:
                return None
            
            steps = []
            valid_types = {'thought', 'action', 'observation', 'decision', 'conclusion', 'reflection'}
            
            for step_data in steps_data:
                step_type = step_data.get('type', 'thought').lower()
                if step_type not in valid_types:
                    step_type = 'thought'
                
                content = step_data.get('content', '')
                if content:
                    steps.append(ReasoningStep(type=step_type, content=content[:1000]))  # type: ignore
            
            if len(steps) < self.cfg.min_steps:
                return None
            
            trace_id = f"trace_{hashlib.sha256(original_text[:500].encode()).hexdigest()[:12]}"
            
            return ReasoningTrace(
                trace_id=trace_id,
                steps=steps,
                has_explicit_markup=False,
                task_context=self._infer_task_context(original_text, steps),
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
    
    def _infer_task_context(self, text: str, steps: List[ReasoningStep]) -> TaskContext:
        """Infer task context from text and steps."""
        text_lower = text.lower()
        
        # Infer task type
        task_type = None
        if any(kw in text_lower for kw in ['bug', 'fix', 'error', 'issue', 'debug']):
            task_type = 'debugging'
        elif any(kw in text_lower for kw in ['create', 'implement', 'build', 'write']):
            task_type = 'code_generation'
        elif any(kw in text_lower for kw in ['analyze', 'review', 'explain', 'understand']):
            task_type = 'analysis'
        elif any(kw in text_lower for kw in ['solve', 'problem', 'challenge']):
            task_type = 'problem_solving'
        
        # Infer domain
        domain = None
        if any(kw in text_lower for kw in ['python', '.py', 'def ', 'import ']):
            domain = 'python'
        elif any(kw in text_lower for kw in ['javascript', 'typescript', '.js', '.ts', 'const ', 'let ']):
            domain = 'javascript'
        elif any(kw in text_lower for kw in ['react', 'component', 'jsx', 'tsx']):
            domain = 'frontend'
        elif any(kw in text_lower for kw in ['api', 'endpoint', 'server', 'backend']):
            domain = 'backend'
        elif any(kw in text_lower for kw in ['sql', 'database', 'query', 'table']):
            domain = 'database'
        
        # Infer complexity based on step count and content length
        total_content = sum(len(s.content) for s in steps)
        if len(steps) >= 6 or total_content > 2000:
            complexity = 'high'
        elif len(steps) >= 4 or total_content > 1000:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # Extract goal from first thought or conclusion
        goal = None
        for step in steps:
            if step.type in ('thought', 'conclusion') and len(step.content) > 20:
                goal = step.content[:200]
                break
        
        return TaskContext(
            goal=goal,
            input=text[:500] if len(text) > 500 else text,
            task_type=task_type,
            domain=domain,
            complexity=complexity,
        )
    
    def _evaluate_trace(self, trace: ReasoningTrace) -> ReasoningEvaluation:
        """Evaluate the quality of a reasoning trace."""
        issues = []
        suggestions = []
        
        # Check for loops (repeated content)
        has_loops = False
        contents = [s.content.lower()[:100] for s in trace.steps]
        if len(contents) != len(set(contents)):
            has_loops = True
            issues.append({
                'type': 'loop',
                'description': 'Repeated reasoning steps detected',
                'severity': 'medium'
            })
        
        # Check for redundancy (very similar steps)
        has_redundancy = False
        for i, c1 in enumerate(contents):
            for c2 in contents[i+1:]:
                # Simple Jaccard similarity
                words1 = set(c1.split())
                words2 = set(c2.split())
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity > 0.8:
                        has_redundancy = True
                        break
            if has_redundancy:
                break
        
        if has_redundancy:
            issues.append({
                'type': 'redundancy',
                'description': 'Very similar steps detected',
                'severity': 'low'
            })
        
        # Calculate step diversity
        step_types = set(s.type for s in trace.steps)
        step_diversity = len(step_types) / 6  # 6 possible types
        
        if step_diversity < 0.3:
            suggestions.append('Consider adding more diverse reasoning steps')
        
        # Check minimum content
        avg_content_len = sum(len(s.content) for s in trace.steps) / len(trace.steps)
        if avg_content_len < 30:
            issues.append({
                'type': 'shallow',
                'description': 'Steps have very short content',
                'severity': 'medium'
            })
        
        # Calculate quality score
        quality_score = 1.0
        
        # Penalize issues
        for issue in issues:
            if issue['severity'] == 'high':
                quality_score -= 0.4
            elif issue['severity'] == 'medium':
                quality_score -= 0.2
            else:
                quality_score -= 0.1
        
        # Bonus for diversity
        quality_score += step_diversity * 0.2
        
        # Bonus for step count (up to 6)
        quality_score += min(len(trace.steps), 6) * 0.05
        
        # Clamp to [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))
        
        return ReasoningEvaluation(
            quality_score=quality_score,
            has_loops=has_loops,
            has_redundancy=has_redundancy,
            step_diversity=step_diversity,
            issues=issues,
            suggestions=suggestions,
        )


# System prompt for LLM extraction
EXTRACTION_SYSTEM_PROMPT = """You are a reasoning trace extractor. Your job is to identify and extract distinct reasoning steps from text.

Focus on:
1. Explicit reasoning markers (Thought:, Action:, etc.)
2. Implicit reasoning patterns (let me think, first I'll, because, therefore)
3. Problem-solving sequences
4. Decision-making processes

Output ONLY a JSON array of reasoning steps. Each step should have:
- "type": one of "thought", "action", "observation", "decision", "conclusion", "reflection"
- "content": the actual reasoning content

Be precise and only extract actual reasoning, not general statements."""
