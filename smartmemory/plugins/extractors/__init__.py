from .llm import LLMExtractor
from .llm_single import LLMSingleExtractor, GroqExtractor
from .reasoning import ReasoningExtractor
from .spacy import SpacyExtractor

__all__ = [
    # Primary extractors
    'GroqExtractor',          # Default: Groq Llama-3.3 (100% E-F1, 89.3% R-F1, 878ms)
    'LLMExtractor',           # Pure LLM (two-call for precision)
    'LLMSingleExtractor',     # Fast LLM (single-call, configurable model)
    'ReasoningExtractor',     # System 2: Chain-of-thought traces
    'SpacyExtractor',         # Zero-dependency fallback: no API key, powers EntityRuler stage
]
