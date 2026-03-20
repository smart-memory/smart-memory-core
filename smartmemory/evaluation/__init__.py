"""SmartMemory Evaluation — offline quality measurement for memory retrieval.

Reads interaction logs (JSONL), groups into sessions, judges relevance via LLM,
and computes metrics (relevance@k, hit rate, repetition rate).
"""

from smartmemory.evaluation.dataset import InteractionLog, Session, load_interactions, group_sessions
from smartmemory.evaluation.metrics import relevance_at_k, hit_rate, repetition_rate, mean_latency, redundancy_rate
from smartmemory.evaluation.runner import EvalRunner, EvalResult

__all__ = [
    "InteractionLog",
    "Session",
    "load_interactions",
    "group_sessions",
    "relevance_at_k",
    "hit_rate",
    "repetition_rate",
    "mean_latency",
    "redundancy_rate",
    "EvalRunner",
    "EvalResult",
]
