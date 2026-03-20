"""LLM-as-judge for memory retrieval relevance scoring."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from smartmemory.evaluation.dataset import Session

JUDGE_PROMPT = """\
You are evaluating a memory retrieval system used during code development.
Below is a sequence of search queries from one development session, with the
memories returned for each query.

For each query, rate the result set:
- 3 = results directly answer what the developer needed
- 2 = results are relevant but incomplete
- 1 = results are tangentially related
- 0 = results are irrelevant or unhelpful

Also flag if results contain information the developer likely already stated
in an earlier query in this session (indicates the system retrieved stale/redundant context).

{session_text}

Return ONLY valid JSON with no extra text:
{{"scores": [{{"query_index": 0, "score": 3, "redundant": false}}, ...]}}"""


@dataclass
class JudgeScore:
    """Score for a single query in a session."""

    query_index: int
    score: int
    redundant: bool


@dataclass
class SessionJudgment:
    """Judgment for an entire session."""

    session_index: int
    scores: list[JudgeScore]

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


def _format_session_for_judge(session: Session) -> str:
    """Format a session's interactions into text for the judge prompt."""
    parts: list[str] = []
    for i, interaction in enumerate(session.interactions):
        parts.append(f"--- Query {i} ---")
        parts.append(f"Query: {interaction.query}")
        if interaction.results:
            parts.append(f"Results ({len(interaction.results)}):")
            for r in interaction.results:
                snippet = r.get("content_snippet", "")
                score = r.get("score")
                score_str = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                parts.append(f"  - [{r.get('memory_type', '?')}]{score_str} {snippet}")
        else:
            parts.append("Results: (none)")
        parts.append("")
    return "\n".join(parts)


def _parse_judge_response(text: str) -> list[JudgeScore]:
    """Parse the judge's JSON response into JudgeScore objects."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)
    scores: list[JudgeScore] = []
    for entry in data.get("scores", []):
        scores.append(JudgeScore(
            query_index=entry["query_index"],
            score=entry["score"],
            redundant=entry.get("redundant", False),
        ))
    return scores


class RelevanceJudge:
    """Judges memory retrieval relevance using an LLM.

    Supports OpenAI-compatible APIs (OpenAI, Groq, DeepSeek) via the openai package.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")
        self.api_key = api_key
        self.base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create an OpenAI client, auto-detecting provider from model name."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for LLM judging: pip install openai")

        if self.base_url:
            self._client = OpenAI(api_key=self.api_key or "dummy", base_url=self.base_url)
        elif "groq" in self.model.lower() or self.model.lower().startswith(("llama", "mixtral")):
            self._client = OpenAI(
                api_key=self.api_key or os.environ.get("GROQ_API_KEY", ""),
                base_url="https://api.groq.com/openai/v1",
            )
        elif "deepseek" in self.model.lower():
            self._client = OpenAI(
                api_key=self.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com",
            )
        else:
            self._client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""))

        return self._client

    def judge_session(self, session: Session, session_index: int = 0) -> SessionJudgment:
        """Judge the relevance of results for all queries in a session.

        Args:
            session: The session to judge.
            session_index: Index of this session in the evaluation run.

        Returns:
            SessionJudgment with per-query scores.
        """
        if not session.interactions:
            return SessionJudgment(session_index=session_index, scores=[])

        session_text = _format_session_for_judge(session)
        prompt = JUDGE_PROMPT.format(session_text=session_text)

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            scores = _parse_judge_response(raw)
        except (json.JSONDecodeError, KeyError):
            # Fallback: return empty scores rather than crash
            scores = []

        return SessionJudgment(session_index=session_index, scores=scores)

    def judge_sessions(self, sessions: list[Session]) -> list[SessionJudgment]:
        """Judge all sessions sequentially.

        Args:
            sessions: List of sessions to judge.

        Returns:
            List of SessionJudgment objects.
        """
        return [self.judge_session(s, i) for i, s in enumerate(sessions)]
