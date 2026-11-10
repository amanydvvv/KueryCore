"""
DeepEval judge wrapper — Groq-backed DeepEvalBaseLLM subclass.

Model: openai/gpt-oss-20b (Groq endpoint, OpenAI-compatible API)
Temperature: 0.0 (deterministic judgements)

This judge is used by the DeepEval metric classes in
backend/eval/run_deepeval.py and backend/tests/test_deepeval.py.
It is NOT the same as the internal _invoke_judge() in app/services/evaluation.py
which judges against the proprietary dimension schema. Both judge the same answers
but measure different things:

  - Internal harness (evaluation.py)  -> groundedness / completeness /
    retrieval_pass / topic_leak (rule-based + judge prompt)
  - DeepEval metrics (this module)    -> Faithfulness / AnswerRelevancy /
    ContextualPrecision / ContextualRecall (standard RAG metrics)
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    # Graceful fallback when deepeval is not installed (e.g. lean CI environments)
    class DeepEvalBaseLLM:  # type: ignore[no-redef]
        """Fallback base class when deepeval package is not installed."""
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name


class GroqJudge(DeepEvalBaseLLM):
    """
    Thin adapter that makes a Groq-hosted model available as a DeepEval judge.

    Usage::

        from eval.judge import GroqJudge
        judge = GroqJudge()
        # then pass as `model=judge` to any deepeval Metric constructor

    Environment variables required:
        GROQ_API_KEY   — Groq API key
    Optional:
        EVAL_JUDGE_MODEL — model string; defaults to "openai/gpt-oss-20b"
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model or os.environ.get(
            "EVAL_JUDGE_MODEL", "openai/gpt-oss-20b"
        )
        self.temperature = temperature
        super().__init__(model_name=self.model_name)

    # ------------------------------------------------------------------
    # DeepEvalBaseLLM interface (sync + async required by the base class)
    # ------------------------------------------------------------------

    def load_model(self):
        """Lazy import of groq client."""
        from groq import Groq

        return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

    def generate(self, prompt: str) -> str:
        """Synchronous prompt generation for evaluation metrics."""
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        """Asynchronous prompt generation for evaluation metrics."""
        from groq import AsyncGroq

        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", ""))
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        return self.model_name
