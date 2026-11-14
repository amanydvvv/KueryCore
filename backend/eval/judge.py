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

import json
import os
import re
from typing import Any, Optional

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    # Graceful fallback when deepeval is not installed (e.g. lean CI environments)
    class DeepEvalBaseLLM:  # type: ignore[no-redef]
        """Fallback base class when deepeval package is not installed."""

        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_schema_output(raw: str, schema: Any) -> Any:
    """Parse raw LLM text into the requested Pydantic schema."""
    text = raw.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return schema.model_validate_json(text)
    except Exception:
        # Attempt substring extraction if the model wrapped JSON in prose
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            return schema.model_validate(json.loads(candidate))
        raise


def _build_prompt(prompt: str, schema: Any) -> str:
    """Append JSON-schema instruction only when DeepEval requests structured output."""
    if schema is None:
        return prompt
    schema_json = ""
    if hasattr(schema, "model_json_schema"):
        schema_json = json.dumps(schema.model_json_schema())
    return (
        f"{prompt}\n\n"
        f"IMPORTANT: Respond ONLY with a valid JSON object matching this JSON schema "
        f"(no markdown fences, no commentary): {schema_json}"
    )


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

    def generate(self, prompt: str, schema: Optional[Any] = None) -> Any:
        """Synchronous prompt generation for evaluation metrics."""
        client = self.load_model()
        final_prompt = _build_prompt(prompt, schema)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": self.temperature,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if schema is not None:
            return _parse_schema_output(content, schema)
        return content

    async def a_generate(self, prompt: str, schema: Optional[Any] = None) -> Any:
        """Asynchronous prompt generation for evaluation metrics."""
        from groq import AsyncGroq

        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", ""))
        final_prompt = _build_prompt(prompt, schema)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": self.temperature,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        if schema is not None:
            return _parse_schema_output(content, schema)
        return content

    def get_model_name(self) -> str:
        return self.model_name
