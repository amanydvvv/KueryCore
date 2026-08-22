"""
KueryCore AI — Agentic Intent Router (Phase 1)

Classifies user queries into distinct execution paths to optimize retrieval depth
and generation strategy:
  1. FACTUAL_LOOKUP: Pinpointed single-document or specific fact queries (default top_k=5).
  2. COMPARISON: Cross-document / cross-topic synthesis queries (top_k=10).
  3. SUMMARIZATION: Broad document overviews / key takeaway queries (top_k=12).
  4. CHITCHAT: Conversational, greetings, meta-questions (skip retrieval entirely).

Contract (fail-closed): Any timeout, LLM failure, or low-confidence classification
gracefully degrades to standard FACTUAL_LOOKUP with default top_k.
"""

import asyncio
import json
import logging
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.generation import get_llm

logger = logging.getLogger(__name__)

# Timeout ceiling for the routing decision to prevent added latency.
ROUTER_TIMEOUT_SECONDS = 1.8


class IntentType(str, Enum):
    """Supported query intent categories for agentic RAG routing."""
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    CHITCHAT = "chitchat"


class RouterDecision(BaseModel):
    """Structured decision output from the Intent Router."""
    intent: IntentType = Field(default=IntentType.FACTUAL_LOOKUP)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="Standard document question")
    suggested_top_k: int = Field(default=5, ge=0, le=20)
    skip_retrieval: bool = Field(default=False)



# Fast heuristics for zero-latency classification of common queries
_GREETING_RE = re.compile(
    r"^(hi|hello|hey(\s+there)?|good\s+(morning|afternoon|evening)|howdy|greetings|who\s+are\s+you|what\s+can\s+you\s+do|thanks|thank\s+you|bye|goodbye)[\s\!\.\?]*$",
    re.IGNORECASE,
)


_SUMMARIZE_RE = re.compile(
    r"^(summarize|give\s+me\s+a\s+summary|summarise|overview|key\s+takeaways|tldr|what\s+is\s+this\s+(document|pdf|file)\s+about)\b",
    re.IGNORECASE,
)

_COMPARE_RE = re.compile(
    r"\b(compare|difference\s+between|differences\s+between|versus|\bvs\b|contrast|how\s+does\s+.*compare)\b",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> Optional[RouterDecision]:
    """Fast regex-based intent classification for instant response on obvious patterns."""
    q = query.strip()
    if not q:
        return RouterDecision(
            intent=IntentType.CHITCHAT,
            confidence=1.0,
            reasoning="Empty input",
            suggested_top_k=0,
            skip_retrieval=True,
        )

    if _GREETING_RE.match(q):
        return RouterDecision(
            intent=IntentType.CHITCHAT,
            confidence=1.0,
            reasoning="Greeting or conversational chitchat",
            suggested_top_k=0,
            skip_retrieval=True,
        )

    if _SUMMARIZE_RE.search(q):
        return RouterDecision(
            intent=IntentType.SUMMARIZATION,
            confidence=0.95,
            reasoning="Explicit summarization request",
            suggested_top_k=12,
            skip_retrieval=False,
        )

    if _COMPARE_RE.search(q):
        return RouterDecision(
            intent=IntentType.COMPARISON,
            confidence=0.9,
            reasoning="Explicit comparative analysis request",
            suggested_top_k=10,
            skip_retrieval=False,
        )

    return None


ROUTER_PROMPT_TEMPLATE = (
    "You are an intent-classification router for a Document Q&A RAG engine.\n"
    "Categorize the following user query into exactly ONE intent:\n"
    "- 'factual_lookup': Asking for specific facts, numbers, terms, or definitions from documents.\n"
    "- 'comparison': Comparing two or more items, options, metrics, or documents.\n"
    "- 'summarization': Asking for a general summary, main points, or overview of documents.\n"
    "- 'chitchat': Greetings, compliments, meta-questions about the AI, or non-document conversation.\n"
    "\n"
    "USER QUERY: {query}\n"
    "\n"
    "Respond with ONLY a valid JSON object matching this schema:\n"
    '{{"intent": "factual_lookup" | "comparison" | "summarization" | "chitchat", "confidence": 0.95, "reasoning": "brief explanation"}}\n'
)


def _extract_json_payload(raw: str) -> Optional[dict]:
    """Parse JSON object from LLM response with fallback scanner."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (ValueError, TypeError):
                pass
    return None


async def classify_intent(
    query: str,
    chat_history: Optional[List] = None,
    default_top_k: int = 5,
) -> RouterDecision:
    """
    Classify the user's intent to dynamically adapt retrieval strategy.

    Falls back to IntentType.FACTUAL_LOOKUP if router is disabled, times out,
    or encounters any error.
    """
    settings = get_settings()
    if not settings.AGENTIC_ROUTER_ENABLED:
        return RouterDecision(
            intent=IntentType.FACTUAL_LOOKUP,
            confidence=1.0,
            reasoning="Router disabled via settings",
            suggested_top_k=default_top_k,
            skip_retrieval=False,
        )

    # 1. Check fast heuristics (0ms latency)
    fast_match = _heuristic_classify(query)
    if fast_match is not None:
        return fast_match

    # 2. Call fast LLM classifier with bounded timeout
    try:
        llm = get_llm(temperature=0.0)
        prompt = ROUTER_PROMPT_TEMPLATE.format(query=query)

        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=ROUTER_TIMEOUT_SECONDS,
        )

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        
        data = _extract_json_payload(str(content))
        if not data or "intent" not in data:
            return RouterDecision(
                intent=IntentType.FACTUAL_LOOKUP,
                confidence=0.5,
                reasoning="Malformed classifier JSON, defaulted to factual_lookup",
                suggested_top_k=default_top_k,
                skip_retrieval=False,
            )

        intent_str = str(data["intent"]).lower().strip()
        confidence = float(data.get("confidence", 0.8))
        reasoning = str(data.get("reasoning", ""))

        if intent_str == "chitchat":
            return RouterDecision(
                intent=IntentType.CHITCHAT,
                confidence=confidence,
                reasoning=reasoning or "Conversational query",
                suggested_top_k=0,
                skip_retrieval=True,
            )
        elif intent_str == "summarization":
            return RouterDecision(
                intent=IntentType.SUMMARIZATION,
                confidence=confidence,
                reasoning=reasoning or "Document summarization request",
                suggested_top_k=max(default_top_k, 12),
                skip_retrieval=False,
            )
        elif intent_str == "comparison":
            return RouterDecision(
                intent=IntentType.COMPARISON,
                confidence=confidence,
                reasoning=reasoning or "Comparative analysis across entities",
                suggested_top_k=max(default_top_k, 10),
                skip_retrieval=False,
            )
        else:
            return RouterDecision(
                intent=IntentType.FACTUAL_LOOKUP,
                confidence=confidence,
                reasoning=reasoning or "Standard fact lookup",
                suggested_top_k=default_top_k,
                skip_retrieval=False,
            )

    except asyncio.TimeoutError:
        logger.info("Agentic router timed out, falling back to factual_lookup")
        return RouterDecision(
            intent=IntentType.FACTUAL_LOOKUP,
            confidence=0.5,
            reasoning="Classifier timeout, default fallback",
            suggested_top_k=default_top_k,
            skip_retrieval=False,
        )
    except Exception as e:
        logger.warning(f"Agentic router failed ({e}), falling back to factual_lookup")
        return RouterDecision(
            intent=IntentType.FACTUAL_LOOKUP,
            confidence=0.5,
            reasoning=f"Classifier exception: {e}",
            suggested_top_k=default_top_k,
            skip_retrieval=False,
        )
