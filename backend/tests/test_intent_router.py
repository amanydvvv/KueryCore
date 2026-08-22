"""
KueryCore AI — Agentic Intent Router Tests (Phase 1)

Tests:
1. Fast regex heuristics (greetings, summary, comparison) for zero-latency routing.
2. LLM-based classification parsing and structured output extraction.
3. Fail-safe fallbacks on timeout, malformed responses, and router kill switch.
4. Dynamic top_k suggestion adaptation based on detected intent.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.config import get_settings
from app.services.router import (
    IntentType,
    RouterDecision,
    _heuristic_classify,
    _extract_json_payload,
    classify_intent,
)
from app.services import router


# ─────────────────────────────────────────────────────────────
# 1. Fast Heuristics Tests
# ─────────────────────────────────────────────────────────────

def test_heuristic_greetings_classified_as_chitchat():
    greetings = [
        "hi",
        "Hello!",
        "hey there",
        "Good morning",
        "who are you",
        "what can you do?",
        "Thank you!",
        "bye",
    ]
    for g in greetings:
        decision = _heuristic_classify(g)
        assert decision is not None, f"Failed to classify greeting: {g}"
        assert decision.intent == IntentType.CHITCHAT
        assert decision.skip_retrieval is True
        assert decision.suggested_top_k == 0


def test_heuristic_summarization_queries():
    queries = [
        "summarize this document",
        "give me a summary of the quarterly report",
        "Overview of the architecture",
        "key takeaways from the meeting",
        "what is this document about?",
        "TLDR of section 2",
    ]
    for q in queries:
        decision = _heuristic_classify(q)
        assert decision is not None, f"Failed to classify summarization: {q}"
        assert decision.intent == IntentType.SUMMARIZATION
        assert decision.skip_retrieval is False
        assert decision.suggested_top_k >= 10


def test_heuristic_comparison_queries():
    queries = [
        "compare the two architecture proposals",
        "what is the difference between plan A and plan B?",
        "Postgres vs MySQL latency differences",
        "contrast the performance metrics",
    ]
    for q in queries:
        decision = _heuristic_classify(q)
        assert decision is not None, f"Failed to classify comparison: {q}"
        assert decision.intent == IntentType.COMPARISON
        assert decision.skip_retrieval is False
        assert decision.suggested_top_k >= 8


def test_heuristic_passes_unmatched_to_llm():
    general_queries = [
        "What is the maximum token budget for context history?",
        "When was the server migration completed?",
        "Explain how the RRF fusion algorithm computes ranking scores.",
    ]
    for q in general_queries:
        assert _heuristic_classify(q) is None


# ─────────────────────────────────────────────────────────────
# 2. JSON Parser Resilience Tests
# ─────────────────────────────────────────────────────────────

def test_extract_json_clean():
    raw = '{"intent": "factual_lookup", "confidence": 0.95, "reasoning": "Specific fact query"}'
    parsed = _extract_json_payload(raw)
    assert parsed == {"intent": "factual_lookup", "confidence": 0.95, "reasoning": "Specific fact query"}


def test_extract_json_markdown_wrapped():
    raw = '```json\n{"intent": "comparison", "confidence": 0.9, "reasoning": "Comparing metrics"}\n```'
    parsed = _extract_json_payload(raw)
    assert parsed is not None
    assert parsed["intent"] == "comparison"


def test_extract_json_embedded_in_text():
    raw = 'Here is the decision:\n{"intent": "summarization", "confidence": 0.85, "reasoning": "General overview"}\nHope this helps!'
    parsed = _extract_json_payload(raw)
    assert parsed is not None
    assert parsed["intent"] == "summarization"


def test_extract_json_invalid_returns_none():
    raw = "I think this is a factual query but I won't format as json"
    assert _extract_json_payload(raw) is None


# ─────────────────────────────────────────────────────────────
# 3. LLM Router & Fail-Safe Tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_intent_llm_factual_lookup(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"intent": "factual_lookup", "confidence": 0.95, "reasoning": "Fact lookup"}'
        )
    )
    monkeypatch.setattr(router, "get_llm", lambda **kwargs: mock_llm)

    decision = await classify_intent("What is the server port?", default_top_k=5)
    assert decision.intent == IntentType.FACTUAL_LOOKUP
    assert decision.suggested_top_k == 5
    assert decision.skip_retrieval is False


@pytest.mark.asyncio
async def test_classify_intent_llm_comparison(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"intent": "comparison", "confidence": 0.92, "reasoning": "Comparing two options"}'
        )
    )
    monkeypatch.setattr(router, "get_llm", lambda **kwargs: mock_llm)

    decision = await classify_intent("Which database option provides better throughput?", default_top_k=5)
    assert decision.intent == IntentType.COMPARISON
    assert decision.suggested_top_k == 10
    assert decision.skip_retrieval is False


@pytest.mark.asyncio
async def test_classify_intent_timeout_fails_safe_to_factual(monkeypatch):
    async def _mock_timeout(*args, **kwargs):
        await asyncio.sleep(5.0)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=_mock_timeout)
    monkeypatch.setattr(router, "get_llm", lambda **kwargs: mock_llm)
    monkeypatch.setattr(router, "ROUTER_TIMEOUT_SECONDS", 0.05)

    decision = await classify_intent("Tell me about the policy guidelines", default_top_k=5)
    assert decision.intent == IntentType.FACTUAL_LOOKUP
    assert decision.suggested_top_k == 5
    assert decision.skip_retrieval is False


@pytest.mark.asyncio
async def test_classify_intent_malformed_llm_fails_safe(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="I am an AI and I cannot output json today.")
    )
    monkeypatch.setattr(router, "get_llm", lambda **kwargs: mock_llm)

    decision = await classify_intent("What are the quarterly profits?", default_top_k=5)
    assert decision.intent == IntentType.FACTUAL_LOOKUP
    assert decision.suggested_top_k == 5


@pytest.mark.asyncio
async def test_classify_intent_router_disabled_kill_switch(monkeypatch):
    monkeypatch.setattr(get_settings(), "AGENTIC_ROUTER_ENABLED", False)

    # Even a greeting should not be routed when kill switch is active
    decision = await classify_intent("hello", default_top_k=5)
    assert decision.intent == IntentType.FACTUAL_LOOKUP
    assert decision.suggested_top_k == 5
    assert decision.skip_retrieval is False
