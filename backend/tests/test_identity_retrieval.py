"""
KueryCore AI — Identity Retrieval Regression Tests

Validates the identity query guard added to retrieval.py:
1. The _IDENTITY_PATTERN regex matches all expected query forms (including typos)
2. The header chunk injection logic surfaces chunk index=0 for name/identity queries
3. The RAG_PROMPT_TEMPLATE includes instructions for resume name extraction
"""

import re
import uuid
from types import SimpleNamespace

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Copied inline so this test doesn't require the full DB stack to run
IDENTITY_PATTERN = re.compile(
    r"\b(name|who\s*am\s*i|whose\s*(resume|cv|profile|document)|who\s*is\s*this|my\s*name|candidate('s)?\s*name|smy\s*name)\b",
    re.IGNORECASE,
)


def _make_chunk(content: str, index: int = 5) -> SimpleNamespace:
    """Stand-in for a SQLAlchemy Chunk ORM object."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        content=content,
        index=index,
        document_id=uuid.uuid4(),
        metadata_={"filename": "resume.pdf"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Identity Pattern Matcher Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityPatternMatcher:
    """Verify the regex captures all expected identity query variants."""

    @pytest.mark.parametrize("query", [
        "what is my name",
        "what smy name",          # The original typo bug
        "what's my name",
        "who am I",
        "who am i",
        "whose resume is this",
        "whose cv is this",
        "whose document is this",
        "whose profile is this",
        "who is this",
        "my name",
        "candidate name",
        "candidate's name",
        "what is the candidate's name",
        "tell me the candidate name",
        "smy name",               # typo variant
        "what is the name",
    ])
    def test_identity_pattern_matches(self, query: str):
        """Query must be matched by the identity guard pattern."""
        assert IDENTITY_PATTERN.search(query), (
            f"Identity pattern should match '{query}' but did not"
        )

    @pytest.mark.parametrize("query", [
        "what is my GPA",
        "what skills do I have",
        "summarize the document",
        "what are my projects",
        "what is my CGPA",
        "hello",
        "who hired me",
        "what companies did I work at",
    ])
    def test_non_identity_pattern_does_not_match(self, query: str):
        """Non-identity queries must NOT match the identity guard pattern."""
        assert not IDENTITY_PATTERN.search(query), (
            f"Identity pattern should NOT match '{query}' but it did"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Header Chunk Injection Logic Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHeaderChunkInjection:
    """Validate the header chunk injection merging logic in isolation."""

    def _simulate_injection(self, reranked_top_k, header_chunks, top_k):
        """Simulate the injection logic from retrieval.py without a DB."""
        existing_ids = {c.id for c, _ in reranked_top_k}
        for hc in header_chunks:
            if hc.id not in existing_ids:
                reranked_top_k.insert(0, (hc, 0.99))
                existing_ids.add(hc.id)
        if len(reranked_top_k) > max(top_k, 5):
            reranked_top_k = reranked_top_k[:max(top_k, 5)]
        return reranked_top_k

    def test_header_chunk_injected_at_top_when_missing(self):
        """Header chunk not in results must be injected at position 0."""
        other_chunk = _make_chunk("Aman has 3 years of experience.", index=5)
        header_chunk = _make_chunk("Aman Yadav\naman@email.com\n9876543210", index=0)

        reranked = [(other_chunk, 0.85)]
        result = self._simulate_injection(reranked, [header_chunk], top_k=5)

        assert result[0][0].id == header_chunk.id, "Header chunk must be at position 0"
        assert result[0][1] == 0.99, "Header chunk must have injected score of 0.99"
        assert len(result) == 2

    def test_header_chunk_not_duplicated_when_already_present(self):
        """Header chunk already in results must NOT be added again."""
        header_chunk = _make_chunk("Aman Yadav\naman@email.com", index=0)
        other_chunk = _make_chunk("Aman has 3 years of experience.", index=5)

        reranked = [(header_chunk, 0.75), (other_chunk, 0.60)]
        result = self._simulate_injection(reranked, [header_chunk], top_k=5)

        ids = [c.id for c, _ in result]
        assert ids.count(header_chunk.id) == 1, "Header chunk must not be duplicated"

    def test_result_capped_at_max_top_k_after_injection(self):
        """After injection the result list is capped at max(top_k, 5)."""
        header_chunk = _make_chunk("Aman Yadav", index=0)
        other_chunks = [(_make_chunk(f"chunk {i}", index=i + 1), 0.7 - i * 0.05) for i in range(5)]

        reranked = list(other_chunks)
        result = self._simulate_injection(reranked, [header_chunk], top_k=5)

        assert len(result) <= 5, f"Result must be capped at 5, got {len(result)}"

    def test_identity_guard_does_not_trigger_for_non_identity_query(self):
        """Non-identity queries must not match the pattern (no injection)."""
        query = "what is my CGPA"
        assert not IDENTITY_PATTERN.search(query), (
            "CGPA query should NOT trigger identity guard"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG Prompt Contract Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPromptContract:
    """Verify the RAG prompt template contains required identity instructions."""

    def _get_prompt(self):
        from app.services.generation import RAG_PROMPT_TEMPLATE
        return RAG_PROMPT_TEMPLATE

    def test_prompt_instructs_name_extraction_from_resume_header(self):
        """The prompt must explicitly mention resume header name extraction."""
        prompt = self._get_prompt()
        assert "name" in prompt.lower(), "Prompt must mention name"
        assert "resume" in prompt.lower() or "cv" in prompt.lower(), (
            "Prompt must mention resume or CV for name extraction guidance"
        )

    def test_prompt_contains_identity_query_examples(self):
        """The prompt must include example identity queries for the LLM."""
        prompt = self._get_prompt()
        assert "what is my name" in prompt.lower() or "who am i" in prompt.lower(), (
            "Prompt must include identity query examples"
        )

    def test_prompt_does_not_suppress_header_name_answers(self):
        """The insufficient-information rule must not apply to resume name queries."""
        prompt = self._get_prompt()
        # The updated rule 6 should reference header/title in documents
        assert "header" in prompt.lower() or "title" in prompt.lower() or "prominently" in prompt.lower(), (
            "Prompt should guide the LLM to look at document headers for names"
        )
