import logging
import os
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from app.models import Chunk, Message
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_HISTORY_TOKENS = 3000

# Minimum pre-normalization vector similarity a related-topic offer (rule 6)
# may reference. Offers are permitted only for a clearly related topic that
# appears in the retrieved context and whose raw similarity clears this
# floor - anything weaker reads as speculation, so declining stays the
# default.
ALLOWED_OFFER_MIN_SIMILARITY = 0.45


def build_token_budgeted_history(messages: List[Message]) -> List[Message]:
    """
    Sliding context window for conversation history based on a token budget.

    Expects `messages` sorted newest-first, and returns the newest messages
    that fit within MAX_HISTORY_TOKENS, re-ordered chronologically (oldest
    first) for the prompt template. Older messages are discarded the moment
    the budget is exceeded.
    """
    selected = []
    accumulated_tokens = 0
    for msg in messages:
        accumulated_tokens += max(1, len(msg.content) // 4)
        if accumulated_tokens > MAX_HISTORY_TOKENS:
            break
        selected.append(msg)
    selected.reverse()
    return selected


def get_llm(temperature: float = 0.3, model_name: Optional[str] = None):
    """
    Build a resilient LLM with a fallback cascade:
      1. Groq  settings.GENERATIVE_MODEL  (fast, primary — configured via env var)
      2. Groq  llama-3.3-70b-versatile    (higher capacity pool)
      3. Groq  qwen3-32b                  (different model family, separate queue)
      4. Gemini 1.5 Flash                 (cross-provider safety net)

    If any model returns a 503, 429, or any other error, LangChain's
    with_fallbacks() automatically tries the next one in the chain.

    `temperature` (default 0.3) is applied to every model in the cascade.
    `model_name`, when provided, bypasses the cascade entirely and returns a
    single ChatGroq instance pinned to that model — used by the eval harness
    judge, which must grade on its own model (EVAL_JUDGE_MODEL) at
    temperature=0 rather than the generation cascade.

    NOTE: To change the primary model, set GENERATIVE_MODEL in .env.
    Never hardcode a model name here — that's what caused the Phase 13 fire drill.
    """
    groq_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY

    if not groq_key and not gemini_key:
        raise RuntimeError(
            "Neither GROQ_API_KEY nor GEMINI_API_KEY is set. "
            "At least one LLM provider key is required for chat generation."
        )

    if model_name is not None:
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY is required when an explicit model_name is requested."
            )
        return ChatGroq(
            api_key=groq_key,
            model_name=model_name,
            temperature=temperature,
            max_retries=1,
        )

    fallbacks = []

    # --- Primary: Configured via settings.GENERATIVE_MODEL (env var) ---
    # NEVER hardcode the model name here. Change GENERATIVE_MODEL in .env instead.
    primary = None
    if groq_key:
        primary = ChatGroq(
            api_key=groq_key,
            model_name=settings.GENERATIVE_MODEL,
            temperature=temperature,
            max_retries=1,  # one retry with backoff before cascading
        )

        # --- Fallback 1: Groq openai/gpt-oss-120b (larger, same family) ---
        fallbacks.append(ChatGroq(
            api_key=groq_key,
            model_name="openai/gpt-oss-120b",
            temperature=temperature,
            max_retries=0,
        ))

        # --- Fallback 2: Groq qwen/qwen3.6-27b ---
        fallbacks.append(ChatGroq(
            api_key=groq_key,
            model_name="qwen/qwen3.6-27b",
            temperature=temperature,
            max_retries=0,
        ))

    # --- Fallback 3 (cross-provider): Gemini 1.5 Flash ---
    if gemini_key:
        gemini_fallback = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_key,
            temperature=temperature,
            max_output_tokens=1024,
        )
        if primary is None:
            # Groq key missing — Gemini is the only provider
            primary = gemini_fallback
        else:
            fallbacks.append(gemini_fallback)

    if primary is None:
        raise RuntimeError(
            "Could not initialise any LLM. Check your API key environment variables."
        )

    if fallbacks:
        logger.info(
            "LLM fallback cascade active: %s → %s",
            primary.model_name if hasattr(primary, 'model_name') else str(primary),
            " → ".join(
                fb.model_name if hasattr(fb, 'model_name') else str(fb)
                for fb in fallbacks
            ),
        )
        return primary.with_fallbacks(fallbacks)

    return primary


RAG_PROMPT_TEMPLATE = """
You are KueryCore AI, a precise document analysis assistant. Your ONLY job is to answer questions using the retrieved document Context below.

STRICT RULES:
1. RELEVANCE CHECK FIRST: Before answering, verify the Context actually contains information relevant to the User Question. If the retrieved Context does not address the question, respond: "I do not have sufficient information in the loaded documents to answer this question."
2. STAY ON TOPIC: If the User Question is casual conversation, small talk, or completely unrelated to the documents (e.g. "I think how's it", "hello", "thanks"), respond briefly and naturally without referencing documents.
3. NEVER invent or assume facts not present in the Context.
4. NEVER reference page numbers, source chunks, vector retrieval, or internal processing.
5. Answer directly and concisely. Do not start with "According to the document" or similar hedges.
6. If asked about a person, candidate, or user's name (e.g. "what is my name", "who am I", "whose resume is this", "what smy name") and the document is a resume, CV, bio, or profile, look for the person's name prominently stated in the document (such as in the header or title) and state their name directly.

Prior Context:
{conversation_summary}

{corpus_metadata}

{chat_history_section}

Context:
---------------------
{context}
---------------------

User Question: {query}
{resolved_query_section}

<thought_process>
Step 1: Is the User Question casual chat or off-topic? If yes, respond naturally without documents.
Step 2: Does the Context contain facts that answer the User Question (including candidate/person names stated in document or resume headers)? If no, output the "insufficient information" response.
Step 3: If yes, identify only the relevant facts and compose a precise answer.
</thought_process>
<answer>
"""



class RateLimitError(Exception):
    """Raised when the LLM provider hits rate limits or quota bounds."""
    pass


# Error markers that should trigger fallback (via RateLimitError or LangChain's with_fallbacks)
FALLBACK_TRIGGER_MARKERS = (
    "429",
    "Quota exceeded",
    "ResourceExhausted",
    "rate limit",
    "503",
    "queue is full",
)


def _is_fallback_error(err: Exception) -> bool:
    """Check if an error should trigger the fallback cascade."""
    err_str = str(err)
    return any(marker.lower() in err_str.lower() for marker in FALLBACK_TRIGGER_MARKERS)


class StreamCoTBuffer:
    """
    Server-side buffering state machine for SSE streaming with CoT tags.

    Buffers incoming tokens during <thought_process> phase until <answer> tag appears.
    Discards all thought process tokens.
    Once <answer> tag appears, streams tokens to SSE until </answer> is encountered.
    If <answer> tag is missing (model fallback), flushes buffer gracefully after stripping thoughts.
    """

    def __init__(self):
        self.state = "SEEKING_ANSWER"  # SEEKING_ANSWER -> STREAMING_ANSWER -> COMPLETED
        self.buffer = ""

    def process_token(self, token: str) -> list[str]:
        if self.state == "COMPLETED":
            return []

        self.buffer += token
        output_deltas = []

        if self.state == "SEEKING_ANSWER":
            answer_idx = self.buffer.find("<answer>")
            if answer_idx != -1:
                content_after = self.buffer[answer_idx + len("<answer>"):]
                self.buffer = ""
                self.state = "STREAMING_ANSWER"
                if content_after:
                    output_deltas.extend(self._process_answer_text(content_after))
            else:
                if len(self.buffer) > 1500 and "<thought_process>" not in self.buffer:
                    self.state = "STREAMING_ANSWER"
                    to_flush = self.buffer
                    self.buffer = ""
                    output_deltas.extend(self._process_answer_text(to_flush))

        elif self.state == "STREAMING_ANSWER":
            to_process = self.buffer
            self.buffer = ""
            output_deltas.extend(self._process_answer_text(to_process))

        return output_deltas

    def _process_answer_text(self, text: str) -> list[str]:
        deltas = []
        end_idx = text.find("</answer>")
        if end_idx != -1:
            answer_part = text[:end_idx]
            if answer_part:
                deltas.append(answer_part)
            self.state = "COMPLETED"
            self.buffer = ""
        else:
            possible_closing_prefixes = ["</", "</a", "</an", "</ans", "</answ", "</answe", "</answer"]
            hold_len = 0
            for prefix in possible_closing_prefixes:
                if text.endswith(prefix):
                    hold_len = len(prefix)
                    break
            if hold_len > 0:
                head = text[:-hold_len]
                if head:
                    deltas.append(head)
                self.buffer = text[-hold_len:]
            else:
                deltas.append(text)
        return deltas

    def finalize(self) -> list[str]:
        output_deltas = []
        if self.state == "SEEKING_ANSWER":
            cleaned = self.buffer
            if "<thought_process>" in cleaned:
                start = cleaned.find("<thought_process>")
                end = cleaned.find("</thought_process>", start)
                if end != -1:
                    cleaned = cleaned[end + len("</thought_process>"):].strip()
                else:
                    cleaned = ""
            cleaned = cleaned.replace("<answer>", "").replace("</answer>", "").strip()
            if cleaned:
                output_deltas.append(cleaned)
        elif self.state == "STREAMING_ANSWER":
            if self.buffer and self.buffer != "</answer>":
                cleaned = self.buffer.replace("</answer>", "")
                if cleaned:
                    output_deltas.append(cleaned)
        self.buffer = ""
        self.state = "COMPLETED"
        return output_deltas


def extract_answer_from_cot(raw_text: str) -> str:
    """Extract answer from <answer>...</answer> tags or strip <thought_process>."""
    if not raw_text:
        return ""
    if "<answer>" in raw_text:
        start = raw_text.find("<answer>") + len("<answer>")
        end = raw_text.find("</answer>", start)
        if end != -1:
            return raw_text[start:end].strip()
        return raw_text[start:].strip()
    
    cleaned = raw_text
    if "<thought_process>" in cleaned:
        start = cleaned.find("<thought_process>")
        end = cleaned.find("</thought_process>", start)
        if end != -1:
            cleaned = cleaned[end + len("</thought_process>"):].strip()
        else:
            cleaned = cleaned[:start].strip()
    return cleaned.replace("</answer>", "").strip()


def _format_context_text(chunks: List[Chunk]) -> str:
    snippet_items = []
    for i, chunk in enumerate(chunks):
        title = (chunk.metadata_ or {}).get("display_title") or (chunk.metadata_ or {}).get("filename") or f"Document {i+1}"
        snippet_items.append(f"[{title}]\n{chunk.content}")
    return "\n\n---\n\n".join(snippet_items)


def _build_resolved_query_section(query: str, resolved_query: Optional[str]) -> str:
    """Prompt section carrying the standalone (rewritten) retrieval query.

    Included only when a rewrite actually happened; otherwise the empty
    string keeps the template shape unchanged. The section is plain query
    content, never an instruction block, so it cannot override the rules.
    """
    if resolved_query and resolved_query.strip() and resolved_query.strip() != query:
        return f"Resolved Query: {resolved_query.strip()}"
    return ""


def _build_chain_inputs(
    query: str,
    chunks: List[Chunk],
    chat_history: Optional[List[Message]],
    corpus_metadata: str,
    conversation_summary: Optional[str],
    resolved_query: Optional[str],
) -> dict:
    """Build the prompt input dict consumed by both generate_answer and
    generate_answer_stream. Centralised here so prompt/history changes
    only ever need to be made in one place.
    """
    context_text = _format_context_text(chunks)
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        chat_history_section = (
            "Conversation History:\n"
            + "\n".join(history_lines)
            + "\n---------------------"
        )
    else:
        chat_history_section = ""
    return {
        "corpus_metadata": corpus_metadata,
        "chat_history_section": chat_history_section,
        "context": context_text,
        "query": query,
        "resolved_query_section": _build_resolved_query_section(query, resolved_query),
        "conversation_summary": conversation_summary or "",
    }


async def generate_answer(
    query: str,
    chunks: List[Chunk],
    chat_history: Optional[List[Message]] = None,
    corpus_metadata: str = "",
    conversation_summary: Optional[str] = None,
    resolved_query: Optional[str] = None,
) -> str:
    """
    Generate an answer using the provided chunks as context and prior chat history.

    `resolved_query` is the standalone retrieval query the chunks were found
    with (see app.services.rewrite); when it differs from `query`, it is
    surfaced to the model so it does not misread a deictic follow-up in
    isolation.
    """
    logger.info("Generating answer based on retrieved context and conversation history...")

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=[
            "corpus_metadata", "chat_history_section", "context",
            "query", "resolved_query_section", "conversation_summary",
        ]
    )
    chain = prompt | get_llm()

    try:
        response = await chain.ainvoke(
            _build_chain_inputs(
                query, chunks, chat_history, corpus_metadata,
                conversation_summary, resolved_query,
            )
        )
        raw = response.content if hasattr(response, "content") else str(response)
        return extract_answer_from_cot(raw)
    except Exception as e:
        if _is_fallback_error(e):
            logger.warning(f"LLM error triggers fallback: {e}")
            raise RateLimitError(
                "The AI service is currently at capacity or quota limits have been reached. Please try again in a moment."
            )
        logger.error(f"Error during LLM generation: {e}", exc_info=True)
        raise


async def generate_answer_stream(
    query: str,
    chunks: List[Chunk],
    chat_history: Optional[List[Message]] = None,
    corpus_metadata: str = "",
    conversation_summary: Optional[str] = None,
    resolved_query: Optional[str] = None,
):
    """
    Stream answer tokens as they arrive from the LLM provider.

    `resolved_query` semantics match generate_answer(): the standalone
    retrieval query, shown to the model only when it differs from `query`.
    """
    logger.info("Streaming answer tokens from LLM...")

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=[
            "corpus_metadata", "chat_history_section", "context",
            "query", "resolved_query_section", "conversation_summary",
        ]
    )
    chain = prompt | get_llm()

    cot_buffer = StreamCoTBuffer()
    try:
        async for chunk_response in chain.astream(
            _build_chain_inputs(
                query, chunks, chat_history, corpus_metadata,
                conversation_summary, resolved_query,
            )
        ):
            if chunk_response.content:
                for delta in cot_buffer.process_token(chunk_response.content):
                    if delta:
                        yield delta
        for delta in cot_buffer.finalize():
            if delta:
                yield delta
    except Exception as e:
        if _is_fallback_error(e):
            logger.warning(f"LLM error triggers fallback during stream: {e}")
            raise RateLimitError(
                "The AI service is currently at capacity or quota limits have been reached. Please try again in a moment."
            )
        logger.error(f"Error during LLM token streaming: {e}", exc_info=True)
        raise