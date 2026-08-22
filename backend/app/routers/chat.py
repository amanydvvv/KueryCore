"""
KueryCore AI - Chat Router
RAG Q&A engine endpoint for natural language document querying with multi-turn memory and user tenant isolation.
"""

import logging
import time
import uuid
from json import dumps

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Conversation, Message, QueryLog, Document
from app.models.user import User
from app.core.ratelimit import limiter
from app.core.security import get_current_user
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, Citation
from app.services.retrieval import retrieve_context
from app.services.rewrite import rewrite_followup
from app.services.router import classify_intent, IntentType
from app.services.generation import get_llm

from app.services.memory import update_conversation_summary
from app.services.guardrails import (
    INJECTION_REFUSAL_MESSAGE,
    OUTPUT_SAFE_MESSAGE,
    OUTPUT_DISCLAIMER_DELTA,
    is_injection,
    sanitize_pii,
    validate_output,
)
from app.services.generation import (
    generate_answer,
    generate_answer_stream,
    build_token_budgeted_history,
    RateLimitError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


async def _generate_conversation_title(conv_id: uuid.UUID, question: str):
    """Background task to generate a short title for a new conversation via LLM."""
    try:
        llm = get_llm(temperature=0.1)
        prompt = f"Summarize this query into a concise, human-readable title (3 to 6 words). Return ONLY the title text, no quotes or explanations.\nQuery: {question}"
        resp = await llm.ainvoke(prompt)
        title = str(resp.content).strip().strip('"').strip("'")
        if title:
            # We need a new session since this runs in the background
            from app.database import async_session
            async with async_session() as db:
                await db.execute(update(Conversation).where(Conversation.id == conv_id).values(title=title))
                await db.commit()
    except Exception as e:
        logger.warning(f"Failed to generate title for conv {conv_id}: {e}")

def _guard_input(text: str) -> tuple[str, bool]:
    """Sanitize-once + injection check (plan v3 §2.2, sanitize-once decision
    A.2). Returns (sanitized_text, refusal_needed). Raw text is never stored,
    never sent downstream: the sanitized form drives persistence, history,
    retrieval, the cache key, and the LLM.

    GUARDRAILS_ENABLED=False bypasses both checks entirely — exact
    pre-guardrail behavior.
    """
    settings = get_settings()
    if not settings.GUARDRAILS_ENABLED:
        return text, False
    sanitized = sanitize_pii(text)
    flagged, _ = is_injection(sanitized)
    return sanitized, flagged


CORPUS_SUMMARY_LABEL = "Workspace Documents Summary"


async def _build_corpus_metadata(db: AsyncSession, user_id) -> str:
    """Query the user's actual document count and titles (ground truth for the LLM)."""
    count_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.user_id == user_id,
            Document.status == "completed",
        )
    )
    doc_count = count_result.scalar_one()

    name_result = await db.execute(
        select(Document.filename, Document.display_title).where(
            Document.user_id == user_id,
            Document.status == "completed",
        ).distinct()
    )
    titles = [row[1] if row[1] else row[0] for row in name_result.all()]

    return (
        f"{CORPUS_SUMMARY_LABEL}: You have {doc_count} document(s): {', '.join(titles)}"
    )


def _deduplicate_citations(citations: list) -> list:
    """Keep one citation per document_id (the highest scored chunk)."""
    seen = {}
    for cit in citations:
        doc_id = str(cit.document_id)
        if doc_id not in seen or cit.score > seen[doc_id].score:
            seen[doc_id] = cit
    return list(seen.values())


@router.post("", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    request_body: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Query the knowledge base using Retrieval-Augmented Generation (RAG)
    with multi-turn conversation memory, scoped to current user.
    """
    start_time = time.time()
    user_id = current_user.id

    # Guardrails run before anything else: sanitize-once (persist, history,
    # cache key and LLM all consume the sanitized text) + injection check.
    question, refusal_needed = _guard_input(request_body.question)

    try:
        # 1. Session Management
        conversation_id = request_body.conversation_id
        if conversation_id:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == current_user.id
                )
            )
            conv = result.scalar_one_or_none()
            if not conv:
                raise HTTPException(
                    status_code=404, detail=f"Conversation {conversation_id} not found."
                )
        else:
            _words = question.strip().split()
            conv = Conversation(
                id=uuid.uuid4(),
                user_id=current_user.id,
                title=" ".join(_words[:5]) + ("..." if len(_words) > 5 else "")
            )
            db.add(conv)
            await db.flush()
            conversation_id = conv.id
            background_tasks.add_task(_generate_conversation_title, conversation_id, question)

        # 2. Fetch past conversation history (newest first; token budget applied below)
        hist_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
        )
        chat_history = build_token_budgeted_history(hist_result.scalars().all())

        # 3. Save user's question as a Message (sanitized — raw never stored)
        user_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        db.add(user_msg)
        await db.flush()

        intent_name = "factual_lookup"
        if refusal_needed:
            avg_similarity = 0.0
            citations = []
            citation_dicts = []
            answer = INJECTION_REFUSAL_MESSAGE
            intent_name = "refusal"
        else:
            # 4. Agentic Intent Classification (Phase 1 Router)
            decision = await classify_intent(
                query=question,
                chat_history=chat_history,
                default_top_k=request_body.top_k,
            )
            intent_name = decision.intent.value

            # 5. Query corpus metadata (ground truth for the LLM)
            corpus_metadata = await _build_corpus_metadata(db, current_user.id)

            if decision.skip_retrieval:
                retrieval_query = question
                retrieved_items = []
                chunks = []
            else:
                # Retrieve relevant chunks with router-suggested top_k
                retrieval_query = await rewrite_followup(question, chat_history)
                effective_top_k = max(request_body.top_k, decision.suggested_top_k)
                retrieved_items = await retrieve_context(
                    query=retrieval_query,
                    db=db,
                    document_id=request_body.document_id,
                    user_id=current_user.id,
                    top_k=effective_top_k,
                )
                chunks = [item[0] for item in retrieved_items]

            # 6. Build citations list with real score & filename
            citations = []
            similarity_scores = []
            for chunk, score, filename in retrieved_items:
                page_num = (
                    chunk.page_number
                    if chunk.page_number is not None
                    else (chunk.metadata_.get("page_number", None) if chunk.metadata_ else None)
                )
                source = chunk.metadata_.get("source", None) if chunk.metadata_ else None
                display_title = (chunk.metadata_ or {}).get("display_title") or filename
                similarity_scores.append(score)
                citations.append(
                    Citation(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        filename=filename,
                        display_title=display_title,
                        page_number=page_num,
                        score=score,
                        content_preview=chunk.content[:297] + "..."
                        if len(chunk.content) > 300
                        else chunk.content,
                        source=source,
                    )
                )

            # Deduplicate citations: one per document, keeping highest score
            citations = _deduplicate_citations(citations)

            avg_similarity = (
                round(sum(similarity_scores) / len(similarity_scores), 4)
                if similarity_scores
                else 0.0
            )

            # 7. Generate answer using Groq (Llama 3.1) with chat history + corpus metadata
            if not chunks and not decision.skip_retrieval:
                answer = "I couldn't find any relevant information in your uploaded documents to answer your question."
            else:
                answer = await generate_answer(
                    query=question,
                    chunks=chunks,
                    chat_history=chat_history,
                    corpus_metadata=corpus_metadata,
                    conversation_summary=conv.context_summary,
                    resolved_query=retrieval_query if not decision.skip_retrieval else None,
                )


            # Output guardrail: replace a flagged answer, but log the flag
            # (with reasons) — never a silent swap.
            if get_settings().GUARDRAILS_ENABLED:
                validate_ok, reasons = validate_output(answer)
                if not validate_ok:
                    logger.warning(
                        "output guardrail replaced answer for user %s: %s",
                        current_user.id,
                        ", ".join(reasons),
                    )
                    answer = OUTPUT_SAFE_MESSAGE

        latency_ms = int((time.time() - start_time) * 1000)

        # 8. Save assistant's answer as a Message
        citation_dicts = [
            {
                "chunk_id": str(c.chunk_id),
                "document_id": str(c.document_id),
                "filename": c.filename,
                "display_title": c.display_title,
                "page_number": c.page_number,
                "score": c.score,
                "content_preview": c.content_preview,
                "source": c.source,
            }
            for c in citations
        ]

        assistant_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            citations=citation_dicts,
            latency_ms=latency_ms,
        )
        db.add(assistant_msg)

        # 9. Persist QueryLog for analytics (question = sanitized text)
        query_log = QueryLog(
            id=uuid.uuid4(),
            user_id=current_user.id,
            question=question,
            retrieved_chunks=citation_dicts,
            top_k=request_body.top_k,
            avg_similarity=avg_similarity,
            latency_ms=latency_ms,
        )
        db.add(query_log)
        await db.commit()

        # Summary Buffer Memory: summarize the last 5 messages in the background.
        # The service opens its own session (request-scoped db is closed by then).
        background_tasks.add_task(update_conversation_summary, conversation_id)

        return ChatResponse(
            answer=answer,
            citations=citations,
            conversation_id=conversation_id,
            latency_ms=latency_ms,
            avg_similarity=avg_similarity,
            intent=intent_name,
        )


    except RateLimitError as e:
        await db.rollback()
        raise HTTPException(status_code=429, detail=str(e))
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Chat endpoint failed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")


@router.post("/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    request_body: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Query the knowledge base using RAG with real-time SSE token streaming, scoped to current user.
    """
    start_time = time.time()

    # Guardrails run synchronously pre-stream (plan v3 §2.2): sanitize-once
    # plus injection check. Raw text never reaches persistence or the LLM.
    question, refusal_needed = _guard_input(request_body.question)

    # 1. Session Management
    conversation_id = request_body.conversation_id
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id} not found."
            )
    else:
        _words = question.strip().split()
        conv = Conversation(
            id=uuid.uuid4(),
            user_id=current_user.id,
            title=" ".join(_words[:5]) + ("..." if len(_words) > 5 else "")
        )
        db.add(conv)
        await db.flush()
        conversation_id = conv.id
        background_tasks.add_task(_generate_conversation_title, conversation_id, question)

    # 2. Fetch past history (newest first; token budget applied below)
    hist_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
    )
    chat_history = build_token_budgeted_history(hist_result.scalars().all())

    # 3. Save user's question (sanitized — raw never stored)
    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    await db.flush()

    # 3b. Injection short-circuit, fully pre-stream: persist the refusal turn
    # now (user message + assistant refusal + QueryLog), then stream the
    # normal event sequence over canned content. No retrieval, no LLM.
    if refusal_needed:
        refusal_latency_ms = int((time.time() - start_time) * 1000)
        assistant_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=INJECTION_REFUSAL_MESSAGE,
            citations=[],
            latency_ms=refusal_latency_ms,
        )
        db.add(assistant_msg)
        query_log = QueryLog(
            id=uuid.uuid4(),
            user_id=current_user.id,
            question=question,
            retrieved_chunks=[],
            top_k=request_body.top_k,
            avg_similarity=0.0,
            latency_ms=refusal_latency_ms,
        )
        db.add(query_log)
        await db.commit()

        async def refusal_generator():
            meta_payload = dumps({
                "conversation_id": str(conversation_id),
                "citations": [],
                "avg_similarity": 0.0,
            })
            yield f"event: metadata\ndata: {meta_payload}\n\n"
            token_payload = dumps({"delta": INJECTION_REFUSAL_MESSAGE})
            yield f"event: token\ndata: {token_payload}\n\n"
            done_payload = dumps({"latency_ms": refusal_latency_ms})
            yield f"event: done\ndata: {done_payload}\n\n"

        return StreamingResponse(
            refusal_generator(),
            media_type="text/event-stream",
            background=BackgroundTask(update_conversation_summary, conversation_id),
        )

    # 4. Agentic Intent Classification (Phase 1 Router)
    decision = await classify_intent(
        query=question,
        chat_history=chat_history,
        default_top_k=request_body.top_k,
    )
    intent_name = decision.intent.value

    # 5. Query corpus metadata (ground truth for the LLM)
    corpus_metadata = await _build_corpus_metadata(db, current_user.id)

    if decision.skip_retrieval:
        retrieval_query = question
        retrieved_items = []
        chunks = []
    else:
        # Retrieve context chunks scoped to user
        retrieval_query = await rewrite_followup(question, chat_history)
        effective_top_k = max(request_body.top_k, decision.suggested_top_k)
        retrieved_items = await retrieve_context(
            query=retrieval_query,
            db=db,
            document_id=request_body.document_id,
            user_id=current_user.id,
            top_k=effective_top_k,
        )
        chunks = [item[0] for item in retrieved_items]

    citations = []
    similarity_scores = []
    for chunk, score, filename in retrieved_items:
        page_num = (
            chunk.page_number
            if chunk.page_number is not None
            else (chunk.metadata_.get("page_number", None) if chunk.metadata_ else None)
        )
        source = chunk.metadata_.get("source", None) if chunk.metadata_ else None
        display_title = (chunk.metadata_ or {}).get("display_title") or filename
        similarity_scores.append(score)
        citations.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                display_title=display_title,
                page_number=page_num,
                score=score,
                content_preview=chunk.content[:297] + "..."
                if len(chunk.content) > 300
                else chunk.content,
                source=source,
            )
        )

    # Deduplicate citations: one per document, keeping highest score
    citations = _deduplicate_citations(citations)

    avg_similarity = (
        round(sum(similarity_scores) / len(similarity_scores), 4)
        if similarity_scores
        else 0.0
    )

    citation_dicts = [
        {
            "id": str(c.chunk_id),
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "filename": c.filename,
            "display_title": c.display_title,
            "page_number": c.page_number,
            "score": c.score,
            "content_preview": c.content_preview,
            "source": c.source,
        }
        for c in citations
    ]

    async def event_generator():
        full_answer = []
        try:
            # Event 1: Metadata (includes intent)
            meta_payload = dumps({
                "conversation_id": str(conversation_id),
                "citations": citation_dicts,
                "avg_similarity": avg_similarity,
                "intent": intent_name,
            })
            yield f"event: metadata\ndata: {meta_payload}\n\n"

            if not chunks and not decision.skip_retrieval:
                no_info = "I couldn't find any relevant information in your uploaded documents to answer your question."
                full_answer.append(no_info)
                token_payload = dumps({"delta": no_info})
                yield f"event: token\ndata: {token_payload}\n\n"
            else:
                async for token in generate_answer_stream(
                    query=question,
                    chunks=chunks,
                    chat_history=chat_history,
                    corpus_metadata=corpus_metadata,
                    conversation_summary=conv.context_summary,
                    resolved_query=retrieval_query if not decision.skip_retrieval else None,
                ):
                    full_answer.append(token)
                    token_payload = dumps({"delta": token})
                    yield f"event: token\ndata: {token_payload}\n\n"


            complete_text = "".join(full_answer)
            latency_ms = int((time.time() - start_time) * 1000)

            # Output guardrail (post-stream, never mutates emitted tokens):
            # on a flag, append the disclaimer as one more token delta before
            # `done` and log the reasons — honest, non-destructive.
            disclaimer = None
            if get_settings().GUARDRAILS_ENABLED:
                out_ok, out_reasons = validate_output(complete_text)
                if not out_ok:
                    logger.warning(
                        "output guardrail flagged streamed answer for user %s: %s",
                        current_user.id,
                        ", ".join(out_reasons),
                    )
                    disclaimer = OUTPUT_DISCLAIMER_DELTA
                    token_payload = dumps({"delta": disclaimer})
                    yield f"event: token\ndata: {token_payload}\n\n"

            persisted_answer = (
                complete_text if disclaimer is None else complete_text + disclaimer
            )

            # Persist assistant message & query log
            assistant_msg = Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                role="assistant",
                content=persisted_answer,
                citations=citation_dicts,
                latency_ms=latency_ms,
            )
            db.add(assistant_msg)

            query_log = QueryLog(
                id=uuid.uuid4(),
                user_id=current_user.id,
                question=question,
                retrieved_chunks=citation_dicts,
                top_k=request_body.top_k,
                avg_similarity=avg_similarity,
                latency_ms=latency_ms,
            )
            db.add(query_log)
            await db.commit()

            done_payload = dumps({"latency_ms": latency_ms})
            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception:
            await db.rollback()
            err_payload = dumps({"detail": "An internal error occurred during generation."})
            yield f"event: error\ndata: {err_payload}\n\n"

    # Background Tasks: update memory summary + generate concise LLM title
    background_tasks.add_task(update_conversation_summary, conversation_id)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        background=background_tasks,
    )
