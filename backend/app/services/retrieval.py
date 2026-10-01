"""
KueryCore AI - High-Rigor Two-Stage Hybrid Retrieval Engine
Combines pgvector HNSW dense vector search with PostgreSQL Full-Text Search (BM25 lexical search) concurrently via asyncio.gather.
Fuses candidates using Reciprocal Rank Fusion (RRF, k=60), and re-ranks top candidates using Option 2b Phrase Coverage & Lexical Re-Scorer.
"""

import asyncio
import logging
import re
import time
from typing import List, Optional, Tuple, Dict
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.models import Chunk, Document
from app.config import get_settings
from app.services.query_cache import query_cache
from app.database import async_session as make_session

logger = logging.getLogger(__name__)
settings = get_settings()

# Candidate Pool Configuration Constants
VECTOR_TOP_N = 20       # Top candidates from pgvector HNSW dense search
LEXICAL_TOP_N = 20      # Top candidates from PostgreSQL FTS lexical search
RRF_K = 60              # Reciprocal Rank Fusion smoothing constant (higher = more rank dilution)
RRF_FUSED_TOP_K = 10    # Top candidates surviving RRF rank fusion into Stage 2
FINAL_TOP_K = 5         # Final re-ranked candidates passed to generation.py

# Stage 2 re-ranker blend weights (normalized feature combination)
WEIGHT_VECTOR_NORM = 0.50  # normalized pgvector dense score
WEIGHT_LEXICAL_NORM = 0.30  # normalized FTS lexical rank
WEIGHT_PHRASE_NORM = 0.20  # normalized phrase-coverage score

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model=f"models/{settings.EMBEDDING_MODEL}",
    google_api_key=settings.GEMINI_API_KEY or "dummy-key-for-init",
    timeout=30.0,
)


def _compute_independent_phrase_coverage(query: str, content: str) -> float:
    """
    Calculate an independent phrase-coverage & term-matching heuristic score [0.0, 1.0].
    Strictly measures query token overlap and exact multi-word phrase presence in the chunk content,
    completely independent of vector embedding distances or FTS rank values.
    """
    clean_query = re.sub(r"[^\w\s]", "", query.lower()).strip()
    clean_content = re.sub(r"[^\w\s]", "", content.lower()).strip()

    if not clean_query or not clean_content:
        return 0.0

    stopwords = {"a", "an", "the", "is", "are", "was", "were", "of", "for", "in", "to", "on", "with", "and", "or", "what", "who", "where", "how", "why"}
    query_tokens = [t for t in clean_query.split() if t not in stopwords and len(t) > 1]

    if not query_tokens:
        query_tokens = clean_query.split()

    matched_tokens = sum(1 for token in query_tokens if token in clean_content)
    token_score = matched_tokens / len(query_tokens) if query_tokens else 0.0

    # Independent exact multi-word phrase bonus
    phrase_bonus = 0.3 if len(query_tokens) > 1 and clean_query in clean_content else 0.0
    return min(1.0, round(token_score + phrase_bonus, 4))


async def _retrieve_vector_candidates(
    query: str,
    db: AsyncSession,
    document_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    limit: int = VECTOR_TOP_N,
) -> List[Tuple[Chunk, float]]:
    """
    Stage 1 Dense Vector Search via pgvector or fallback JSONB vector calculation.
    Returns List[Tuple[Chunk, similarity_score]]
    """
    query_vector = await asyncio.to_thread(
        embeddings.embed_query,
        query,
        output_dimensionality=settings.EMBEDDING_DIMENSION,
    )

    try:
        distance_col = Chunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = select(Chunk, distance_col)

        if user_id is not None or document_id is not None:
            stmt = stmt.join(Document, Chunk.document_id == Document.id)

        if user_id is not None:
            stmt = stmt.where(Document.user_id == user_id)

        if document_id is not None:
            stmt = stmt.where(Chunk.document_id == document_id)

        stmt = stmt.order_by(distance_col).limit(limit)
        result = await db.execute(stmt)
        rows = result.all()

        candidates = []
        for chunk, distance in rows:
            dist_val = float(distance) if distance is not None else 1.0
            similarity = max(0.0, round(1.0 - dist_val, 4))
            candidates.append((chunk, similarity))

        return candidates
    except Exception as pgvec_err:
        # Fallback for DB instances without pgvector C extension compiled.
        # Log at WARNING — this path should never silently hide real DB errors
        # in production. If this fires repeatedly, the pgvector extension is
        # likely missing or the embedding column type is misconfigured.
        logger.warning(
            "pgvector HNSW query failed (%s); falling back to in-memory cosine scan. "
            "Performance may be degraded on large corpora.",
            pgvec_err,
        )
        stmt = select(Chunk)
        if user_id is not None or document_id is not None:
            stmt = stmt.join(Document, Chunk.document_id == Document.id)

        if user_id is not None:
            stmt = stmt.where(Document.user_id == user_id)

        if document_id is not None:
            stmt = stmt.where(Chunk.document_id == document_id)

        # Cap the fallback scan — a full table scan with no LIMIT could OOM
        # on large corpora. 10× top-K gives the cosine scorer enough candidates.
        stmt = stmt.limit(limit * 10)
        result = await db.execute(stmt)
        chunks = result.scalars().all()

        import math
        def dot_product(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        def magnitude(v):
            return math.sqrt(sum(x * x for x in v))

        q_mag = magnitude(query_vector)
        scored = []

        for chk in chunks:
            try:
                emb = chk.embedding if isinstance(chk.embedding, list) else list(chk.embedding)
                c_mag = magnitude(emb)
                if q_mag > 0 and c_mag > 0:
                    sim = max(0.0, round(dot_product(query_vector, emb) / (q_mag * c_mag), 4))
                else:
                    sim = 0.0
                scored.append((chk, sim))
            except Exception:
                scored.append((chk, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


async def _retrieve_lexical_candidates(
    query: str,
    db: AsyncSession,
    document_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    limit: int = LEXICAL_TOP_N,
) -> List[Tuple[Chunk, float]]:
    """
    Stage 1 Sparse Lexical Search via PostgreSQL Full-Text Search (ts_rank_cd).
    Returns List[Tuple[Chunk, fts_rank_score]]
    """
    clean_query = re.sub(r"[^\w\s]", " ", query).strip()
    if not clean_query:
        return []

    ts_vec = func.to_tsvector("english", Chunk.content)
    ts_query = func.plainto_tsquery("english", clean_query)
    rank_col = func.ts_rank_cd(ts_vec, ts_query).label("rank")

    stmt = select(Chunk, rank_col).where(ts_vec.op("@@")(ts_query))

    if user_id is not None or document_id is not None:
        stmt = stmt.join(Document, Chunk.document_id == Document.id)

    if user_id is not None:
        stmt = stmt.where(Document.user_id == user_id)

    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)

    stmt = stmt.order_by(rank_col.desc()).limit(limit)
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        logger.info(
            "FTS returned 0 rows for query %r; skipping lexical path (vector-only fallback)",
            clean_query,
        )
        return []

    candidates = []
    for chunk, rank in rows:
        rank_val = float(rank) if rank is not None else 0.0
        candidates.append((chunk, rank_val))

    return candidates


def _reciprocal_rank_fusion(
    vector_candidates: List[Tuple[Chunk, float]],
    lexical_candidates: List[Tuple[Chunk, float]],
    k: int = RRF_K,
    top_fused_limit: int = RRF_FUSED_TOP_K,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF): Pure rank-based candidate pool generation.
    RRF_Score(d) = 1/(k + rank_vector) + 1/(k + rank_lexical)
    Deduplicates by chunk.id: a chunk surfaced by BOTH retrieval paths is merged
    into a single fused entry carrying both ranks (never passed to the LLM twice).
    Returns sorted list of fused candidate dicts.
    """
    fused_map: Dict[UUID, Dict] = {}

    for rank_idx, (chunk, vec_score) in enumerate(vector_candidates, start=1):
        c_id = chunk.id
        if c_id not in fused_map:
            fused_map[c_id] = {
                "chunk": chunk,
                "raw_vec_score": vec_score,
                "raw_lex_score": 0.0,
                "vec_rank": rank_idx,
                "lex_rank": None,
                "rrf_score": 0.0,
            }

    for rank_idx, (chunk, lex_score) in enumerate(lexical_candidates, start=1):
        c_id = chunk.id
        if c_id not in fused_map:
            fused_map[c_id] = {
                "chunk": chunk,
                "raw_vec_score": 0.0,
                "raw_lex_score": lex_score,
                "vec_rank": None,
                "lex_rank": rank_idx,
                "rrf_score": 0.0,
            }
        else:
            fused_map[c_id]["raw_lex_score"] = lex_score
            fused_map[c_id]["lex_rank"] = rank_idx

    for item in fused_map.values():
        rrf = 0.0
        if item["vec_rank"] is not None:
            rrf += 1.0 / (k + item["vec_rank"])
        if item["lex_rank"] is not None:
            rrf += 1.0 / (k + item["lex_rank"])
        item["rrf_score"] = rrf

    fused_list = list(fused_map.values())
    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_list[:top_fused_limit]


def _min_max_normalize(scores: List[float]) -> List[float]:
    """Min-Max normalize scores to [0.0, 1.0] range over the candidate pool."""
    if not scores:
        return []
    min_val, max_val = min(scores), max(scores)
    spread = max_val - min_val
    if spread < 1e-6:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [round((s - min_val) / spread, 4) for s in scores]


def _phrase_coverage_rerank(
    fused_candidates: List[Dict], query: str, final_top_k: int = FINAL_TOP_K
) -> List[Tuple[Chunk, float]]:
    """
    Stage 2 Re-Ranking (Option 2b - Phrase Coverage & Lexical Re-Scorer):
    Min-Max normalizes 3 independent feature signals over the fused RRF candidate pool (N=10):
      - S_vec_norm: Normalized pgvector dense semantic similarity
      - S_lex_norm: Normalized PostgreSQL FTS sparse lexical rank
      - S_phrase_norm: Normalized independent exact phrase & token coverage
    Applies weighted combination (0.50 S_vec_norm + 0.30 S_lex_norm + 0.20 S_phrase_norm).
    """
    if not fused_candidates:
        return []

    raw_vecs = [item.get("raw_vec_score", 0.0) for item in fused_candidates]
    raw_lexs = [item.get("raw_lex_score", 0.0) for item in fused_candidates]
    raw_phrases = [_compute_independent_phrase_coverage(query, item["chunk"].content) for item in fused_candidates]

    norm_vecs = _min_max_normalize(raw_vecs)
    norm_lexs = _min_max_normalize(raw_lexs)
    norm_phrases = _min_max_normalize(raw_phrases)

    reranked = []
    for i, item in enumerate(fused_candidates):
        chunk = item["chunk"]
        # Combined Min-Max Normalized Re-Rank Score across 3 independent signals
        final_score = round(
            (WEIGHT_VECTOR_NORM * norm_vecs[i])
            + (WEIGHT_LEXICAL_NORM * norm_lexs[i])
            + (WEIGHT_PHRASE_NORM * norm_phrases[i]),
            4,
        )
        # Surface the pre-normalization raw similarity for transparent
        # affordance-upstream diagnostics (citations display, eval reports).
        # Best-effort: RRF items always carry it; phase-coverage add-ons may
        # not, in which case the raw similarity is skipped, not fabricated.
        if chunk.metadata_ is not None and "raw_similarity" not in chunk.metadata_:
            if "raw_vec_score" in item:
                chunk.metadata_["raw_similarity"] = round(item["raw_vec_score"], 4)
        reranked.append((chunk, final_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:final_top_k]


async def retrieve_context(
    query: str,
    db: AsyncSession,
    document_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    top_k: int = FINAL_TOP_K,
) -> List[Tuple[Chunk, float, str]]:
    """
    High-Rigor Two-Stage Hybrid Retrieval Pipeline with Multi-Tenant User Isolation:
    1. Candidate Retrieval: pgvector HNSW (top 20) + PostgreSQL FTS (top 20) filtered by user_id
    2. Reciprocal Rank Fusion (RRF, k=60): Rank-based candidate pool generation (top 10 surviving)
    3. Min-Max Normalized Stage 2 Re-Ranking: Returns top `top_k` candidates (default 5) for LLM generation
    """
    start_time = time.time()
    logger.info(f"Executing Multi-Tenant Hybrid Retrieval for user {user_id}, query: '{query}'")

    # Cache-first: identical normalized queries within TTL skip embed + DB work.
    cached = query_cache.get(user_id, document_id, top_k, query)
    if cached is not None:
        logger.info(
            "Retrieval served from cache for user %s, query: '%s' (%d chunks)",
            user_id,
            query,
            len(cached),
        )
        return cached

    try:
        # 1. Fetch vector and lexical candidates concurrently.
        #    asyncpg does not support concurrent operations on the same session,
        #    so the lexical leg opens its own short-lived session.
        async def _run_lexical():
            async with make_session() as lex_db:
                return await _retrieve_lexical_candidates(
                    query=query, db=lex_db, document_id=document_id,
                    user_id=user_id, limit=LEXICAL_TOP_N,
                )

        vector_candidates, lexical_candidates = await asyncio.gather(
            _retrieve_vector_candidates(
                query=query, db=db, document_id=document_id, user_id=user_id, limit=VECTOR_TOP_N
            ),
            _run_lexical(),
        )

        stage1_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Stage 1 Concurrent Retrieval: {len(vector_candidates)} vector & {len(lexical_candidates)} lexical candidates in {stage1_ms}ms."
        )

        # 2. Reciprocal Rank Fusion (RRF) candidate selection (top 10 surviving)
        fused_candidates = _reciprocal_rank_fusion(
            vector_candidates=vector_candidates,
            lexical_candidates=lexical_candidates,
            k=RRF_K,
            top_fused_limit=RRF_FUSED_TOP_K,
        )

        # 3. Min-Max Normalized Stage 2 Re-Ranking (dynamic top_k from ChatRequest)
        reranked_top_k = _phrase_coverage_rerank(
            fused_candidates=fused_candidates, query=query, final_top_k=top_k
        )

        # 3b. Identity Query Guard: For queries asking about name/identity, guarantee document header (index=0) is included
        identity_pattern = re.compile(
            r"\b(name|who\s*am\s*i|whose\s*(resume|cv|profile|document)|who\s*is\s*this|my\s*name|candidate('s)?\s*name|smy\s*name)\b",
            re.IGNORECASE,
        )
        if identity_pattern.search(query):
            header_stmt = select(Chunk).where(Chunk.index == 0)
            if document_id is not None:
                header_stmt = header_stmt.where(Chunk.document_id == document_id)
            elif user_id is not None:
                header_stmt = header_stmt.join(Document, Chunk.document_id == Document.id).where(Document.user_id == user_id)
            header_res = await db.execute(header_stmt.limit(3))
            header_chunks = header_res.scalars().all()
            existing_ids = {c.id for c, _ in reranked_top_k}
            for hc in header_chunks:
                if hc.id not in existing_ids:
                    # Insert header chunk at the beginning for identity resolution
                    reranked_top_k.insert(0, (hc, 0.99))
                    existing_ids.add(hc.id)
            if len(reranked_top_k) > max(top_k, 5):
                reranked_top_k = reranked_top_k[:max(top_k, 5)]

        total_retrieval_ms = int((time.time() - start_time) * 1000)

        # 4. Resolve source document filenames and display_titles
        doc_ids = {chunk.document_id for chunk, _ in reranked_top_k}
        doc_map = {}
        if doc_ids:
            doc_res = await db.execute(
                select(Document.id, Document.filename, Document.display_title).where(Document.id.in_(doc_ids))
            )
            doc_map = {}
            for row in doc_res.all():
                d_id = row[0]
                fn = row[1]
                dt = row[2] if len(row) > 2 and row[2] else fn
                doc_map[d_id] = (fn, dt)

        retrieved = []
        for chunk, score in reranked_top_k:
            fn, dt = doc_map.get(chunk.document_id, ("unknown", "unknown"))
            meta = chunk.metadata_ or {}
            filename = meta.get("filename") or fn
            display_title = meta.get("display_title") or dt
            if chunk.metadata_ is not None:
                chunk.metadata_["display_title"] = display_title
            retrieved.append((chunk, score, filename))

        logger.info(
            f"Hybrid retrieval pipeline completed in {total_retrieval_ms}ms (Stage 1: {stage1_ms}ms, Stage 2 Re-Rank: {total_retrieval_ms - stage1_ms}ms). Returning {len(retrieved)} chunks."
        )

        query_cache.put(user_id, document_id, top_k, query, retrieved)

        return retrieved

    except Exception as e:
        logger.error(f"Error during hybrid context retrieval: {e}", exc_info=True)
        raise
