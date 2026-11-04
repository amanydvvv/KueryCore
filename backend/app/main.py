"""
KueryCore AI — FastAPI Application Entry Point
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db, close_db
from app.schemas import HealthResponse
from app.routers import (
    documents_router,
    chat_router,
    conversations_router,
    analytics_router,
    auth_router,
)

settings = get_settings()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle hooks."""
    import asyncio

    # --- Database ---
    # Wrapped in try/except so a transient connection hiccup at startup
    # doesn't crash-loop the service. Tables are created by alembic upgrade
    # head (Render Pre-Deploy Command); init_db() here only enables the
    # pgvector extension and is safe to retry on the next request.
    try:
        await init_db()
    except Exception as e:
        logger.critical(
            "Database init failed at startup: %s -- server starting in degraded mode.", e
        )

    # --- Startup model smoke-test ---
    # Runs in non-development environments only. Fails loudly (CRITICAL log)
    # but does NOT crash the server — a degraded server (DB up, LLM temporarily
    # unreachable) is preferable to a hard 502 for every user on Render.
    # The /api/health endpoint will report generative_model: "unreachable" so
    # monitoring alerts fire without taking the whole service down.
    if settings.ENVIRONMENT.lower() != "development":
        try:
            from app.services.generation import get_llm
            llm = get_llm()
            await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, "Reply with one word: OK"),
                timeout=10.0,
            )
            logger.info(
                "Startup smoke-test PASSED — primary model: %s",
                settings.GENERATIVE_MODEL,
            )
        except Exception as e:
            logger.critical(
                "Startup smoke-test FAILED for model %s: %s — "
                "server starting in degraded mode. Check GENERATIVE_MODEL env var.",
                settings.GENERATIVE_MODEL,
                e,
            )

    yield
    # --- Shutdown ---
    await close_db()


app = FastAPI(
    title="KueryCore API",
    version=settings.APP_VERSION,
    description="AI-powered technical documentation assistant with RAG",
    lifespan=lifespan,
    redirect_slashes=False,
)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# CORS
origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"]
# Apply strict origin enforcement and disable credentials unless explicitly needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" in origins else origins,
    # Updated regex to only allow specific trusted domains (adjust as needed).
    allow_origin_regex=r"^https://([a-z0-9-]+-[a-z0-9]+-[a-z0-9]+\.vercel\.app|docu-mind-ai-iota\.vercel\.app)$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Middleware to log CORS rejections (status 403) for unexpected origins.
@app.middleware("http")
async def log_cors_rejections(request: Request, call_next):
    response = await call_next(request)
    origin = request.headers.get("origin")
    if origin and response.status_code == 403:
        logger.warning("CORS rejection for origin: %s", origin)
    return response

# Rate limiting
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from app.core.ratelimit import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429, content={"detail": "Too many requests. Please slow down and try again later."}
))
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred."}
    )

# Register routers
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(analytics_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check(quick: bool = False):
    """Check server, database, embedding provider, and generative model connectivity.

    quick=true  -- DB-only check (SELECT 1). Skips Gemini + Groq calls.
                  Used by keep-alive cron to avoid burning API quotas.
    quick=false (default) -- full check: DB + embedding + generative model.
    """
    import asyncio
    from sqlalchemy import text
    from app.database import async_session

    # --- Database check ---
    db_status = "unhealthy"
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
            db_status = "healthy"
    except Exception as e:
        logger.warning("DB health check failed: %s", e)
        db_status = "unhealthy"

    # --- Quick mode: DB-only (skips Gemini + Groq). Used by keep-alive cron. ---
    if quick:
        return HealthResponse(
            status="healthy" if db_status == "healthy" else "unhealthy",
            database=db_status,
            llm_provider="skipped",
            generative_model="skipped",
            migration_version=migration_version,
            version=settings.APP_VERSION,
        )

    # --- Embedding provider check (Gemini) ---
    llm_status = "unhealthy"
    if settings.GEMINI_API_KEY:
        try:
            from app.services.retrieval import embeddings
            await asyncio.wait_for(
                asyncio.to_thread(
                    embeddings.embed_query,
                    "healthcheck",
                    output_dimensionality=settings.EMBEDDING_DIMENSION,
                ),
                timeout=4.0,
            )
            llm_status = "healthy"
        except Exception:
            llm_status = "unreachable"

    # --- Generative model check (Groq cascade primary) ---
    # Short prompt, 5s timeout — tests the configured GENERATIVE_MODEL specifically.
    gen_status = "unconfigured"
    if settings.GROQ_API_KEY or settings.GEMINI_API_KEY:
        try:
            from app.services.generation import get_llm
            llm = get_llm()
            await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, "Reply with one word: OK"),
                timeout=5.0,
            )
            gen_status = "healthy"
        except Exception as e:
            logger.warning("Generative model health check failed: %s", e)
            gen_status = "unreachable"

    # --- Overall status ---
    if db_status != "healthy":
        overall = "unhealthy"
    elif llm_status != "healthy" or gen_status not in ("healthy", "unconfigured"):
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthResponse(
        status=overall,
        database=db_status,
        llm_provider=llm_status,
        generative_model=gen_status,
        version=settings.APP_VERSION,
    )
