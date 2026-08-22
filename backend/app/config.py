"""
KueryCore AI — Configuration
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import secrets
import tempfile
from pathlib import Path

from pydantic import field_validator, model_validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = "KueryCore AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://kuerycore:kuerycore_dev@localhost:5435/kuerycore"
    DATABASE_URL_SYNC: str = "postgresql://kuerycore:kuerycore_dev@localhost:5435/kuerycore"
    
    @field_validator("DATABASE_URL")
    @classmethod
    def sanitize_database_url(cls, v: str) -> str:
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # LLM & Embeddings
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 768
    GENERATIVE_MODEL: str = "openai/gpt-oss-20b"
    # Judge model for the eval harness. Deliberately a different family from
    # the primary generation model so an answer is never graded by the same
    # model that produced it (avoids same-model self-grading bias).
    # NOTE: defaulted to openai/gpt-oss-20b because qwen3-32b returned 404
    # ("model_not_found") on the actual Groq org during baseline runs - the
    # plan's documented migration path (§1.4) for exactly this situation.
    EVAL_JUDGE_MODEL: str = "openai/gpt-oss-20b"
    # Groq vision model used for OCR fallback on scanned (image-only) PDF pages.
    # Default tracks Groq's current vision-capable model.
    VISION_MODEL: str = "qwen/qwen3.6-27b"
    # Fast model used by the ingestion pipeline to generate human-readable document
    # titles. Speed matters more than capability here — a cheap 8b model is sufficient.
    TITLE_GENERATION_MODEL: str = "llama-3.1-8b-instant"

    # RAG Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    AGENTIC_ROUTER_ENABLED: bool = True


    # Guardrails (plan v3, Part 2)
    # GUARDRAILS_ENABLED is the master kill switch for all input/output
    # guardrail rules (PII sanitization, injection block, output checks).
    # GUARDRAILS_STRICT additionally enables obfuscation-pattern detection
    # (base64-looking runs, unusual unicode) — off by default because it
    # trades false positives for extra coverage.
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_STRICT: bool = False

    # Retrieval Cache
    # TTL for cached (user_id, normalized_query) retrieval results.
    CACHE_TTL_SECONDS: int = 600
    # Max distinct (user_id, normalized_query) entries kept per process.
    CACHE_MAX_ENTRIES: int = 300

    # Environment: "development" auto-generates a JWT secret; anything else
    # (default "production") requires JWT_SECRET_KEY to be explicitly set.
    ENVIRONMENT: str = "production"

    # Auth
    JWT_SECRET_KEY: str | None = None

    # Upload
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = ["pdf", "md", "txt"]
    # Single source of truth for the upload directory. OS-native temp dir
    # (survives non-root containers on Render and varies safely by platform).
    UPLOAD_DIR: str = str(Path(tempfile.gettempdir()) / "kuerycore_uploads")

    # Email Providers
    # 1. Brevo REST API / SMTP (https://brevo.com — 300 free emails/day to anyone)
    BREVO_API_KEY: str | None = None

    # 2. SMTP (e.g. Gmail with Google App Password — 500 free emails/day)
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_FROM_NAME: str = "KueryCore AI"
    SMTP_FROM_EMAIL: str | None = None

    # 3. Resend REST API (https://resend.com)
    RESEND_API_KEY: str | None = None
    RESEND_FROM_EMAIL: str = "KueryCore <noreply@kuerycore.ai>"

    # Frontend base URL — used to build password-reset links in emails.
    FRONTEND_URL: str = "http://localhost:5173"

    # CORS
    # Explicit allowlist. Dynamic Vercel preview origins are additionally
    # covered by allow_origin_regex in main.py; this list carries the stable
    # origins (local dev + production frontend).
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://docu-mind-ai-iota.vercel.app"
    ]

    @model_validator(mode="after")
    def enforce_jwt_secret(self) -> "Settings":
        # Never fall back to a fixed, predictable secret. In production (any
        # non-development environment) an unset JWT_SECRET_KEY is a hard startup
        # error so the app fails loudly instead of silently signing tokens with
        # an insecure default. Only explicit ENVIRONMENT=development may
        # auto-generate a random secret for local convenience.
        if not self.JWT_SECRET_KEY:
            if self.ENVIRONMENT.lower() == "development":
                self.JWT_SECRET_KEY = secrets.token_hex(32)
            else:
                raise ValueError(
                    "JWT_SECRET_KEY must be set. Export a random value (e.g. "
                    "`python -c \"import secrets; print(secrets.token_hex(32))\"`) "
                    "or set ENVIRONMENT=development to auto-generate one for local dev."
                )
        return self

    @model_validator(mode="after")
    def backfill_gemini_key(self) -> "Settings":
        # Accept legacy GOOGLE_API_KEY as the Gemini key when GEMINI_API_KEY
        # is absent, so one, both, or neither can be supplied without crashing.
        if not self.GEMINI_API_KEY and self.GOOGLE_API_KEY:
            self.GEMINI_API_KEY = self.GOOGLE_API_KEY
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        # Ignore unknown/legacy env vars instead of crashing startup.
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
