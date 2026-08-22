"""
KueryCore AI — Pydantic Schemas
Request/response models for API validation and serialization.
"""

from datetime import datetime
from uuid import UUID
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
#  Documents
# ──────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Document metadata returned from API."""
    id: UUID
    filename: str
    display_title: Optional[str] = None
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: list[DocumentResponse]
    total: int


# ──────────────────────────────────────────────
#  Chat / Q&A
# ──────────────────────────────────────────────

class Citation(BaseModel):
    """A single citation linking an answer claim to a source chunk."""
    chunk_id: UUID
    document_id: UUID
    filename: str
    display_title: Optional[str] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    score: float
    content_preview: str = Field(..., max_length=300)
    # How the chunk text was obtained: "text" (extracted from the file's
    # text layer) or "ocr" (read from the page image by a vision model).
    # Lets the UI label OCR-derived content so users can weigh its trust.
    source: Optional[str] = None


class ChatRequest(BaseModel):
    """User question sent to the RAG engine."""
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Complete (non-streaming) response with answer and citations."""
    answer: str
    citations: list[Citation]
    conversation_id: UUID
    latency_ms: int
    avg_similarity: float
    intent: Optional[str] = None



# ──────────────────────────────────────────────
#  Conversations
# ──────────────────────────────────────────────

class MessageResponse(BaseModel):
    """A single message in a conversation."""
    id: UUID
    role: str
    content: str
    citations: Optional[list[Citation]] = None
    latency_ms: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Conversation with its messages."""
    id: UUID
    title: Optional[str] = None
    messages: list[MessageResponse] = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """List of conversation summaries (without full messages)."""
    conversations: list[ConversationResponse]
    total: int


# ──────────────────────────────────────────────
#  Analytics
# ──────────────────────────────────────────────

class QueryLogResponse(BaseModel):
    """A single query log entry."""
    id: UUID
    question: str
    top_k: int
    avg_similarity: float
    latency_ms: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalyticsSummary(BaseModel):
    """Aggregate analytics stats."""
    total_queries: int
    avg_latency_ms: float
    avg_similarity: float
    total_documents: int
    total_chunks: int


class DocumentQueryFrequency(BaseModel):
    """How often a document's chunks are retrieved."""
    document_id: UUID
    filename: str
    query_count: int


# ──────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Server health check response."""
    status: str  # 'healthy' | 'degraded' | 'unhealthy'
    database: str
    llm_provider: str           # embedding model (Gemini)
    generative_model: str = "unconfigured"  # chat generation model (Groq cascade)
    version: str

