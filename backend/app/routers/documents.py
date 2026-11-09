"""
KueryCore AI — Document Management Router
Upload, list, detail, delete, and reindex documents with user tenant isolation.
"""

import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import Response
from app.core.ratelimit import limiter
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Document, Chunk
from app.models.user import User
from app.core.security import get_current_user
from app.schemas import DocumentResponse, DocumentListResponse
from app.services.ingestion import ingest_document
from app.services.query_cache import query_cache

settings = get_settings()
router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# Upload directory — single source of truth is settings.UPLOAD_DIR (config.py),
# which resolves to the OS-native temp dir + kuerycore_uploads.
UPLOAD_DIR = Path(settings.UPLOAD_DIR)


def _allowed_extension(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.ALLOWED_EXTENSIONS


def _file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return "markdown" if ext == "md" else ext


def _compute_hash(content: bytes) -> str:
    """SHA-256 hex digest of document content used for deduplication."""
    return hashlib.sha256(content).hexdigest()


@limiter.limit("5/minute")
@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF or Markdown file for ingestion scoped to current user."""
    if not file.filename or not _allowed_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    file_size = len(content)

    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE_MB}MB",
        )

    content_hash = _compute_hash(content)

    # Check for duplicate for this user
    existing = await db.execute(
        select(Document).where(
            Document.content_hash == content_hash,
            Document.user_id == current_user.id
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="This document has already been uploaded to your workspace.",
        )

    # Create document record with raw bytes stored in DB
    file_id = uuid.uuid4()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    doc = Document(
        id=file_id,
        user_id=current_user.id,
        filename=file.filename,
        content_hash=content_hash,
        file_type=_file_type(file.filename),
        file_size=file_size,
        status="pending",
        raw_bytes=content,  # Persist PDF bytes in database
    )
    db.add(doc)
    await db.flush()

    # Corpus changed: cached retrieval results for this tenant are stale.
    query_cache.invalidate_user(current_user.id)

    # Create temp file ONLY for ingestion pipeline (auto-cleanup by ingestion worker)
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    # Queue background ingestion with the temporary file path
    background_tasks.add_task(ingest_document, str(file_id), temp_path)

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        display_title=doc.display_title,
        file_type=doc.file_type,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        error_message=doc.error_message,
        chunk_count=0,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all documents owned by current authenticated user."""
    chunk_counts = (
        select(Chunk.document_id, func.count(Chunk.id).label("chunk_count"))
        .group_by(Chunk.document_id)
        .subquery()
    )

    result = await db.execute(
        select(Document, func.coalesce(chunk_counts.c.chunk_count, 0).label("chunk_count"))
        .outerjoin(chunk_counts, Document.id == chunk_counts.c.document_id)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )

    documents = []
    for row in result.all():
        doc = row[0]
        count = row[1]
        documents.append(
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                display_title=doc.display_title,
                file_type=doc.file_type,
                file_size=doc.file_size,
                page_count=doc.page_count,
                status=doc.status,
                error_message=doc.error_message,
                chunk_count=count,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
        )

    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get document details if owned by current user."""
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    )
    chunk_count = chunk_result.scalar() or 0

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        display_title=doc.display_title,
        file_type=doc.file_type,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        error_message=doc.error_message,
        chunk_count=chunk_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve the raw PDF file for an owned document (PDF citation viewer).

    Tenant-scoped like every other route: a document owned by another user
    (or deleted) is indistinguishable and returns 404 so no ownership leaks.
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.file_type != "pdf":
        raise HTTPException(
            status_code=415, detail="Only PDF documents can be viewed in the document viewer."
        )

    if not doc.raw_bytes:
        # Legacy fallback — mainly useful for anything the rescue script missed,
        # not the primary recovery path
        legacy_path = UPLOAD_DIR / f"{doc.id}.pdf"
        if legacy_path.exists():
            doc.raw_bytes = legacy_path.read_bytes()
            await db.commit()
        else:
            logger.warning("PDF bytes missing in DB and no legacy file for document %s (user %s)", doc.id, current_user.id)
            raise HTTPException(
                status_code=410,
                detail="The source file for this document is no longer available.",
            )

    return Response(
        content=doc.raw_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc.filename}"'},
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document owned by current user."""
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Use direct DML delete so PostgreSQL ON DELETE CASCADE handles chunk
    # removal without loading all chunk rows into the ORM session first.
    await db.execute(
        delete(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    await db.commit()

    # Corpus changed: cached retrieval results for this tenant are stale.
    query_cache.invalidate_user(current_user.id)


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-chunk and re-embed a document owned by current user."""
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not doc.raw_bytes:
        raise HTTPException(
            status_code=410,
            detail="Cannot reindex: source file bytes not available in storage.",
        )

    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    doc.status = "pending"
    doc.error_message = None
    await db.commit()

    # Chunks replaced: cached retrieval results for this tenant are stale.
    query_cache.invalidate_user(current_user.id)

    # Write raw_bytes to a temp file for ingestion pipeline
    ext = doc.file_type if doc.file_type != "markdown" else "md"
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(doc.raw_bytes)
        temp_path = tmp.name

    background_tasks.add_task(ingest_document, str(document_id), temp_path)

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        display_title=doc.display_title,
        file_type=doc.file_type,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        error_message=doc.error_message,
        chunk_count=0,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )
