"""
KueryCore AI — PasswordResetToken Model
Single-use, short-lived tokens for the forgot-password flow.

Security design:
 • Only the SHA-256 hash of the raw token is stored at rest.
 • The raw token (secrets.token_urlsafe(32)) is transmitted exactly once —
   in the reset-password email link — and never persisted in plaintext.
 • expires_at is set to UTC+30 minutes at creation time.
 • used_at is set to UTC now on the first valid redemption; subsequent
   redemptions with the same token hash are rejected (single-use guarantee).
 • Any previous unused tokens for the same user are hard-deleted on each
   new request, so at most one live token exists per user at any time.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PasswordResetToken(Base):
    """Single-use, 30-minute-expiry password reset token (hash-only at rest)."""

    __tablename__ = "password_reset_tokens"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # SHA-256 hex digest of the raw token — never the raw value itself.
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


