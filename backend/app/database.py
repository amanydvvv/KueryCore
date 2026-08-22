"""
KueryCore AI — Database Setup
Async SQLAlchemy engine + session factory with pgvector support.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import get_settings

settings = get_settings()

# We use pgvector with an HNSW index (see app/models/__init__.py).
# Using HNSW avoids the "0 rows returned on small tables" issue that ivfflat suffers from
# when probed with standard asyncpg prepared statements, meaning we do NOT need to disable
# statement_cache_size (which would hurt performance).
import sys
import os
from sqlalchemy.pool import NullPool

is_testing = (
    "pytest" in sys.modules
    or bool(os.environ.get("PYTEST_CURRENT_TEST"))
    or bool(os.environ.get("CI"))
)

engine_kwargs = {
    "echo": settings.DEBUG,
    "pool_pre_ping": True,
    "connect_args": {
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    },
}

if is_testing:
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20
    engine_kwargs["pool_recycle"] = 300


engine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields a database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create tables and enable pgvector extension if available."""
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Dispose engine connections on shutdown."""
    await engine.dispose()
