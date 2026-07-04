"""Async SQLAlchemy engine, session factory, and the declarative `Base`.

Everything is async end to end (asyncpg driver) per CLAUDE.md — no blocking calls
on the event loop. `Base` is the metadata target for Alembic; models register
against it in their owning slices.
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from minddrill.config import get_settings


class Base(DeclarativeBase):
    """Declarative base; `Base.metadata` is Alembic's `target_metadata`."""


engine = create_async_engine(get_settings().database_url)

SessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped async session."""
    async with SessionLocal() as session:
        yield session
