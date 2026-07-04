"""Shared test fixtures.

Tests run against a real Postgres database (pgvector/pg_search can't be faked),
pointed at by `DATABASE_URL` — expected to be a dedicated test database (e.g.
`minddrill_test`), migrated via `alembic upgrade head` before running `pytest -q`.
Never point this at a database with real data: the fixture below truncates it.
"""

import pytest
from sqlalchemy import text

from minddrill.db.session import engine
from minddrill.models import user as _user  # noqa: F401  registers User on Base.metadata


@pytest.fixture(autouse=True)
async def _clean_users_table():
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE users CASCADE"))
    yield
