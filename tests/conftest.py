"""Shared test fixtures.

Tests run against a real Postgres database (pgvector/pg_search can't be faked).
The test database is *never* read from `DATABASE_URL` directly — it is derived
from it below by suffixing the database name with `_test`, so tests can never
accidentally run (and truncate/drop) whatever database `DATABASE_URL` points at
for the running app. The derived database, extensions, and tables are created
automatically; no manual `alembic upgrade head` step is required.

Embedding and LLM calls are faked via dependency overrides so unit tests never
hit Gemini. The fake embedder is deterministic (bag-of-words) so nearest-neighbour
ordering is predictable.
"""

import asyncio
import os
import re
from urllib.parse import urlsplit, urlunsplit

from minddrill.config import get_settings


def _resolve_test_database_url() -> str:
    """Derive the test DB URL from DATABASE_URL by suffixing the db name with `_test`.

    Never used verbatim: this exists so tests can't be pointed at a real database
    by accident. Aborts loudly if the resolved name doesn't end in `_test`.
    """
    base_url = get_settings().database_url
    parts = urlsplit(base_url)
    db_name = parts.path.lstrip("/")
    if not db_name:
        raise RuntimeError(
            f"DATABASE_URL has no database name to derive a test database from: {base_url!r}"
        )
    test_name = f"{db_name}_test"
    test_url = urlunsplit(
        (parts.scheme, parts.netloc, f"/{test_name}", parts.query, parts.fragment)
    )

    resolved_name = urlsplit(test_url).path.lstrip("/")
    if not resolved_name.endswith("_test"):
        raise RuntimeError(
            f"Refusing to run tests: resolved database name {resolved_name!r} "
            "does not end in '_test'. Tests must never run against a non-test database."
        )
    return test_url


# Must happen before any import that creates the DB engine (minddrill.db.session,
# minddrill.main, ...), so the whole app wires up against the test database.
os.environ["DATABASE_URL"] = _resolve_test_database_url()
get_settings.cache_clear()

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from minddrill.db.session import Base, engine  # noqa: E402
from minddrill.main import app  # noqa: E402
from minddrill.models import chunk as _chunk  # noqa: E402,F401  registers Chunk on Base.metadata
from minddrill.models import document as _document  # noqa: E402,F401  registers Document
from minddrill.models import ingestion_job as _ingestion_job  # noqa: E402,F401  registers IngestionJob
from minddrill.models import user as _user  # noqa: E402,F401  registers User on Base.metadata
from minddrill.providers.gemini import get_llm  # noqa: E402
from minddrill.rag.embedder import get_embedder  # noqa: E402
from minddrill.worker import tasks as _tasks  # noqa: E402
from minddrill.worker.celery_app import celery_app  # noqa: E402


async def _init_test_schema() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
        await conn.run_sync(Base.metadata.create_all)
    # Runs in its own event loop, separate from the one pytest-asyncio uses for
    # tests; dispose the pool so no connection is reused across loops.
    await engine.dispose()


asyncio.run(_init_test_schema())


@pytest.fixture(autouse=True)
async def _clean_tables():
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE users, documents, chunks, ingestion_jobs CASCADE")
        )
    yield


@pytest.fixture(autouse=True)
def _celery_eager(monkeypatch):
    """Run ingestion tasks inline, with the deterministic fake embedder.

    The FastAPI dependency override only reaches the endpoint; the Celery task
    resolves its own embedder, so we patch that seam too.
    """
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    monkeypatch.setattr(_tasks, "get_embedder", lambda: FakeEmbedder())


def _bag_of_words(s: str) -> list[float]:
    dim = get_settings().embed_dim
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", s.lower()):
        vec[hash(word) % dim] += 1.0
    if not any(vec):
        vec[0] = 1.0  # pgvector rejects an all-zero cosine vector
    return vec


class FakeEmbedder:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_bag_of_words(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _bag_of_words(text)


class FakeLLM:
    """Records the prompt it was given and returns a canned grounded answer."""

    def __init__(self) -> None:
        self.last_messages: list[dict] | None = None

    async def stream(self, messages, **kwargs):
        yield "canned answer [1]"

    async def generate(self, messages, **kwargs) -> str:
        self.last_messages = list(messages)
        return "canned answer [1]"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def client(embedder: FakeEmbedder, llm: FakeLLM):
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_llm] = lambda: llm
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_embedder, None)
    app.dependency_overrides.pop(get_llm, None)


async def register_user(
    c: httpx.AsyncClient, username: str, password: str = "hunter2"
) -> tuple[str, dict]:
    """Register + login; return (user_id, auth headers)."""
    reg = await c.post(
        "/api/v1/auth/register", json={"username": username, "password": password}
    )
    user_id = reg.json()["user_id"]
    login = await c.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    token = login.json()["access_token"]
    return user_id, {"Authorization": f"Bearer {token}"}
