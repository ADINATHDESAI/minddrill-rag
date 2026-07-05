"""Shared test fixtures.

Tests run against a real Postgres database (pgvector/pg_search can't be faked),
pointed at by `DATABASE_URL` — expected to be a dedicated test database (e.g.
`minddrill_test`), migrated via `alembic upgrade head` before running `pytest -q`.
Never point this at a database with real data: the fixture below truncates it.

Embedding and LLM calls are faked via dependency overrides so unit tests never
hit Gemini. The fake embedder is deterministic (bag-of-words) so nearest-neighbour
ordering is predictable.
"""

import re

import httpx
import pytest
from sqlalchemy import text

from minddrill.config import get_settings
from minddrill.db.session import engine
from minddrill.main import app
from minddrill.models import chunk as _chunk  # noqa: F401  registers Chunk on Base.metadata
from minddrill.models import document as _document  # noqa: F401  registers Document
from minddrill.models import user as _user  # noqa: F401  registers User on Base.metadata
from minddrill.providers.gemini import get_llm
from minddrill.rag.embedder import get_embedder


@pytest.fixture(autouse=True)
async def _clean_tables():
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE users, documents, chunks CASCADE"))
    yield


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
