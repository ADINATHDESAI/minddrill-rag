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

from minddrill.config import get_settings
from minddrill.db.testing import resolve_test_database_url

# Must happen before any import that creates the DB engine (minddrill.db.session,
# minddrill.main, ...), so the whole app wires up against the test database.
os.environ["DATABASE_URL"] = resolve_test_database_url()
# Tests don't depend on the dev secret's strength (or even its presence) — pin a
# fixed, sufficiently long one so HS256 signing never warns about a weak key.
os.environ["JWT_SECRET"] = "test-only-jwt-signing-secret-not-for-production-0123456789"
# Tracing is exercised against a recording fake (see `FakeLangfuse` below), never
# the real client — blank the keys so nothing here can ever reach real Langfuse,
# regardless of what's configured in the developer's own .env.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
get_settings.cache_clear()

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from minddrill.db.session import SessionLocal, engine  # noqa: E402
from minddrill.db.testing import init_test_schema  # noqa: E402
from minddrill.main import app  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from minddrill.agent.model import get_agent_model, get_fallback_model  # noqa: E402
from minddrill.providers.failover import get_providers  # noqa: E402
from minddrill.rag.embedder import get_embedder  # noqa: E402
from minddrill.rag.reranker import get_reranker  # noqa: E402
from minddrill.worker import tasks as _tasks  # noqa: E402
from minddrill.worker.celery_app import celery_app  # noqa: E402

# Runs in its own event loop, separate from the one pytest-asyncio uses for
# tests; init_test_schema disposes the pool so no connection leaks across loops.
asyncio.run(init_test_schema())


@pytest.fixture(autouse=True)
async def _clean_tables():
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE TABLE users, documents, chunks, ingestion_jobs, "
                "sessions, messages CASCADE"
            )
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
    """A single fake provider: streams canned grounded tokens, records the prompt."""

    def __init__(self, tokens=("canned ", "answer ", "[1]")) -> None:
        self.tokens = tuple(tokens)
        self.last_messages: list[dict] | None = None
        self.streamed = False
        self.last_usage = {"input_tokens": 3, "output_tokens": len(self.tokens)}

    async def stream(self, messages, **kwargs):
        self.last_messages = list(messages)
        self.streamed = True
        for tok in self.tokens:
            yield tok

    async def generate(self, messages, **kwargs) -> str:
        return "".join([tok async for tok in self.stream(messages, **kwargs)])


class FakeReranker:
    """Records that it ran and returns the fused order (scored), clamped to top_n.

    The fixed high score keeps the grounding gate open for the happy path;
    ordering semantics are exercised directly in test_reranker.py.
    """

    def __init__(self, score: float = 5.0) -> None:
        self.calls: list[int] = []
        self.score = score

    async def rerank(self, query: str, chunks: list, top_n: int) -> list:
        self.calls.append(len(chunks))
        top = chunks[: min(top_n, len(chunks))]
        return [(c, self.score) for c in top]


class FakeSpan:
    """Records what production code reported on one observation."""

    def __init__(
        self, name: str, as_type: str, trace_id: str | None, input=None, model=None
    ) -> None:
        self.name = name
        self.as_type = as_type
        self.trace_id = trace_id
        self.input = input
        self.model = model
        self.output = None
        self.metadata: dict = {}
        self.usage_details: dict | None = None
        self.level: str | None = None
        self.status_message: str | None = None
        self.ended = False

    def update(
        self,
        *,
        output=None,
        metadata=None,
        usage_details=None,
        level=None,
        status_message=None,
        **_kwargs,
    ) -> "FakeSpan":
        if output is not None:
            self.output = output
        if metadata:
            self.metadata.update(metadata)
        if usage_details is not None:
            self.usage_details = usage_details
        if level is not None:
            self.level = level
        if status_message is not None:
            self.status_message = status_message
        return self


class _FakeSpanContext:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, *exc_info) -> bool:
        self.span.ended = True
        return False


class FakeLangfuse:
    """Records every span opened via `start_as_current_observation`.

    Mirrors the one method our code calls (always as `with ... as span:`), so
    tests can assert on trace id, span names/types, nesting order, and what got
    attached to each span without touching the real Langfuse client/network.
    """

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_as_current_observation(
        self,
        *,
        trace_context=None,
        name: str,
        as_type: str = "span",
        input=None,
        model=None,
        **_kwargs,
    ) -> _FakeSpanContext:
        trace_id = trace_context["trace_id"] if trace_context else None
        span = FakeSpan(name, as_type, trace_id, input=input, model=model)
        self.spans.append(span)
        return _FakeSpanContext(span)

    def flush(self) -> None:
        pass


@pytest.fixture
def langfuse_fake(monkeypatch) -> FakeLangfuse:
    """Swap the Langfuse client seam for a recording fake in both hot paths."""
    from minddrill.agent import loop as agent_loop
    from minddrill.rag import retrieve as rag_retrieve

    fake = FakeLangfuse()
    monkeypatch.setattr(rag_retrieve, "get_langfuse", lambda: fake)
    monkeypatch.setattr(agent_loop, "get_langfuse", lambda: fake)
    return fake


class FakeToolModel(FakeMessagesListChatModel):
    """A scripted chat model for the agent loop.

    `responses` is a list of `AIMessage`s returned in order — script tool calls
    by giving an AIMessage with `tool_calls`. `bind_tools` is a no-op so
    `create_agent` can bind the tool schemas without a real provider.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@pytest.fixture
def agent_model() -> FakeToolModel:
    return FakeToolModel(responses=[AIMessage(content="canned answer [1]")])


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
async def db_session():
    """A raw async session for exercising retrieval directly, without the API."""
    async with SessionLocal() as session:
        yield session


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def client(
    embedder: FakeEmbedder,
    llm: FakeLLM,
    reranker: FakeReranker,
    agent_model: FakeToolModel,
):
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_providers] = lambda: [llm]
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_agent_model] = lambda: agent_model
    app.dependency_overrides[get_fallback_model] = lambda: None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_embedder, None)
    app.dependency_overrides.pop(get_providers, None)
    app.dependency_overrides.pop(get_reranker, None)
    app.dependency_overrides.pop(get_agent_model, None)
    app.dependency_overrides.pop(get_fallback_model, None)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into an ordered list of (event, data) pairs."""
    import json

    events: list[tuple[str, dict]] = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        if not block.strip():
            continue
        name = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        events.append((name, json.loads(data) if data else None))
    return events


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
