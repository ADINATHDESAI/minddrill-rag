"""Streaming query path: provider failover, grounding decline, disconnect."""

import time
import uuid

import httpx
import pytest

from minddrill.main import app
from minddrill.providers.failover import (
    ProvidersUnavailable,
    get_providers,
    open_stream,
)
from minddrill.rag.reranker import get_reranker
from minddrill.rag.retrieve import AnswerPlan, stream_answer
from tests._pdfgen import make_pdf
from tests.conftest import FakeReranker, parse_sse, register_user

_PDF_LINES = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
]


async def _ingest(client: httpx.AsyncClient, headers: dict, tmp_path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf(_PDF_LINES))
    resp = await client.post(
        "/api/v1/ingest",
        json={"source_type": "pdf", "source_uri": str(path)},
        headers=headers,
    )
    assert resp.status_code == 202, resp.text


class _OkProvider:
    def __init__(self, tokens=("hello ", "world")) -> None:
        self.tokens = tuple(tokens)
        self.streamed = False
        self.last_usage = {"input_tokens": 1, "output_tokens": len(self.tokens)}

    async def stream(self, messages, **kwargs):
        self.streamed = True
        for tok in self.tokens:
            yield tok


class _FailBeforeFirstToken:
    """Raises a 429-looking error before yielding any token."""

    def __init__(self) -> None:
        self.streamed = False

    async def stream(self, messages, **kwargs):
        self.streamed = True
        raise RuntimeError("429 quota exceeded")
        yield  # pragma: no cover


# --- open_stream unit tests -------------------------------------------------


async def test_open_stream_commits_to_primary():
    ok = _OkProvider()
    other = _OkProvider(tokens=("x",))
    tokens, provider = await open_stream([ok, other], [])
    assert [t async for t in tokens] == ["hello ", "world"]
    assert provider is ok
    assert not other.streamed  # primary produced a token; fallback never touched


async def test_open_stream_fails_over_before_first_token():
    primary = _FailBeforeFirstToken()
    fallback = _OkProvider()
    tokens, provider = await open_stream([primary, fallback], [])
    assert [t async for t in tokens] == ["hello ", "world"]
    assert provider is fallback
    assert primary.streamed  # it was tried, and failed before a token


async def test_open_stream_all_down_raises():
    with pytest.raises(ProvidersUnavailable):
        await open_stream([_FailBeforeFirstToken(), _FailBeforeFirstToken()], [])


# --- SSE-level failover / 503 ----------------------------------------------


async def test_query_fails_over_to_second_provider(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "failover_asker")
    await _ingest(client, headers, tmp_path)

    fallback = _OkProvider()
    app.dependency_overrides[get_providers] = lambda: [
        _FailBeforeFirstToken(),
        fallback,
    ]
    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    tokens = [d["text"] for name, d in events if name == "token"]
    assert "".join(tokens) == "hello world"
    assert fallback.streamed


async def test_query_all_providers_down_returns_503(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "alldown_asker")
    await _ingest(client, headers, tmp_path)

    app.dependency_overrides[get_providers] = lambda: [
        _FailBeforeFirstToken(),
        _FailBeforeFirstToken(),
    ]
    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )

    # Failover resolves before the stream opens, so this is a plain JSON 503.
    assert resp.status_code == 503
    assert not resp.headers["content-type"].startswith("text/event-stream")
    assert resp.json()["error"]["code"] == "providers_unavailable"


# --- grounding gate ---------------------------------------------------------


async def test_weak_context_declines_without_streaming(
    client: httpx.AsyncClient, llm, tmp_path
) -> None:
    _, headers = await register_user(client, "weak_asker")
    await _ingest(client, headers, tmp_path)

    # Retrieval returns chunks, but every rerank score is below the grounding
    # floor (0.0) -> decline instead of generating.
    app.dependency_overrides[get_reranker] = lambda: FakeReranker(score=-1.0)
    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    names = [name for name, _ in parse_sse(resp.text)]
    assert "decline" in names
    assert "token" not in names
    assert not llm.streamed


# --- client disconnect ------------------------------------------------------


class _CountingTokens:
    def __init__(self, n: int) -> None:
        self.n = n
        self.pulled = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self.pulled >= self.n:
            raise StopAsyncIteration
        self.pulled += 1
        return f"t{self.pulled}"

    async def aclose(self) -> None:
        self.closed = True


async def test_disconnect_closes_token_stream():
    """A client drop makes sse-starlette cancel the body generator; simulate that
    by aclose()-ing it mid-stream and assert the upstream token stream is closed.
    """
    plan = AnswerPlan(
        start=time.perf_counter(),
        user_id=uuid.uuid4(),
        request_id="req-disconnect",
        sources=[{"id": 1, "chunk_id": "c", "document_id": "d", "score": 1.0}],
        messages=[{"role": "user", "content": "q"}],
    )
    tokens = _CountingTokens(n=10)
    gen = stream_answer(plan, tokens, provider=_OkProvider(), ttft_ms=1, failover=False)

    # Drive the events sse-starlette would send: sources, status, first token.
    assert (await gen.__anext__())["event"] == "sources"
    assert (await gen.__anext__())["event"] == "status"
    assert (await gen.__anext__())["event"] == "token"

    # Client drops -> the transport cancels the generator via aclose().
    await gen.aclose()

    assert tokens.closed  # upstream provider stream released to stop token spend
    assert tokens.pulled < 10  # did not drain the whole stream


async def test_stream_error_closes_token_stream():
    """A mid-stream provider error emits `error` and still closes the stream."""

    class _BoomTokens:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise RuntimeError("upstream exploded")

        async def aclose(self) -> None:
            self.closed = True

    plan = AnswerPlan(
        start=time.perf_counter(),
        user_id=uuid.uuid4(),
        request_id="req-boom",
        sources=[{"id": 1, "chunk_id": "c", "document_id": "d", "score": 1.0}],
        messages=[{"role": "user", "content": "q"}],
    )
    tokens = _BoomTokens()
    events = [
        e
        async for e in stream_answer(
            plan, tokens, provider=_OkProvider(), ttft_ms=1, failover=False
        )
    ]
    names = [e["event"] for e in events]

    assert "error" in names
    assert "done" not in names
    assert tokens.closed


class _EmptyThenOk:
    """Yields no tokens at all — an empty completion."""

    def __init__(self) -> None:
        self.streamed = False

    async def stream(self, messages, **kwargs):
        self.streamed = True
        return
        yield  # pragma: no cover


async def test_open_stream_fails_over_on_empty_completion():
    empty = _EmptyThenOk()
    fallback = _OkProvider()
    tokens, provider = await open_stream([empty, fallback], [])
    assert [t async for t in tokens] == ["hello ", "world"]
    assert provider is fallback
    assert empty.streamed
