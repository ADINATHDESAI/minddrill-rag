import httpx

from minddrill.main import app
from minddrill.providers.gemini import get_llm
from tests._pdfgen import make_pdf
from tests.conftest import register_user

_PDF_LINES = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "Bread and cheese are staples of French cuisine.",
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


async def test_query_returns_answer_and_sources(
    client: httpx.AsyncClient, llm, reranker, tmp_path
) -> None:
    _, headers = await register_user(client, "asker")
    await _ingest(client, headers, tmp_path)

    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"]
    assert len(body["sources"]) > 0
    for src in body["sources"]:
        assert src["chunk_id"]
        assert src["document_id"]
    assert llm.last_messages is not None  # the LLM was actually called
    assert reranker.calls  # re-ranking sat between retrieval and the prompt


async def test_query_with_no_documents_declines_without_calling_llm(
    client: httpx.AsyncClient, llm
) -> None:
    _, headers = await register_user(client, "emptyasker")

    resp = await client.post(
        "/api/v1/query",
        json={"question": "anything at all?"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sources"] == []
    assert body["answer"]
    assert llm.last_messages is None  # no context -> no LLM spend


class _RateLimitedLLM:
    async def stream(self, messages, **kwargs):
        raise RuntimeError("429 quota exceeded")
        yield  # pragma: no cover

    async def generate(self, messages, **kwargs) -> str:
        raise RuntimeError("429 quota exceeded")


async def test_query_maps_provider_rate_limit_to_429(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "throttled")
    await _ingest(client, headers, tmp_path)

    # The client fixture removes this override on teardown.
    app.dependency_overrides[get_llm] = lambda: _RateLimitedLLM()
    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )

    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "rate_limited"
