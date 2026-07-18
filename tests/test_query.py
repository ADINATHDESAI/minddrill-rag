import httpx

from tests._pdfgen import make_pdf
from tests.conftest import parse_sse, register_user

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


async def test_query_streams_sources_then_tokens_then_done(
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
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    names = [name for name, _ in events]

    # sources appears exactly once, before any token; done is last.
    assert names.count("sources") == 1
    assert names.index("sources") < names.index("token")
    assert names[-1] == "done"

    sources = dict(events)["sources"]["sources"]
    assert sources and sources[0]["id"] == 1
    for src in sources:
        assert src["chunk_id"] and src["document_id"]
        assert "score" in src

    tokens = [data["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "canned answer [1]"
    assert llm.streamed  # the provider was actually streamed
    assert reranker.calls  # re-ranking sat between retrieval and the prompt


async def test_query_done_carries_ttft_and_usage(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "metrics")
    await _ingest(client, headers, tmp_path)

    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )

    done = dict(parse_sse(resp.text))["done"]
    assert done["grounded"] is True
    assert isinstance(done["ttft_ms"], int) and done["ttft_ms"] >= 0
    assert isinstance(done["latency_ms"], int) and done["latency_ms"] >= 0
    assert "input_tokens" in done["usage"] and "output_tokens" in done["usage"]


async def test_query_with_no_documents_declines_without_streaming(
    client: httpx.AsyncClient, llm
) -> None:
    _, headers = await register_user(client, "emptyasker")

    resp = await client.post(
        "/api/v1/query",
        json={"question": "anything at all?"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    names = [name for name, _ in events]

    assert "decline" in names
    assert "token" not in names  # no LLM spend
    assert not llm.streamed
