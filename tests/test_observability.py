"""Langfuse tracing: one trace per request, keyed to the logging request_id.

The Langfuse client is swapped for `FakeLangfuse` (see conftest) so these
assert on what production code reported to the tracing seam, never on a real
network call.
"""

import httpx

from minddrill.observability import trace_id_for
from tests._pdfgen import make_pdf
from tests.conftest import FakeLangfuse, parse_sse, register_user


async def _ingest(client: httpx.AsyncClient, headers: dict, tmp_path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf(["The Eiffel Tower is located in Paris."]))
    resp = await client.post(
        "/api/v1/ingest",
        json={"source_type": "pdf", "source_uri": str(path)},
        headers=headers,
    )
    assert resp.status_code == 202, resp.text


async def test_query_produces_one_trace_with_retrieve_rerank_generate_spans(
    client: httpx.AsyncClient, langfuse_fake: FakeLangfuse, tmp_path
) -> None:
    _, headers = await register_user(client, "trace_asker")
    await _ingest(client, headers, tmp_path)

    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    names = [s.name for s in langfuse_fake.spans]
    assert names == ["retrieve", "rerank", "generate"]
    # One trace: every span shares the same (derived) trace id.
    trace_ids = {s.trace_id for s in langfuse_fake.spans}
    assert len(trace_ids) == 1


async def test_query_trace_id_equals_derived_request_id(
    client: httpx.AsyncClient, langfuse_fake: FakeLangfuse, tmp_path
) -> None:
    _, headers = await register_user(client, "trace_id_asker")
    await _ingest(client, headers, tmp_path)

    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    request_id = resp.headers["x-request-id"]
    expected_trace_id = trace_id_for(request_id)
    assert langfuse_fake.spans  # sanity: spans were actually opened
    for span in langfuse_fake.spans:
        assert span.trace_id == expected_trace_id


async def test_query_usage_and_ttft_attach_to_generate_span_on_completion(
    client: httpx.AsyncClient, langfuse_fake: FakeLangfuse, tmp_path
) -> None:
    _, headers = await register_user(client, "trace_usage_asker")
    await _ingest(client, headers, tmp_path)

    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    generate = next(s for s in langfuse_fake.spans if s.name == "generate")
    assert generate.ended  # closed only once the stream actually completed
    assert generate.usage_details == {"input_tokens": 3, "output_tokens": 3}
    assert generate.metadata["ttft_ms"] >= 0
    assert generate.metadata["latency_ms"] >= 0
    assert generate.output == "canned answer [1]"

    # The done event carries the same numbers the span recorded.
    done = dict(parse_sse(resp.text))["done"]
    assert done["usage"] == generate.usage_details
    assert done["ttft_ms"] == generate.metadata["ttft_ms"]


async def test_query_decline_produces_no_generate_span(
    client: httpx.AsyncClient, langfuse_fake: FakeLangfuse
) -> None:
    _, headers = await register_user(client, "trace_decline_asker")

    resp = await client.post(
        "/api/v1/query",
        json={"question": "anything at all?"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    names = [s.name for s in langfuse_fake.spans]
    assert "generate" not in names  # no LLM spend, so no generation to trace


async def test_query_traces_carry_no_secrets(
    client: httpx.AsyncClient, langfuse_fake: FakeLangfuse, tmp_path
) -> None:
    _, headers = await register_user(client, "trace_privacy_asker")
    await _ingest(client, headers, tmp_path)

    resp = await client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    token = headers["Authorization"].split(" ", 1)[1]
    for span in langfuse_fake.spans:
        blob = repr((span.input, span.output, span.metadata))
        assert token not in blob
        assert "Authorization" not in blob


async def test_chat_opens_one_span_on_the_derived_trace(
    client: httpx.AsyncClient, langfuse_fake: FakeLangfuse
) -> None:
    _, headers = await register_user(client, "trace_chatter")
    resp = await client.post("/api/v1/sessions", json={"title": "t"}, headers=headers)
    session_id = resp.json()["session_id"]

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "hello", "session_id": session_id},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    request_id = resp.headers["x-request-id"]
    names = [s.name for s in langfuse_fake.spans]
    assert names == ["chat"]
    assert langfuse_fake.spans[0].trace_id == trace_id_for(request_id)
    assert langfuse_fake.spans[0].ended
