import httpx

from tests._pdfgen import make_pdf
from tests.conftest import register_user

_A_LINES = [
    "Project Zephyr is a confidential internal initiative.",
    "The launch code for Zephyr is alpha seven seven.",
]


async def test_user_b_query_never_returns_user_a_chunks(
    client: httpx.AsyncClient, llm, tmp_path
) -> None:
    # User A ingests a private document.
    _, a_headers = await register_user(client, "alice_iso")
    path = tmp_path / "secret.pdf"
    path.write_bytes(make_pdf(_A_LINES))
    ingest = await client.post(
        "/api/v1/ingest",
        json={"source_type": "pdf", "source_uri": str(path)},
        headers=a_headers,
    )
    assert ingest.status_code == 200, ingest.text
    assert ingest.json()["chunk_count"] > 0

    # User B asks the exact question A's document answers.
    _, b_headers = await register_user(client, "bob_iso")
    resp = await client.post(
        "/api/v1/query",
        json={"question": "What is the launch code for Project Zephyr?"},
        headers=b_headers,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The retrieval filter WHERE user_id = current_user must exclude A's chunks.
    assert body["sources"] == []
    assert llm.last_messages is None  # nothing retrieved -> LLM never sees A's text
