import httpx
from sqlalchemy import func, select

from minddrill.db.session import SessionLocal
from minddrill.models.chunk import Chunk
from tests._pdfgen import make_pdf
from tests.conftest import register_user

_PDF_LINES = [
    "MindDrill is a retrieval augmented generation assistant.",
    "It ingests documents, embeds chunks, and answers questions.",
    "Each chunk is stored with the owning user id for isolation.",
    "Hybrid retrieval fuses vector search with keyword search.",
]


def _write_pdf(tmp_path) -> str:
    path = tmp_path / "tiny.pdf"
    path.write_bytes(make_pdf(_PDF_LINES))
    return str(path)


async def test_ingest_stores_chunks_with_user_id(
    client: httpx.AsyncClient, tmp_path
) -> None:
    user_id, headers = await register_user(client, "ingestor")
    uri = _write_pdf(tmp_path)

    resp = await client.post(
        "/api/v1/ingest",
        json={"source_type": "pdf", "source_uri": uri},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunk_count"] > 0
    document_id = body["document_id"]

    async with SessionLocal() as session:
        rows = (
            await session.scalars(
                select(Chunk).where(Chunk.document_id == document_id)
            )
        ).all()
    assert len(rows) == body["chunk_count"]
    assert all(str(c.user_id) == user_id for c in rows)


async def test_ingest_is_idempotent_per_user(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "reingest")
    uri = _write_pdf(tmp_path)
    payload = {"source_type": "pdf", "source_uri": uri}

    first = (await client.post("/api/v1/ingest", json=payload, headers=headers)).json()
    second = (await client.post("/api/v1/ingest", json=payload, headers=headers)).json()

    assert first["document_id"] == second["document_id"]
    assert first["chunk_count"] == second["chunk_count"]

    async with SessionLocal() as session:
        total = await session.scalar(select(func.count()).select_from(Chunk))
    assert total == first["chunk_count"]


async def test_ingest_rejects_non_pdf_source_type(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "badtype")

    resp = await client.post(
        "/api/v1/ingest",
        json={"source_type": "markdown", "source_uri": _write_pdf(tmp_path)},
        headers=headers,
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


async def test_ingest_rejects_pdf_with_no_text(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "emptypdf")
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([]))

    resp = await client.post(
        "/api/v1/ingest",
        json={"source_type": "pdf", "source_uri": str(path)},
        headers=headers,
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"
