from unittest.mock import MagicMock
from uuid import UUID

import httpx
from sqlalchemy import func, select

from minddrill.db.session import SessionLocal
from minddrill.models.chunk import Chunk
from minddrill.models.ingestion_job import IngestionJob
from minddrill.worker import tasks as worker_tasks
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


async def test_ingest_returns_202_and_creates_job(
    client: httpx.AsyncClient, tmp_path
) -> None:
    user_id, headers = await register_user(client, "ingestor")
    uri = _write_pdf(tmp_path)

    resp = await client.post(
        "/api/v1/ingest",
        json={"source_type": "pdf", "source_uri": uri},
        headers=headers,
    )

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]

    async with SessionLocal() as session:
        job = await session.get(IngestionJob, job_id)
    assert job is not None
    assert str(job.user_id) == user_id


async def test_task_success_stores_chunks_and_marks_done(
    client: httpx.AsyncClient, tmp_path
) -> None:
    user_id, headers = await register_user(client, "successful")
    uri = _write_pdf(tmp_path)

    job_id = (
        await client.post(
            "/api/v1/ingest",
            json={"source_type": "pdf", "source_uri": uri},
            headers=headers,
        )
    ).json()["job_id"]

    status = await client.get(f"/api/v1/ingest/{job_id}", headers=headers)
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["status"] == "done"
    document_id = body["document_id"]
    assert document_id is not None
    assert body["error"] is None

    async with SessionLocal() as session:
        rows = (
            await session.scalars(
                select(Chunk).where(Chunk.document_id == document_id)
            )
        ).all()
    assert len(rows) > 0
    assert all(str(c.user_id) == user_id for c in rows)


async def test_reingest_same_content_is_idempotent(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "reingest")
    uri = _write_pdf(tmp_path)
    payload = {"source_type": "pdf", "source_uri": uri}

    first_job = (await client.post("/api/v1/ingest", json=payload, headers=headers)).json()
    second_job = (await client.post("/api/v1/ingest", json=payload, headers=headers)).json()

    first = (await client.get(f"/api/v1/ingest/{first_job['job_id']}", headers=headers)).json()
    second = (await client.get(f"/api/v1/ingest/{second_job['job_id']}", headers=headers)).json()

    assert first["status"] == "done"
    assert second["status"] == "done"
    assert first["document_id"] == second["document_id"]

    async with SessionLocal() as session:
        total = await session.scalar(select(func.count()).select_from(Chunk))
        chunks_for_doc = await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == first["document_id"])
        )
    assert total == chunks_for_doc  # no duplicate chunks from the second ingest


async def test_failure_marks_job_failed_with_error(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "poison")

    job_id = (
        await client.post(
            "/api/v1/ingest",
            json={"source_type": "pdf", "source_uri": str(tmp_path / "missing.pdf")},
            headers=headers,
        )
    ).json()["job_id"]

    body = (await client.get(f"/api/v1/ingest/{job_id}", headers=headers)).json()
    assert body["status"] == "failed"
    assert body["error"]
    assert body["document_id"] is None


async def test_empty_pdf_marks_job_failed(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, headers = await register_user(client, "blankpdf")
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([]))

    job_id = (
        await client.post(
            "/api/v1/ingest",
            json={"source_type": "pdf", "source_uri": str(path)},
            headers=headers,
        )
    ).json()["job_id"]

    body = (await client.get(f"/api/v1/ingest/{job_id}", headers=headers)).json()
    assert body["status"] == "failed"
    assert body["error"]


class _RateLimitedEmbedder:
    """Always rate-limits — a transient failure that should trigger a retry."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("429 quota exceeded")

    async def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("429 quota exceeded")


async def test_rate_limit_requests_retry_with_backoff(
    client: httpx.AsyncClient, tmp_path, monkeypatch
) -> None:
    # Eager mode re-runs retries inline (with real backoff sleeps), so stub
    # self.retry to assert the *decision*: a 429 schedules a retry with
    # exponential backoff and bumps retry_count, rather than failing the job.
    user_id, _ = await register_user(client, "throttled")
    uri = _write_pdf(tmp_path)

    async with SessionLocal() as session:
        job = IngestionJob(
            user_id=UUID(user_id), source_type="pdf", source_uri=uri, status="pending"
        )
        session.add(job)
        await session.commit()
        job_id = job.id

    monkeypatch.setattr(worker_tasks, "get_embedder", lambda: _RateLimitedEmbedder())
    retry = MagicMock(side_effect=RuntimeError("retry requested"))
    monkeypatch.setattr(worker_tasks.ingest_document, "retry", retry)

    worker_tasks.ingest_document.apply(args=[str(job_id), user_id, uri], throw=False)

    assert retry.called
    assert retry.call_args.kwargs["countdown"] == 1  # 2 ** 0 on the first attempt

    async with SessionLocal() as session:
        job = await session.get(IngestionJob, job_id)
    assert job.status == "processing"  # a poison doc would be "failed" instead
    assert job.retry_count == 1


async def test_get_another_users_job_returns_404(
    client: httpx.AsyncClient, tmp_path
) -> None:
    _, a_headers = await register_user(client, "owner")
    uri = _write_pdf(tmp_path)
    job_id = (
        await client.post(
            "/api/v1/ingest",
            json={"source_type": "pdf", "source_uri": uri},
            headers=a_headers,
        )
    ).json()["job_id"]

    _, b_headers = await register_user(client, "intruder")
    resp = await client.get(f"/api/v1/ingest/{job_id}", headers=b_headers)

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


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
