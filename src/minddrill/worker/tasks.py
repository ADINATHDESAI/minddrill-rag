"""The ingestion task: load → split → embed → store, with durable status.

Wraps the async `ingest_pdf` core with `ingestion_jobs` state transitions. A
transient provider 429 retries with exponential backoff; any other failure is
terminal (a poison document must not loop forever).
"""

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from minddrill.config import get_settings
from minddrill.models.ingestion_job import IngestionJob
from minddrill.providers.gemini import is_rate_limit_error
from minddrill.rag.embedder import get_embedder
from minddrill.rag.ingest import ingest_pdf
from minddrill.rag.schemas import IngestRequest
from minddrill.worker.celery_app import celery_app

log = structlog.get_logger(__name__)


def _run_sync(coro):
    """Drive a coroutine to completion in a fresh thread with its own loop.

    Celery tasks are synchronous. A dedicated thread lets us use the async
    ingest core in the worker and under ``task_always_eager`` in tests (where an
    outer event loop is already running and a bare ``asyncio.run`` would raise).
    """
    box: dict = {}

    def _target() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised on the calling thread below
            box["error"] = exc

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


@asynccontextmanager
async def _worker_sessionmaker():
    """A task-scoped async engine (NullPool).

    asyncpg connections are bound to the loop that opened them, so the worker
    never reuses the query path's pooled engine across its per-task loops.
    """
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _set_status(job_id: uuid.UUID, status: str, **fields) -> None:
    async with _worker_sessionmaker() as Session, Session() as session:
        job = await session.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = status
        for key, value in fields.items():
            setattr(job, key, value)
        await session.commit()


async def _process(job_id: uuid.UUID, source_uri: str) -> None:
    async with _worker_sessionmaker() as Session:
        async with Session() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                return
            # Trust the persisted owner, not a task arg, for the write.
            owner_id = job.user_id
            source_type = job.source_type
            job.status = "processing"
            await session.commit()

        async with Session() as session:
            req = IngestRequest(source_type=source_type, source_uri=source_uri)
            result = await ingest_pdf(session, owner_id, req, get_embedder())

        async with Session() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                return
            job.status = "done"
            job.document_id = result.document_id
            await session.commit()


@celery_app.task(bind=True, max_retries=3, name="ingest_document")
def ingest_document(self, job_id: str, user_id: str, source_uri: str) -> None:
    job_uuid = uuid.UUID(job_id)
    log.info("ingest.processing", job_id=job_id)
    try:
        _run_sync(_process(job_uuid, source_uri))
    except Exception as exc:
        if is_rate_limit_error(exc) and self.request.retries < self.max_retries:
            attempt = self.request.retries + 1
            _run_sync(_set_status(job_uuid, "processing", retry_count=attempt))
            log.warning("ingest.retry", job_id=job_id, attempt=attempt)
            raise self.retry(exc=exc, countdown=2**self.request.retries)
        _run_sync(_set_status(job_uuid, "failed", error=str(exc)))
        log.error("ingest.failed", job_id=job_id, error=str(exc))
        return
    log.info("ingest.done", job_id=job_id)
