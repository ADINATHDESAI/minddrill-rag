"""RAG endpoints — async ingest (queued) and non-streaming query.

Ingestion is enqueued to a Celery worker; the query path stays synchronous and
hand-written and is never queued.
"""

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.auth.deps import get_current_user
from minddrill.db.session import get_session
from minddrill.models.ingestion_job import IngestionJob
from minddrill.models.user import User
from minddrill.providers.gemini import GeminiProvider, get_llm, is_rate_limit_error
from minddrill.rag.embedder import Embedder, get_embedder
from minddrill.rag.retrieve import answer_question
from minddrill.rag.schemas import (
    IngestJobResponse,
    IngestJobStatus,
    IngestRequest,
    QueryRequest,
    QueryResponse,
)
from minddrill.worker.tasks import ingest_document

router = APIRouter(tags=["rag"])


@router.post(
    "/ingest",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest(
    body: IngestRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> IngestJobResponse:
    if body.source_type != "pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only source_type 'pdf' is supported",
        )

    job = IngestionJob(
        user_id=current_user.id,
        source_type=body.source_type,
        source_uri=body.source_uri,
        status="pending",
    )
    session.add(job)
    await session.commit()

    # .delay() publishes to the broker synchronously (blocking socket I/O), so
    # keep it off the event loop.
    await asyncio.to_thread(
        ingest_document.delay, str(job.id), str(current_user.id), body.source_uri
    )
    return IngestJobResponse(job_id=job.id, status="pending")


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
async def ingest_status(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> IngestJobStatus:
    job = await session.scalar(
        select(IngestionJob).where(
            IngestionJob.id == job_id, IngestionJob.user_id == current_user.id
        )
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
        )
    return IngestJobStatus(
        job_id=job.id,
        status=job.status,
        document_id=job.document_id,
        error=job.error,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    llm: GeminiProvider = Depends(get_llm),
) -> QueryResponse:
    try:
        return await answer_question(
            session, current_user.id, body.question, embedder, llm
        )
    except Exception as exc:
        if is_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="inference provider rate limit",
            )
        raise
