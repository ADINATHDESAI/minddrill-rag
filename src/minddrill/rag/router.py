"""RAG endpoints — synchronous ingest and non-streaming query."""

from fastapi import APIRouter, Depends, HTTPException, status
from pypdf.errors import PyPdfError
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.auth.deps import get_current_user
from minddrill.db.session import get_session
from minddrill.models.user import User
from minddrill.providers.gemini import GeminiProvider, get_llm, is_rate_limit_error
from minddrill.rag.embedder import Embedder, get_embedder
from minddrill.rag.ingest import EmptyDocumentError, ingest_pdf
from minddrill.rag.retrieve import answer_question
from minddrill.rag.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> IngestResponse:
    if body.source_type != "pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only source_type 'pdf' is supported",
        )
    try:
        return await ingest_pdf(session, current_user.id, body, embedder)
    except (FileNotFoundError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="could not read source_uri",
        )
    except EmptyDocumentError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="the PDF contains no extractable text",
        )
    except PyPdfError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="could not parse the PDF",
        )
    except Exception as exc:
        if is_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="embedding provider rate limit",
            )
        raise


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
