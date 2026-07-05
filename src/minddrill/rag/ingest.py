"""Synchronous PDF ingestion: load → split → embed → store.

`source_uri` is a local filesystem path. Idempotency is per-user on
`(user_id, content_hash)`.
"""

import asyncio
import hashlib
import uuid

import structlog
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.models.chunk import Chunk
from minddrill.models.document import Document
from minddrill.rag.embedder import Embedder
from minddrill.rag.schemas import IngestRequest, IngestResponse

log = structlog.get_logger(__name__)


class EmptyDocumentError(ValueError):
    """The source produced no extractable text to chunk."""

# Character-based sizing approximating ~600-token chunks with ~100-token overlap
# (~4 chars/token). Keeps ingestion dependency-light this slice.
_CHARS_PER_TOKEN = 4
_CHUNK_CHARS = 600 * _CHARS_PER_TOKEN
_OVERLAP_CHARS = 100 * _CHARS_PER_TOKEN


def _hash_and_split(source_uri: str) -> tuple[str, list]:
    with open(source_uri, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()
    pages = PyPDFLoader(source_uri).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_CHARS, chunk_overlap=_OVERLAP_CHARS
    )
    return content_hash, splitter.split_documents(pages)


async def _existing_document(
    session: AsyncSession, user_id: uuid.UUID, content_hash: str
) -> IngestResponse | None:
    existing = await session.scalar(
        select(Document).where(
            Document.user_id == user_id, Document.content_hash == content_hash
        )
    )
    if existing is None:
        return None
    chunk_count = await session.scalar(
        select(func.count()).select_from(Chunk).where(
            Chunk.document_id == existing.id
        )
    )
    return IngestResponse(document_id=existing.id, chunk_count=chunk_count)


async def ingest_pdf(
    session: AsyncSession,
    user_id: uuid.UUID,
    req: IngestRequest,
    embedder: Embedder,
) -> IngestResponse:
    content_hash, splits = await asyncio.to_thread(_hash_and_split, req.source_uri)

    duplicate = await _existing_document(session, user_id, content_hash)
    if duplicate is not None:
        log.info("ingest.skip_duplicate", document_id=str(duplicate.document_id))
        return duplicate

    if not splits:
        raise EmptyDocumentError("the PDF contains no extractable text")

    embeddings = await embedder.embed_texts([s.page_content for s in splits])

    document = Document(
        user_id=user_id,
        source_type=req.source_type,
        source_uri=req.source_uri,
        content_hash=content_hash,
        status="done",
        metadata_=req.metadata,
    )
    session.add(document)
    await session.flush()

    for index, (split, embedding) in enumerate(zip(splits, embeddings)):
        session.add(
            Chunk(
                document_id=document.id,
                user_id=user_id,
                chunk_index=index,
                content=split.page_content,
                token_count=max(1, len(split.page_content) // _CHARS_PER_TOKEN),
                embedding=embedding,
                metadata_=split.metadata,
            )
        )

    try:
        await session.commit()
    except IntegrityError:
        # A concurrent ingest of the same content won the unique (user_id,
        # content_hash) race; return the row it committed.
        await session.rollback()
        duplicate = await _existing_document(session, user_id, content_hash)
        if duplicate is None:
            raise
        return duplicate

    log.info("ingest.done", document_id=str(document.id), chunk_count=len(splits))
    return IngestResponse(document_id=document.id, chunk_count=len(splits))
