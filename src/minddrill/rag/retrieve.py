"""Non-streaming query: embed → vector top-5 (scoped to user) → prompt → answer.

The `WHERE user_id = :current_user` filter in `_retrieve` is the real
multi-tenant boundary — enforced in the query, not just the API dependency.
"""

import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.models.chunk import Chunk
from minddrill.providers.gemini import GeminiProvider
from minddrill.rag.embedder import Embedder
from minddrill.rag.schemas import QueryResponse, Source

log = structlog.get_logger(__name__)

_TOP_K = 5
_NO_CONTEXT_ANSWER = (
    "I don't have any relevant context in your documents to answer that."
)
_SYSTEM_PROMPT = (
    "You answer strictly from the numbered sources below. Cite them inline with "
    "markers like [1]. If the sources do not answer the question, say you don't "
    "know. Do not use outside knowledge."
)


async def _retrieve(
    session: AsyncSession, user_id: uuid.UUID, query_vec: list[float]
) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.user_id == user_id)  # scope to the caller before ranking
        .order_by(Chunk.embedding.cosine_distance(query_vec))
        .limit(_TOP_K)
    )
    return list(await session.scalars(stmt))


def _build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    numbered = "\n\n".join(
        f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{numbered}\n\nQuestion: {question}"},
    ]


async def answer_question(
    session: AsyncSession,
    user_id: uuid.UUID,
    question: str,
    embedder: Embedder,
    llm: GeminiProvider,
) -> QueryResponse:
    query_vec = await embedder.embed_query(question)
    chunks = await _retrieve(session, user_id, query_vec)

    if not chunks:
        log.info("query.no_context", user_id=str(user_id))
        return QueryResponse(answer=_NO_CONTEXT_ANSWER, sources=[])

    answer = await llm.generate(_build_messages(question, chunks))
    log.info("query.answered", user_id=str(user_id), source_count=len(chunks))
    return QueryResponse(
        answer=answer,
        sources=[Source(chunk_id=c.id, document_id=c.document_id) for c in chunks],
    )
