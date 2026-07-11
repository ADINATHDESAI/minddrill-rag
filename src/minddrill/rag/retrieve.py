"""Non-streaming query: embed → hybrid retrieve (scoped to user) → prompt → answer.

Hybrid retrieval runs two arms and fuses them with Reciprocal Rank Fusion:
a pgvector cosine arm and a ParadeDB `pg_search` BM25 keyword arm. Both arms
filter `WHERE user_id = :current_user` — that filter, in *both* arms, is the
real multi-tenant boundary, enforced in the query and not just the API dep.
Fusing by rank (RRF) means the two arms' different score scales never need
normalization.
"""

import uuid

import structlog
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.models.chunk import Chunk
from minddrill.providers.gemini import GeminiProvider
from minddrill.rag.embedder import Embedder
from minddrill.rag.schemas import QueryResponse, Source

log = structlog.get_logger(__name__)

_TOP_K = 5  # chunks handed to the LLM after fusion
_ARM_K = 20  # candidates pulled from each arm before fusion
_RRF_K = 60  # RRF damping constant
_NO_CONTEXT_ANSWER = (
    "I don't have any relevant context in your documents to answer that."
)
_SYSTEM_PROMPT = (
    "You answer strictly from the numbered sources below. Cite them inline with "
    "markers like [1]. If the sources do not answer the question, say you don't "
    "know. Do not use outside knowledge."
)


def _rrf(rankings: list[list[uuid.UUID]], k: int = _RRF_K) -> list[uuid.UUID]:
    """Reciprocal Rank Fusion: score(id) = sum 1/(k + rank), rank 1-based.

    Fuses by rank, so the arms' incomparable score scales never interact.
    """
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    """The two retrieval arms and their RRF fusion, scoped to one user.

    The `ts_rank` fallback for hosts without `pg_search` is an internal detail
    of the BM25 arm.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def semantic_search(
        self, query_vec: list[float], user_id: uuid.UUID, k: int
    ) -> list[uuid.UUID]:
        stmt = (
            select(Chunk.id)
            .where(Chunk.user_id == user_id)  # scope to the caller before ranking
            .order_by(Chunk.embedding.cosine_distance(query_vec))
            .limit(k)
        )
        return list(await self._session.scalars(stmt))

    async def bm25_search(
        self, query: str, user_id: uuid.UUID, k: int
    ) -> list[uuid.UUID]:
        params = {"user_id": user_id, "query": query, "k": k}
        # The `user_id` predicate guarantees isolation even if ParadeDB
        # post-filters rather than pushing the filter into the BM25 index.
        pg_search = text(
            "SELECT id FROM chunks "
            "WHERE user_id = :user_id AND content @@@ :query "
            "ORDER BY paradedb.score(id) DESC LIMIT :k"
        )
        try:
            async with (
                self._session.begin_nested()
            ):  # savepoint: keep session usable on fallback
                rows = await self._session.execute(pg_search, params)
                return [row[0] for row in rows]
        except DBAPIError:
            log.warning("bm25.pg_search_unavailable", user_id=str(user_id))
            return await self._ts_rank_fallback(params)

    async def _ts_rank_fallback(self, params: dict) -> list[uuid.UUID]:
        stmt = text(
            "SELECT id FROM chunks "
            "WHERE user_id = :user_id "
            "AND to_tsvector('english', content) @@ plainto_tsquery('english', :query) "
            "ORDER BY ts_rank("
            "to_tsvector('english', content), plainto_tsquery('english', :query)"
            ") DESC LIMIT :k"
        )
        rows = await self._session.execute(stmt, params)
        return [row[0] for row in rows]

    async def hybrid_search(
        self, query: str, query_vec: list[float], user_id: uuid.UUID, k: int
    ) -> list[Chunk]:
        # One asyncpg connection can't run two queries at once, so the arms run
        # sequentially; true parallelism would need separate connections.
        semantic = await self.semantic_search(query_vec, user_id, _ARM_K)
        keyword = await self.bm25_search(query, user_id, _ARM_K)

        fused_ids = _rrf([semantic, keyword])[:k]
        if not fused_ids:
            return []

        by_id = {
            c.id: c
            for c in await self._session.scalars(
                select(Chunk).where(Chunk.id.in_(fused_ids))
            )
        }
        return [by_id[cid] for cid in fused_ids]


def _build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    numbered = "\n\n".join(f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1))
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
    chunks = await Retriever(session).hybrid_search(
        question, query_vec, user_id, _TOP_K
    )

    if not chunks:
        log.info("query.no_context", user_id=str(user_id))
        return QueryResponse(answer=_NO_CONTEXT_ANSWER, sources=[])

    answer = await llm.generate(_build_messages(question, chunks))
    log.info("query.answered", user_id=str(user_id), source_count=len(chunks))
    return QueryResponse(
        answer=answer,
        sources=[Source(chunk_id=c.id, document_id=c.document_id) for c in chunks],
    )
