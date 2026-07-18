"""Non-streaming query: embed → hybrid retrieve (scoped to user) → prompt → answer.

Hybrid retrieval runs two arms and fuses them with Reciprocal Rank Fusion:
a pgvector cosine arm and a ParadeDB `pg_search` BM25 keyword arm. Both arms
filter `WHERE user_id = :current_user` — that filter, in *both* arms, is the
real multi-tenant boundary, enforced in the query and not just the API dep.
Fusing by rank (RRF) means the two arms' different score scales never need
normalization.
"""

import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import structlog
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.config import get_settings
from minddrill.models.chunk import Chunk
from minddrill.providers.base import LLMProvider
from minddrill.providers.failover import ProvidersUnavailable, open_stream
from minddrill.rag import events
from minddrill.rag.embedder import Embedder
from minddrill.rag.reranker import Reranker

log = structlog.get_logger(__name__)

_CANDIDATE_K = 20  # fused candidates handed to the reranker
_ARM_K = 20  # candidates pulled from each arm before fusion
_RRF_K = 60  # RRF damping constant
_NO_CONTEXT_REASON = "your documents do not contain anything relevant to that question"
_WEAK_CONTEXT_REASON = "the retrieved context does not support the question"
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


@dataclass
class AnswerPlan:
    """The pre-flight decision for a query, made before the SSE stream opens.

    Either a decline (retrieval empty or below the grounding floor) or a ready-
    to-generate plan carrying the `sources` payload and the assembled prompt.
    """

    start: float
    user_id: uuid.UUID
    decline_reason: str | None = None
    sources: list[dict] | None = None
    messages: list[dict] | None = None


async def prepare_answer(
    session: AsyncSession,
    user_id: uuid.UUID,
    question: str,
    embedder: Embedder,
    reranker: Reranker,
) -> AnswerPlan:
    """Embed → retrieve (scoped) → rerank → grounding gate. No provider call yet."""
    start = time.perf_counter()
    query_vec = await embedder.embed_query(question)
    candidates = await Retriever(session).hybrid_search(
        question, query_vec, user_id, _CANDIDATE_K
    )
    if not candidates:
        log.info("query.no_context", user_id=str(user_id))
        return AnswerPlan(
            start=start, user_id=user_id, decline_reason=_NO_CONTEXT_REASON
        )

    # The reranker only reorders chunks already scoped to this user by both
    # retrieval arms, so it opens no new isolation surface.
    settings = get_settings()
    scored = await reranker.rerank(question, candidates, settings.rerank_top_n)

    # Grounding gate: only meaningful with a real relevance score, so it is
    # skipped when reranking is disabled (the passthrough reports 0.0).
    if settings.rerank_enabled and (
        not scored or scored[0][1] < settings.grounding_min_score
    ):
        top = scored[0][1] if scored else None
        log.info("query.declined", user_id=str(user_id), top_score=top)
        return AnswerPlan(
            start=start, user_id=user_id, decline_reason=_WEAK_CONTEXT_REASON
        )

    chunks = [c for c, _ in scored]
    sources = [
        {
            "id": i,
            "chunk_id": str(c.id),
            "document_id": str(c.document_id),
            "score": score,
        }
        for i, (c, score) in enumerate(scored, start=1)
    ]
    return AnswerPlan(
        start=start,
        user_id=user_id,
        sources=sources,
        messages=_build_messages(question, chunks),
    )


async def stream_answer(
    plan: AnswerPlan,
    tokens: AsyncIterator[str],
    provider: LLMProvider,
    ttft_ms: int,
) -> AsyncIterator[dict]:
    """Emit the SSE events for a committed answer: sources → tokens → done.

    `tokens` already has its first token pulled (failover resolved), so opening
    this generator never fails over. On client disconnect, sse-starlette cancels
    the task driving this generator, raising GeneratorExit at the suspended
    `yield`; the `finally` then closes the provider stream to end token spend.
    Doing it that way avoids a second consumer of the ASGI `receive` channel,
    which sse-starlette already owns for disconnect detection.
    """
    yield events.sources(plan.sources)
    yield events.status("generating")
    text_parts: list[str] = []
    try:
        async for tok in tokens:
            text_parts.append(tok)
            yield events.token(tok)
    except Exception as exc:  # failure after the stream opened
        log.warning("query.stream_error", user_id=str(plan.user_id), error=str(exc))
        yield events.error("internal_error", "generation failed mid-stream")
        return
    finally:
        # Runs on normal completion, mid-stream error, and cancellation
        # (disconnect) alike — always releases the upstream provider stream.
        await tokens.aclose()

    latency_ms = int((time.perf_counter() - plan.start) * 1000)
    usage = getattr(provider, "last_usage", None) or {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    log.info(
        "query.answered",
        user_id=str(plan.user_id),
        source_count=len(plan.sources),
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
    )
    yield events.done(usage, ttft_ms, latency_ms, grounded=True)


async def decline_stream(reason: str) -> AsyncIterator[dict]:
    yield events.decline(reason)


async def run_query(
    session: AsyncSession,
    user_id: uuid.UUID,
    question: str,
    embedder: Embedder,
    providers: Sequence[LLMProvider],
    reranker: Reranker,
) -> AsyncIterator[dict]:
    """Full query path as an SSE event generator, with failover resolved first.

    Provider failover is resolved *before* this generator yields its first event,
    so an all-providers-down failure surfaces as `ProvidersUnavailable` (→ 503)
    rather than a broken half-open stream.
    """
    plan = await prepare_answer(session, user_id, question, embedder, reranker)
    if plan.decline_reason is not None:
        return decline_stream(plan.decline_reason)

    tokens, provider = await open_stream(providers, plan.messages)
    ttft_ms = int((time.perf_counter() - plan.start) * 1000)
    return stream_answer(plan, tokens, provider, ttft_ms)


__all__ = [
    "AnswerPlan",
    "ProvidersUnavailable",
    "Retriever",
    "prepare_answer",
    "run_query",
    "stream_answer",
]
