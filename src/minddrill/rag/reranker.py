"""Cross-encoder re-ranking behind an interface.

`Reranker` is the seam; `CrossEncoderReranker` is the local production
implementation (Sentence Transformers cross-encoder). It re-scores every
query–chunk pair and keeps the best `top_n`, tightening the coarse rank-based
RRF ordering from hybrid retrieval.

The model is CPU-bound and blocking, so `predict` runs in a threadpool and never
touches the event loop. `get_reranker` is the FastAPI dependency (loaded once);
when re-ranking is disabled it returns a passthrough that keeps the fused order.
"""

import asyncio
from functools import lru_cache
from typing import Protocol, runtime_checkable

import structlog

from minddrill.config import get_settings
from minddrill.models.chunk import Chunk

log = structlog.get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int
    ) -> list[tuple[Chunk, float]]:
        """Re-score query–chunk pairs; return the top_n (chunk, score), best first.

        The score is the relevance signal the grounding gate and the `sources`
        event both read.
        """
        ...


def _clamp(top_n: int, n: int) -> int:
    return max(0, min(top_n, n))


class CrossEncoderReranker:
    def __init__(self, model: object | None = None) -> None:
        # Import lazily so callers that inject a model (and every test) don't pay
        # the torch import cost.
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(get_settings().rerank_model)
        self._model = model

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int
    ) -> list[tuple[Chunk, float]]:
        top_n = _clamp(top_n, len(chunks))
        if top_n == 0:
            return []
        pairs = [[query, c.content] for c in chunks]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(
            zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        top = [(chunk, float(score)) for chunk, score in ranked[:top_n]]
        log.info(
            "rerank.done",
            candidates=len(chunks),
            top_n=top_n,
            top_ids=[str(c.id) for c, _ in top],
        )
        return top


class PassthroughReranker:
    """No-op reranker: keeps the fused order, clamped to top_n.

    Used when re-ranking is disabled (the eval/benchmark baseline).
    """

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int
    ) -> list[tuple[Chunk, float]]:
        # No cross-encoder score to report; the grounding gate is skipped when
        # reranking is disabled, so the 0.0 placeholder is never used as a signal.
        return [(c, 0.0) for c in chunks[: _clamp(top_n, len(chunks))]]


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    if not settings.rerank_enabled:
        return PassthroughReranker()
    return CrossEncoderReranker()


def warm_reranker() -> None:
    """Construct the singleton so the model loads at startup, not first request."""
    get_reranker()
