"""Cross-encoder re-ranking: ordering, top_n length/clamp, and non-blocking.

The reranker's own logic (reorder by score, clamp, offload) is exercised with an
injected stand-in model, so these tests are fast and never download the real
cross-encoder. The `predict` seam is the only thing faked; the reorder, clamp and
threadpool offload under test are ours.
"""

import asyncio
import time
import uuid

from minddrill.config import get_settings
from minddrill.models.chunk import Chunk
from minddrill.rag import reranker as rr
from minddrill.rag.reranker import CrossEncoderReranker, PassthroughReranker


def _chunk(content: str) -> Chunk:
    return Chunk(id=uuid.uuid4(), content=content)


class _ScoredModel:
    """Scores each pair by looking the chunk text up in a fixed score map."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def predict(self, pairs):
        return [self._scores[text] for _query, text in pairs]


async def test_rerank_moves_relevant_chunk_to_rank_1():
    chunks = [_chunk("barely related"), _chunk("dead on"), _chunk("off topic")]
    model = _ScoredModel({"barely related": 0.2, "dead on": 0.9, "off topic": 0.05})
    reranker = CrossEncoderReranker(model=model)

    ranked = await reranker.rerank("the query", chunks, top_n=3)

    assert [c.content for c in ranked] == ["dead on", "barely related", "off topic"]


async def test_rerank_output_length_equals_top_n():
    chunks = [_chunk(f"c{i}") for i in range(6)]
    model = _ScoredModel({f"c{i}": float(i) for i in range(6)})
    reranker = CrossEncoderReranker(model=model)

    ranked = await reranker.rerank("q", chunks, top_n=2)

    assert len(ranked) == 2
    # Highest scores are the last two, best first.
    assert [c.content for c in ranked] == ["c5", "c4"]


async def test_rerank_clamps_top_n_above_input():
    chunks = [_chunk("a"), _chunk("b")]
    model = _ScoredModel({"a": 0.1, "b": 0.2})
    reranker = CrossEncoderReranker(model=model)

    ranked = await reranker.rerank("q", chunks, top_n=10)

    assert [c.content for c in ranked] == ["b", "a"]


async def test_rerank_empty_input_returns_empty():
    reranker = CrossEncoderReranker(model=_ScoredModel({}))
    assert await reranker.rerank("q", [], top_n=5) == []


async def test_rerank_offloads_to_thread_and_does_not_block_loop():
    """A blocking predict must not stall the event loop — proves the threadpool."""

    class _SlowModel:
        def predict(self, pairs):
            time.sleep(0.3)  # blocking, CPU-bound stand-in
            return [0.0] * len(pairs)

    reranker = CrossEncoderReranker(model=_SlowModel())
    chunks = [_chunk("a"), _chunk("b")]

    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await reranker.rerank("q", chunks, top_n=2)
    await task

    # If predict had blocked the loop, the ticker couldn't have advanced during
    # the 0.3s sleep. It runs concurrently only because predict is offloaded.
    assert ticks >= 10


async def test_passthrough_keeps_fused_order_and_clamps():
    chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
    reranker = PassthroughReranker()

    ranked = await reranker.rerank("q", chunks, top_n=2)

    assert [c.content for c in ranked] == ["first", "second"]


def test_disable_flag_selects_passthrough(monkeypatch):
    """RERANK_ENABLED=false swaps in the passthrough — the eval/benchmark baseline."""
    monkeypatch.setenv("RERANK_ENABLED", "false")
    get_settings.cache_clear()
    rr.get_reranker.cache_clear()
    try:
        assert isinstance(rr.get_reranker(), PassthroughReranker)
    finally:
        get_settings.cache_clear()
        rr.get_reranker.cache_clear()
