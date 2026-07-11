# Slice 5 — Cross-encoder re-ranking (local)

Reranker interface (swappable; Cohere could slot in later):
- rerank(query, chunks, top_n=5) -> reordered top_n chunks with rerank scores.
- Model: a local cross-encoder (e.g. BAAI/bge-reranker-base or
  cross-encoder/ms-marco-MiniLM-L6-v2). Load once at startup, reuse.

Placement in the query path:
- hybrid_search returns top-20 (fused) -> rerank -> top-5 -> prompt.

Async safety (critical):
- The cross-encoder is CPU-bound and blocking. Run it in a threadpool
  (anyio.to_thread.run_sync / run_in_executor). It must NOT block the event
  loop, or every concurrent request stalls.

Config:
- RERANK_MODEL and RERANK_TOP_N in settings. A flag to disable rerank (for the
  eval/benchmark comparison later).

Tests:
- rerank reorders a known query so the most relevant chunk moves to rank 1.
- output length == top_n.
- the call is offloaded to a thread (assert the event loop isn't blocked —
  e.g. a concurrent request still responds during a rerank).

What can go wrong: model load latency at first call (load at startup, not per
request), threadpool starvation under load (fine at our scale), top_n larger
than input (clamp).