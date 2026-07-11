# Slice 4 — Hybrid retrieval (semantic + BM25, RRF)

Prereq migration:
- CREATE EXTENSION IF NOT EXISTS pg_search;  (ParadeDB image already has it)
- Add a bm25 index on chunks: 
  CREATE INDEX chunks_bm25 ON chunks USING bm25 (id, content, user_id)
  WITH (key_field='id');
- Hand-write this migration; autogenerate won't produce it.

Retriever interface (behind one class, ts_rank fallback kept internal):
- semantic_search(query, user_id, k): pgvector cosine top-k
  WHERE user_id = current_user.
- bm25_search(query, user_id, k): pg_search @@@ on content, top-k
  WHERE user_id = current_user, ranked by paradedb.score(id).
- hybrid_search(query, user_id, k): run both arms (k=20 each), fuse with RRF
  (k_rrf=60), return top-k fused chunks.

Isolation: BOTH arms filter user_id. This is non-negotiable — a missing filter
in one arm is a leak.

Wire hybrid_search into /query in place of the vector-only call from slice 2.

Tests:
- bm25 arm returns a keyword-exact chunk that vector search alone ranks low.
- vector arm returns a semantically-close chunk with no keyword overlap.
- RRF fusion: a chunk ranked high by BOTH arms beats one ranked high by one.
- ISOLATION: neither arm returns another user's chunks (test both arms).

What can go wrong: pg_search index missing (falls back to ts_rank), RRF math
off (verify with a tiny hand-checked example), score scales mismatched (RRF
uses ranks, so this shouldn't happen — assert it doesn't).