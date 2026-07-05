# MindDrill — Decision Log

One line of intent per decision: **what** we chose, **why**, and the **cost** we accepted.
Update this when a decision changes. This file is the portable memory — any AI tool reads it and builds the same system.

---

## Data & storage

**Single Postgres via the ParadeDB image** (pgvector + `pg_search` + JSONB). No separate vector DB.
Why: one system to run, back up, reason about; chunk + vector commit together. Cost: pgvector's index weakens past millions of vectors — never bites at our scale.

**True BM25 via ParadeDB `pg_search`** (`USING bm25`, `@@@`), not `ts_rank`.
Why: real Okapi BM25 (TF saturation, IDF, length norm) ranks far better; strong resume/ATS signal. Cost: ties us to the ParadeDB image / self-host; most free managed Postgres can't run it. `ts_rank` kept as a fallback behind the keyword interface.

**JSONB for loose fields only** (metadata, tool args, trace extras).
Why: flexibility without schema churn. Cost: core fields we filter/join on stay real columns, not JSONB.

## Retrieval

**Hybrid search fused with Reciprocal Rank Fusion (k=60).**
Why: semantic and keyword scores live on different scales; RRF fuses by rank, no fragile normalization. Cost: none meaningful.

**Local cross-encoder rerank (Sentence Transformers), not Cohere.**
Why: free, no rate limit, no network hop. Cost: CPU-bound — must run in a threadpool so it never blocks the async loop. Kept behind a `Reranker` interface so Cohere can swap in later.

## Ingestion vs query (the core split)

**Celery + Redis for ingestion only.**
Why: ingestion is long, failure-prone, restart-sensitive — it needs durable jobs, retries, status polling, idempotency, dead-letter. Cost: a broker to run; justified by real need.

**Query path is synchronous and hand-written. Never queued.**
Why: queries are short and interactive; queuing adds latency for zero benefit. This is the portfolio centrepiece — kept explicit, not hidden behind a framework.

## Auth & isolation

**Username/password → JWT.** Passwords hashed (bcrypt/argon2). Access token only in v1.
Why: real per-user isolation without OAuth's weight. Cost: refresh tokens deferred (easy later add).

**Isolation enforced at the query layer**, not just the API. Every read filters `user_id = current_user`, including *both* retrieval arms.
Why: API auth alone doesn't stop cross-user chunk retrieval; the query filter is the real boundary. `user_id` denormalized onto `chunks` so retrieval filters without a join.

**Idempotency is per-user:** unique `(user_id, content_hash)`.
Why: two users uploading the same file must get separate, isolated copies — no shared chunks.

## Inference

**Streaming-first provider interface:** the async stream is the primitive; the string version wraps it.
Why: you can collapse a stream to a string cheaply, but can't split a string back into a stream. Adding streaming later would touch every caller.

**Gemini free tier (primary) → OpenRouter (fallback). Failover before the SSE stream opens.**
Why: streaming works on free tiers; the real limit is rate (429), which arrives before any token — so we fail over cleanly, client never sees a broken stream.

**Gemini models: `gemini-embedding-001` at 768 dims (output_dimensionality) for embeddings, `gemini-2.5-flash` for chat.**
Why: the originally specced `text-embedding-004` is no longer served on current keys, and `gemini-2.0-flash` has zero free-tier generate quota. Cost: pinning `gemini-embedding-001` to 768 keeps the fixed `vector(768)` schema unchanged.

**LangChain chat models as per-provider adapters inside our own interface.**
Why: free multi-provider + typed content blocks; our failover/stream orchestration stays ours.

**Pydantic validation only where a machine consumes output** (tool inputs, DB writes, agent steps) — not streamed chat text.
Why: you can't validate half an object or un-send streamed tokens. Most "structure" (sources, cost, latency) we compute ourselves.

## Streaming transport

**SSE over HTTP with typed events** (`status`, `sources`, `token`, `tool_call`, `tool_result`, `decline`, `done`, `error`).
Why: token streaming is one-directional; SSE is simple, proxy-friendly, curl-testable. Transport sits behind the event protocol so a WebSocket swap later touches nothing else. Citations are inline markers referencing the `sources` event.

## Agent harness

**LangGraph `create_agent` (post-1.0) for the tool loop only.** RAG hot path stays hand-written.
Why: a tool loop is genuine branching — LangGraph's fit. A linear RAG query is not. Middleware gives the guardrails (max tool steps, history trimming) for free.

**LangChain for loaders/splitters + model adapters only** — not retrieval orchestration.
Why: use it for the boring parts; own the interesting parts a reviewer wants to see.

**Two modes:** direct doc question → always retrieve (sync path); open chat → agent loop where retrieval is one tool.
Why: keeps RAG deterministic for direct questions; tools available when needed.

**Short-term memory only in v1** (recent turns trimmed to a token budget; summarize when long). No long-term cross-session memory. Kept separate from RAG.
Why: avoid over-engineering; add long-term only on real need.

## Local models

**Local SLMs benchmarked offline only — not served in production.** ≤3GB RAM, 2–3B dense at Q4_K_M (e.g. Llama 3.2 3B, Qwen 2.5 3B), same size band.
Why: a cheap server can't serve a 7B model; the benchmark report is the portfolio artifact. Fair comparison = same hardware, quant, and size band. Serving happens via free cloud.

## Observability & eval

**Self-hosted Langfuse.** Tee the stream: accumulate full text while yielding, log response + usage in one span. Capture TTFT.
Why: free, self-hosting is an ops signal; streaming unlocks time-to-first-token as a real metric.

**Ragas + golden set (50–200 Q&A) as an offline CI gate** in GitHub Actions. Faithfulness + context precision. Server-independent.
Why: a build that fails on a quality regression is a strong maturity signal.

## Ops

**Docker Compose:** API, worker, ParadeDB Postgres, Redis, Langfuse. Cheap self-host.
Why: one command brings up the stack; real deployment story.

**Repo is the memory.** CLAUDE.md, `specs/`, tests, and this log carry all context — tool-agnostic.
Why: switching AI coding tools costs nothing when context lives in the repo, not a tool's chat history.