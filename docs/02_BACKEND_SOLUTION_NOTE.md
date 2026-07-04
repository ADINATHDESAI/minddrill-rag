# MindDrill — Backend Solution Note

**Version:** 1.0
**Purpose:** Explain *how* the backend is built and *why* each choice was made.

---

## 1. Guiding principle

The project must *look* enterprise-grade to a reviewer while running on a hobby budget. So we spend complexity only where it shows judgment, and stay boring everywhere else. Every added tool is checked against real need, not popularity.

## 2. The shape: split by lifecycle

The three layers are conceptual. The real system splits into two paths that share one database.

- **Ingestion path (offline, async):** load → chunk → embed → store. Slow is fine.
- **Query path (online, sync, traced):** question → retrieve → re-rank → assemble → infer → cited answer. Must stay fast.

**Why split them:** ingestion is write-heavy and rare; queries are read-heavy and interactive. If they share one flow, a big upload blocks a user's question. Separate code paths, one Postgres.

## 3. Datastore — one Postgres, three jobs

We use a single Postgres + pgvector. It holds:

- **Relational data** (documents, chunks, sessions, messages, jobs) → normal SQL tables.
- **Vectors** (embeddings) → `pgvector` column.
- **Keyword search** → true BM25 via ParadeDB `pg_search` (a `USING bm25` index).
- **Flexible fields** (metadata, tool args, trace extras) → `JSONB` column.

**Why:** you already need Postgres. pgvector removes the need for a separate vector DB. One system to run, back up, and reason about. Chunk + vector commit together (transactional consistency).

**Trade-off / where the line is:** pgvector's vector index is weaker than a purpose-built engine past millions of vectors. At our scale this never bites. Say this in the README — it shows you know the limit.

**BM25 engine:** we use ParadeDB `pg_search` for true Okapi BM25 (TF saturation, IDF, length normalization) — `ts_rank` lacks IDF and ranks poorly. The ParadeDB Docker image bundles pgvector, so it stays one database. Cost: most free *managed* Postgres can't run `pg_search` (Neon dropped it, Supabase lacks it), so we self-host the ParadeDB container in Compose. Keep the keyword arm behind an interface so `ts_rank` is a fallback if a host ever forces vanilla Postgres.

**Corner-cut warning:** don't dump core fields into JSONB. If you filter or join on it (like `document_id`, `status`), make it a real column. JSONB is for loose, varying data only.

## 3b. Authentication and per-user isolation

Username/password → JWT. Verify the password (bcrypt/argon2 hash), issue a signed JWT with `user_id` and an expiry, sent as `Authorization: Bearer <jwt>`.

Isolation has two layers, and the second is the one people forget:

- **API layer:** a FastAPI auth dependency validates the JWT and resolves `current_user`.
- **Query layer (the real boundary):** every data read filters by `user_id` — documents, chunks, sessions, messages, and *both* arms of hybrid retrieval. API auth alone does not stop a user from retrieving another user's chunks; the query filter does.

**Ownership model:** `documents`, `sessions`, and `ingestion_jobs` carry `user_id`. `chunks` also carry a denormalized `user_id` (copied from the owning document) so retrieval filters without a join. `messages` inherit ownership through `session_id`.

**Idempotency is per-user:** the same file uploaded by two users produces two separate, isolated copies. The content-hash unique constraint is on `(user_id, content_hash)`, not global.

Keep it simple: access token only for v1 (refresh tokens later if needed). Store passwords hashed, never plaintext.

## 4. Ingestion path — a real worker queue

Embedding a large PDF takes minutes, so ingestion must be async. Once async, a durable queue (**Celery + Redis**) is justified, not decoration:

- **Retries with backoff** — embedding APIs rate-limit; a queue retries.
- **Status polling** — caller gets a `job_id` and polls progress.
- **Restart durability** — a redeploy mid-job resumes, not restarts.
- **Idempotency** — same content hash → no duplicate chunks.
- **Dead-letter** — a poison document fails cleanly with a reason.
- **Worker/API separation** — worker scales apart from the API.

Flow: `POST /ingest` persists a job, returns `job_id` → worker pulls it → parse/chunk/embed with retries → write to Postgres → `GET /ingest/{job_id}` reports status.

**Do not queue the query path.** That asymmetry is the point.

## 5. Query path — the hand-written hot path

This is the centrepiece. Keep it explicit; do not hide it behind framework abstractions.

Steps and seams:

1. **Query embed** — embed the question (same model/dim as ingestion).
2. **Hybrid retrieve** — run two arms in parallel and fuse with **Reciprocal Rank Fusion (RRF)**:
   - *Semantic arm:* pgvector cosine search on `embedding`.
   - *BM25 keyword arm:* true Okapi BM25 via ParadeDB `pg_search` (`USING bm25` index, `@@@` operator, `paradedb.score()`). `ts_rank` kept only as a vanilla-Postgres fallback behind the same interface.
   - *Why RRF:* cosine scores and keyword ranks live on different scales. RRF fuses by rank, so no fragile normalization.
   - *Isolation seam (critical):* **both** arms must filter `WHERE user_id = current_user`. Miss it in one arm and a user can retrieve another user's chunks. This is the real multi-tenant boundary.
   - *Receives:* query, top_k, user_id. *Returns:* candidate chunks with source metadata.
3. **Cross-encoder re-rank (local)** — re-score query–chunk pairs, keep top 5.
   - *Why local:* free, no Cohere cost or rate limit. Put it behind a `Reranker` interface so Cohere can swap in later.
   - *Seam risk:* CPU-bound. Run in a threadpool so it does not block the async loop.
4. **Grounding check** — if top chunks do not support the question, emit a `decline` event and never open the token stream.
5. **Prompt assemble** — build the prompt with a token budget and numbered citation markers (`[1]`, `[2]`) that point to the sources sent to the client.
6. **Inference** — call the provider interface, stream tokens.

## 6. Inference engine — streaming-first, provider-agnostic

**The one rule you must not defer:** the streaming generator is the primitive; the string version is a thin wrapper that drains it. You can collapse a stream into a string cheaply, but you cannot split a string back into a stream.

```
stream(prompt) -> async iterator of chunks      # primitive
generate(prompt) -> str                          # joins the stream
generate_structured(prompt, schema) -> Model     # buffer + validate + retry
```

**Providers:** Gemini (free, primary) → OpenRouter (fallback). Streaming works on free tiers; the real free-tier limit is *rate* (429), not streaming.

**Local models (offline benchmark only):** capped at ≤3GB RAM → 2–3B dense models at Q4_K_M (e.g. Llama 3.2 3B, Qwen 2.5 3B). Benchmark on the same hardware, quant, and size band for a fair comparison. 7B models are out of budget.

**Failover before the stream opens:** a 429 arrives before any token. So we fail over to the next provider *before* opening the SSE stream. The user never sees a broken stream.

**Use LangChain here, carefully:** LangChain's chat-model wrappers give multi-provider support, streaming, and typed content blocks (with citations) for free. Use them as the per-provider *adapter inside* our own interface. Keep our failover and stream orchestration on top. We get the convenience; the seam stays ours.

## 7. Streaming and validation — resolving the clash

Schema validation needs the whole object; streaming sends partial text. Once tokens are sent, you cannot retry them. So:

- **Chat text → stream.** No schema. Citations are inline markers referencing the `sources` event.
- **Machine-consumed output (tool inputs, DB writes, agent steps) → buffer, validate with Pydantic, retry once.**

Key insight: most "structure" (which sources, cost, latency, grounded flag) we compute ourselves from metadata and usage. We don't need the model to emit it as JSON. So streaming and validation rarely fight.

## 8. Typed SSE event protocol

Even with no frontend, define the event contract now and test with `curl`. Transport (SSE) sits behind this protocol so a future WebSocket swap never touches business logic.

Events: `status`, `sources` (sent once, before tokens), `token`, `tool_call`, `tool_result`, `done` (usage, TTFT, latency, grounded flag), `decline`, `error`.

Also handle **client disconnect**: if the client drops, stop generating so we stop paying for tokens.

## 9. Agent harness

- **Tools:** registry of name + Pydantic input schema + function.
- **Loop:** `create_agent` on LangGraph. Middleware gives loop-level guardrails (max tool steps, history trimming, PII) that we would otherwise hand-roll.
- **Streaming with tools:** the same event protocol, now emitting `tool_call` and `tool_result` between tokens.

**Why LangGraph here and not elsewhere:** a tool loop is genuine branching control flow. That is exactly what LangGraph is for. The linear RAG path is not, so it stays hand-written.

## 10. Observability

Self-hosted Langfuse. Wrap the query path in traced spans using its decorator/context manager so tracing is a cross-cut, not smeared through logic.

**Tee the stream:** while yielding tokens, accumulate the full text; on completion, log the whole response + usage in one span. Streaming unlocks **time-to-first-token** as a real, portfolio-worthy metric.

## 11. Evaluation and CI/CD

Ragas + a golden dataset (50–200 Q&A pairs) run in GitHub Actions on every PR. Gate on faithfulness and context precision. This track is **offline and server-independent**. A failing build on a quality regression is a strong maturity signal.

## 12. Deployment

One `docker compose up` brings up API, worker, **Postgres (ParadeDB image, with pgvector + pg_search)**, Redis, and Langfuse. Health checks, graceful startup/shutdown, env-based config. Deploy to a cheap host. The local SLM benchmark is a separate offline report, run on your own hardware.

**Skip (over-engineering):** SSE resumption, reconnection logic, WebSocket, autoscaling, a separate vector DB, a message queue on the query path.