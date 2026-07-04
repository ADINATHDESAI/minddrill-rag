# MindDrill — Project Requirement Document

**Version:** 1.0
**Status:** Living document
**Owner:** Jyoti

---

## 1. What this project is

MindDrill is an enterprise-style "Ask My Doc" assistant. You give it documents. It answers questions from those documents, with citations, and it can decline when the documents do not support an answer.

It is a **backend-only, production-grade RAG system**. It is also a **portfolio centrepiece** — built to show real AI engineering depth to a technical reviewer, not to serve heavy traffic.

## 2. Goals

- Show that I can design and build a production-ready RAG + inference + observability system.
- Show good engineering judgment: the right tool in the right place, and clear reasons why.
- Keep cost near zero using free tiers, without faking "enterprise" claims.
- Learn FastAPI, Pydantic, and the LangChain/LangGraph 1.0 stack by building real slices.

## 3. Non-goals (out of scope)

- **No frontend.** The backend exposes a clean API. A UI can be added later by anyone.
- **No high traffic.** Shared with a few friends only. No autoscaling, no load balancing.
- **No serving of local SLMs in production.** Local models are for an offline benchmark report only. A cheap server cannot run a 7B model well.
- **No long-term cross-session memory in v1.** Added only if a real need appears.

## 4. Scope — the three layers

### Layer 1 — RAG pipeline
- Ingest PDF, Markdown, and web pages.
- Chunk into 500–800 token pieces with ~100 token overlap.
- Store embeddings and text in one Postgres + pgvector database.
- Retrieve with **hybrid search**: semantic (pgvector) + **true BM25 keyword search** (ParadeDB `pg_search`, Okapi BM25 — from day one), fused with Reciprocal Rank Fusion (RRF). `ts_rank` kept only as a vanilla-Postgres fallback behind the same interface.
- Re-rank top results with a **local cross-encoder** (Sentence Transformers).
- Return answers with exact source citations. Decline if unsupported.
- Golden eval dataset (50–200 Q&A pairs) with an offline CI quality gate.

### Layer 2 — Hybrid inference engine
- One streaming-first provider interface.
- Cloud providers behind it: **Google Gemini (free tier, primary)**, **OpenRouter (fallback)**. Anthropic/OpenAI optional for the benchmark.
- Local providers via Ollama — **for the offline benchmark only**. Hard cap: **≤3GB RAM per model**, so only **2–3B dense models at Q4_K_M** (e.g. Llama 3.2 3B ~2.5GB, Qwen 2.5 3B ~1.9GB). 7B models are excluded — they need ~4.5GB at Q4.
- Structured JSON output with Pydantic validation + one retry — **only where a machine consumes the output** (tools, DB writes, agent steps), not for streamed chat text.
- A data-driven benchmark report on the **same hardware, same Q4_K_M quant, same size band** (fair comparison — don't pit a 1B against a 3B): tokens/sec, time-to-first-token, latency, memory, quality.

### Layer 3 — Observability
- Trace every request: retrieval → re-ranking → prompt → response → tokens.
- Metrics: P50/P95 latency, time-to-first-token, cost per request, citation coverage, failure rate.
- Self-hosted **Langfuse** dashboard.
- Regression gating tied to CI/CD: if faithfulness drops below a threshold, block the build.

## 5. The two execution paths (core design rule)

The system splits by **lifecycle**, not by layer:

- **Ingestion path — asynchronous.** Long, failure-prone, restart-sensitive. Runs on **Celery + Redis** with retries, status polling, idempotency, and a dead-letter path.
- **Query path — synchronous.** Short and interactive. Hand-written, streaming, traced. This is the portfolio centrepiece. **Never queued.**

Being able to explain *why ingestion is queued and the query path is not* is a key maturity signal.

## 6. The agent harness

- **Sessions:** each conversation has an id; turns stored in Postgres.
- **Memory (v1):** short-term only — recent turns trimmed to a token budget. Summarize old turns when history is too long. Keep conversation memory separate from RAG retrieval.
- **Tools:** calculator, weather API, web search, knowledge-base search. Each tool = name + Pydantic input schema + function.
- **Agent loop:** built on **LangGraph `create_agent`** (post-1.0). Middleware handles guardrails (max tool steps, history trimming, PII).
- **Two modes:**
  1. Direct doc question → always retrieve (the synchronous RAG path).
  2. Open chat → agent loop with tools, where retrieval is one tool among many.

## 7. Functional requirements

- FR1: Ingest a document and report job status.
- FR2: Re-ingesting the same content must not duplicate chunks (idempotency by content hash).
- FR3: Answer a question from ingested docs with citations.
- FR4: Decline clearly when retrieval does not support the question.
- FR5: Stream the answer token-by-token over SSE.
- FR6: Fail over between providers before the stream opens.
- FR7: Run tools inside an agent loop with a step limit.
- FR8: Trace every request and expose metrics in Langfuse.
- FR9: Run an offline eval gate on every pull request.
- FR10: Authenticate users with username/password and issue a JWT.
- FR11: Isolate data per user — every read is scoped to the authenticated user, including both retrieval arms.

## 8. Non-functional requirements

- **Cost:** free tiers first; cheapest paid option only if unavoidable.
- **Async correctness:** the query path must be async end to end; no blocking calls on the event loop.
- **Portability:** all context (decisions, specs, tests) lives in the repo so any AI coding tool can pick it up.
- **Observability:** no request runs untraced.
- **Reliability:** ingestion survives a worker restart mid-job.
- **Security:** passwords stored hashed (bcrypt/argon2), never plaintext; JWT signed with a secret from config; per-user isolation enforced in every data query.

## 9. Authentication and data isolation (locked)

Each user sees only their own data. One user must never read another user's documents, chunks, sessions, or messages.

- **Auth:** username + password → **JWT** (signed access token). On login, verify the password (hashed with bcrypt/argon2), issue a JWT carrying `user_id` and an expiry. Client sends it as `Authorization: Bearer <jwt>`.
- **Isolation is enforced at the query, not just the API.** Every data read — including *both* arms of hybrid retrieval — filters `WHERE user_id = current_user`. API-level auth alone is not enough; the retrieval filter is the real boundary.
- **Scope:** access token only for v1. Refresh tokens are an easy later add; skip for now (don't over-engineer).
- Keep auth behind one FastAPI dependency so the scheme can change without touching route logic.

## 10. Tech stack

- **Framework:** FastAPI, Pydantic
- **Auth:** JWT (pyjwt or python-jose), password hashing (passlib + bcrypt/argon2)
- **Data:** Postgres (ParadeDB image) + pgvector + `pg_search` BM25 (single store), Redis (broker)
- **Retrieval:** pgvector + `pg_search` BM25, fused via RRF; Sentence Transformers cross-encoder
- **Orchestration:** LangGraph (`create_agent`), LangChain (loaders, splitters, model adapters)
- **Async ingestion:** Celery
- **Inference:** Google Gemini (free), OpenRouter (fallback), Ollama (offline benchmark)
- **Eval:** Ragas
- **Observability:** self-hosted Langfuse
- **CI/CD:** GitHub Actions
- **Deploy:** Docker Compose (API, worker, Postgres, Redis, Langfuse)

## 11. Deliverables

- Running backend (Docker Compose, one command).
- RAG pipeline with hybrid retrieval + local re-ranking.
- Streaming inference engine with provider failover.
- Agent harness with tools.
- Langfuse dashboard.
- Golden eval dataset + CI gate.
- Local SLM benchmark report (offline).
- Repo docs: this PRD, backend solution note, API spec, DB schema, CLAUDE.md, DECISIONS.md.

## 12. Rough build order

1. Repo layout, Docker Compose (Postgres, Redis), Pydantic settings, provider interface stub, `/health`.
2. Thin vertical slice: ingest one PDF → store → retrieve → plain answer.
3. Thicken: hybrid + RRF + re-ranking.
4. Streaming + provider failover (SSE).
5. Agent loop + tools.
6. Observability (Langfuse) + eval gate (Ragas + GitHub Actions).
7. Local SLM benchmark report.