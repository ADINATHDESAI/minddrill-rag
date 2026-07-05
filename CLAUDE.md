# MindDrill

A backend-only, production-grade "Ask My Doc" RAG assistant. Portfolio centrepiece — favour clear engineering judgment over shiny tools.

## Before you build
- Read the matching file in `specs/` first. Detail lives there, not here.
- Full context: `docs/01`–`docs/04` (PRD, solution note, API spec, DB schema) and `docs/DECISIONS.md`.
- Work in one vertical slice at a time. Small changes over large autonomous runs.
- Show evidence — paste real test output — before claiming anything works.

## Locked rules (prefer these; flag me if a task conflicts)
- **One Postgres** via the ParadeDB image: pgvector (semantic) + `pg_search` true BM25 + JSONB. No separate vector DB.
- **Hybrid retrieval** = pgvector + `pg_search` BM25, fused with RRF (k=60). Cross-encoder rerank is local (Sentence Transformers), run in a threadpool.
- **Celery + Redis for ingestion only.** The query path is synchronous and hand-written. Never queue it.
- **Per-user isolation:** every data read filters `user_id = current_user`, including *both* retrieval arms. Auth is username/password → JWT; passwords hashed.
- **Inference is streaming-first:** the async stream is the primitive; the string version wraps it. Gemini free → OpenRouter fallback; fail over *before* the SSE stream opens.
- **SSE with typed events** (`status`, `sources`, `token`, `tool_call`, `tool_result`, `decline`, `done`, `error`). Citations are inline markers referencing the `sources` event.
- **Pydantic validation only for machine-consumed output** (tool inputs, DB writes, agent steps) — not streamed chat text.
- **LangGraph `create_agent` for the agent/tool loop only.** Keep the RAG hot path explicit. Use LangChain for loaders/splitters and as model adapters inside our own provider interface — not for retrieval logic.
- **Short-term memory only** (recent turns trimmed to a token budget). Keep it separate from RAG retrieval.
- Local SLMs are for the **offline benchmark only** (≤3GB RAM, 2–3B Q4). Never served in prod.

## Conventions
- Python 3.12+, FastAPI, Pydantic v2, async end to end (no blocking calls on the event loop).
- Package/deps via `uv`. Lint/format via `ruff`. Migrations via `alembic`.
- Tests: `pytest`. Write the failing test from the spec first, then the code.
- Prompts and config are version-controlled files, loaded by id.

## Test command
`pytest -q`

## Coding principles
- Default to KISS and YAGNI. Build only what the current slice needs.
- Apply SOLID and DRY at interface seams (provider, reranker, retriever,
  tool registry) — not in the RAG hot path, which stays explicit.
- Rule of three: allow duplication until the third repeat, then abstract.
- When principles conflict, KISS/YAGNI win. If an abstraction isn't used
  by at least two callers now, don't add it.

## Comments
- Comment only to explain *why* for non-obvious logic. No narration.
- Never mention slices, specs, CLAUDE.md, the plan, or the build process
  in code or comments.
- Don't restate what the code does. No "as per", no "skeleton only".

## Logging
- Structured JSON logs. Every request carries a request_id (correlation id).
- Log at seams (ingest, retrieve, infer, errors). Never log secrets, JWTs,
  or full document/chunk text — ids only.

