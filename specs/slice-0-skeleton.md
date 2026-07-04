# Slice 0 — Skeleton

Scaffold the app. No business logic.

Deliver:
- docker-compose.yml with two services:
  - db: image paradedb/paradedb:latest (Postgres + pgvector + pg_search),
    POSTGRES_USER/PASSWORD/DB = minddrill, port 5432.
  - redis: image redis:7, port 6379.
- src/minddrill/config.py: Pydantic Settings loading DATABASE_URL, REDIS_URL,
  JWT_SECRET, GEMINI_API_KEY, EMBED_DIM (default 768) from .env.
- src/minddrill/main.py: FastAPI app with GET /health -> {"status":"ok"}.
- src/minddrill/db/session.py: async SQLAlchemy engine + session maker.
- src/minddrill/providers/base.py: an LLMProvider Protocol (stream + generate),
  streaming method is the primitive. Stub only, no implementation.
- src/minddrill/auth/deps.py: get_current_user dependency STUB that raises
  NotImplementedError (real logic in slice 1).
- alembic initialised, pointed at DATABASE_URL, async-friendly env.py.
- tests/test_health.py: asserts GET /health returns 200 and {"status":"ok"}.

Constraints: async end to end. Follow CLAUDE.md rules.