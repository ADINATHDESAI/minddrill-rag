# MindDrill — Database Schema

**Version:** 1.2
**Engine:** PostgreSQL via the **ParadeDB image** (single store)
**Extensions:** `vector` (pgvector) + `pg_search` (ParadeDB, true Okapi BM25) — both from day one, bundled in the ParadeDB Postgres image
**Type:** SQL (relational), with a vector column, a BM25 index, and `JSONB` for loose fields.

One database does the work people usually spread across three: relational store, vector DB, and search engine. All data is **scoped per user**.

---

## Design rules

- Core fields you filter or join on → **real columns** (typed, indexed).
- Loose, varying fields → **`JSONB`**.
- Embeddings → **`vector`** column. Dimension is fixed by the embedding model and cannot change without re-embedding.
- BM25 keyword search → **true BM25 via `pg_search`**: a `USING bm25` index on the `content` column (Okapi BM25 — TF saturation, IDF, length normalization). `tsvector`/`ts_rank` kept only as a fallback if a host forces vanilla Postgres.
- Idempotency for ingestion → unique on **(`user_id`, `content_hash`)** — per-user, so users never share rows.
- Per-user isolation → `user_id` on `documents`, `sessions`, `ingestion_jobs`; **denormalized onto `chunks`** so retrieval filters without a join.

> Set the embedding dimension once (e.g. `768` for a 768-dim model). Keep it in config. Changing it means re-embedding everything.

---

## Tables

### `users`
One row per account.

| Column          | Type          | Notes |
|-----------------|---------------|-------|
| `id`            | `uuid` PK     | |
| `username`      | `text` UNIQUE | login name |
| `password_hash` | `text`        | bcrypt/argon2 — never plaintext |
| `created_at`    | `timestamptz` | default now |

**Indexes:** unique on `username`.

---

### `documents`
One row per ingested source, owned by a user.

| Column         | Type          | Notes |
|----------------|---------------|-------|
| `id`           | `uuid` PK     | |
| `user_id`      | `uuid` FK → users(id) | owner; on delete cascade |
| `source_type`  | `text`        | `pdf` \| `markdown` \| `web` |
| `source_uri`   | `text`        | original location |
| `title`        | `text`        | nullable |
| `content_hash` | `text`        | idempotency; unique **per user** (see index) |
| `status`       | `text`        | `pending` \| `processing` \| `done` \| `failed` |
| `metadata`     | `jsonb`       | loose fields |
| `created_at`   | `timestamptz` | default now |

**Indexes:** unique on **(`user_id`, `content_hash`)**; index on `user_id`; index on `status`.

---

### `chunks`
The retrieval unit. Holds text, its embedding, and its BM25 index.

| Column        | Type            | Notes |
|---------------|-----------------|-------|
| `id`          | `uuid` PK       | BM25 index `key_field` |
| `document_id` | `uuid` FK → documents(id) | on delete cascade |
| `user_id`     | `uuid`          | denormalized from the owning document (see note) |
| `chunk_index` | `int`           | order within the document |
| `content`     | `text`          | the chunk text; the BM25 index is built on this |
| `token_count` | `int`           | for budgeting |
| `embedding`   | `vector(768)`   | semantic search |
| `content_tsv` | `tsvector`      | optional fallback only; primary BM25 is a `pg_search` index on `content` |
| `metadata`    | `jsonb`         | e.g. page number, heading |
| `created_at`  | `timestamptz`   | default now |

**Indexes:**
- Vector: HNSW on `embedding` with `vector_cosine_ops` (or IVFFlat for lower memory).
- BM25: `pg_search` index — `CREATE INDEX chunks_bm25 ON chunks USING bm25 (id, content, user_id) WITH (key_field='id');`. Query with the `@@@` operator, rank with `paradedb.score(id)`. Including `user_id` lets the isolation filter push into the index.
- FK: index on `document_id`.
- Isolation: index on `user_id`. Both retrieval arms filter on `user_id = current_user` **before** ranking.
- Optional: GIN on `metadata` if you filter inside it.
- Fallback only: GIN on `content_tsv` if you ever run vanilla Postgres without `pg_search`.

This table powers **hybrid retrieval**: pgvector on `embedding` + true BM25 (`pg_search`) on `content`, fused with RRF in the query layer — always scoped to the caller.

> `user_id` is denormalized onto `chunks` on purpose: it lets filtered vector and BM25 search scope to the user without joining `documents`. With very selective filters, pgvector HNSW may need iterative scan for full recall — fine at this scale.

---

### `ingestion_jobs`
Tracks async ingestion (Celery). Backs `GET /ingest/{job_id}`.

| Column        | Type          | Notes |
|---------------|---------------|-------|
| `id`          | `uuid` PK     | the `job_id` returned to the client |
| `user_id`     | `uuid` FK → users(id) | owner |
| `document_id` | `uuid` FK     | nullable until the doc row exists |
| `source_type` | `text`        | |
| `source_uri`  | `text`        | |
| `status`      | `text`        | `pending` \| `processing` \| `done` \| `failed` |
| `error`       | `text`        | reason on failure (dead-letter) |
| `retry_count` | `int`         | default 0 |
| `created_at`  | `timestamptz` | default now |
| `updated_at`  | `timestamptz` | updated on each state change |

**Indexes:** index on `status`; index on `document_id`; index on `user_id`.

---

### `sessions`
One conversation, owned by a user.

| Column          | Type          | Notes |
|-----------------|---------------|-------|
| `id`            | `uuid` PK     | |
| `user_id`       | `uuid` FK → users(id) | owner; NOT NULL |
| `title`         | `text`        | nullable |
| `created_at`    | `timestamptz` | default now |
| `last_active_at`| `timestamptz` | |

**Indexes:** index on `user_id`.

---

### `messages`
Turns in a session. This is the short-term memory source. Ownership inherited via `session_id`.

| Column       | Type          | Notes |
|--------------|---------------|-------|
| `id`         | `uuid` PK     | |
| `session_id` | `uuid` FK → sessions(id) | on delete cascade |
| `role`       | `text`        | `user` \| `assistant` \| `system` \| `tool` |
| `content`    | `text`        | |
| `token_count`| `int`         | for trimming to a budget |
| `tool_calls` | `jsonb`       | nullable; tool name + args/results |
| `created_at` | `timestamptz` | default now |

**Indexes:** composite index on `(session_id, created_at)` for fast history reads.

> Keep conversation memory (this table) separate from RAG chunks. They are different sources.

---

### `eval_runs`
One row per CI eval run (Ragas). Backs regression gating. Not user-scoped (offline, system-level).

| Column           | Type          | Notes |
|------------------|---------------|-------|
| `id`             | `uuid` PK     | |
| `git_sha`        | `text`        | commit under test |
| `dataset_version`| `text`        | golden set version |
| `summary`        | `jsonb`       | aggregate scores: faithfulness, context_precision, etc. |
| `created_at`     | `timestamptz` | default now |

---

### `eval_results`
Per-question results for a run.

| Column      | Type          | Notes |
|-------------|---------------|-------|
| `id`        | `uuid` PK     | |
| `run_id`    | `uuid` FK → eval_runs(id) | on delete cascade |
| `question`  | `text`        | |
| `expected`  | `text`        | golden answer |
| `actual`    | `text`        | system answer |
| `scores`    | `jsonb`       | per-metric scores |
| `passed`    | `boolean`     | above threshold? |

**Indexes:** index on `run_id`.

---

## Relationships (summary)

```
users     1───∞ documents
users     1───∞ sessions
users     1───∞ ingestion_jobs
documents 1───∞ chunks
documents 1───∞ ingestion_jobs
sessions  1───∞ messages
eval_runs 1───∞ eval_results
```

---

## Where each special type lives

| Need                | Column                          | Mechanism |
|---------------------|---------------------------------|-----------|
| Semantic search     | `chunks.embedding`              | pgvector + HNSW |
| BM25 keyword search | `chunks.content` (BM25 index)   | ParadeDB `pg_search` — true Okapi BM25 (`USING bm25`, `@@@`) |
| Loose metadata      | `*.metadata`, `messages.tool_calls`, `eval_*` | JSONB |
| Idempotency         | (`documents.user_id`, `content_hash`) | composite UNIQUE (per user) |
| Per-user isolation  | `user_id` on documents/sessions/jobs, denormalized on chunks | filtered in every read |
| Passwords           | `users.password_hash`           | bcrypt/argon2 hash |
| Job status          | `ingestion_jobs.status`         | indexed column |

---

## Migration note

Use a migration tool (Alembic). A few things need explicit steps that ORMs do not auto-handle well:
- `CREATE EXTENSION IF NOT EXISTS vector;`
- `CREATE EXTENSION IF NOT EXISTS pg_search;` — required from day one; use the **ParadeDB Postgres image** (it bundles pgvector too). Then create the `USING bm25` index on `chunks`.
- Vector index (HNSW/IVFFlat) — create in a migration, tune params for pgvector.
- The `(user_id, content_hash)` composite unique constraint on `documents`.
- **Hosting caveat:** most free *managed* Postgres won't run `pg_search` (Neon dropped it for new projects in 2026; Supabase lacks it). Self-host the ParadeDB container in Docker Compose. Keep the keyword arm behind an interface so `ts_rank` stays a vanilla-Postgres fallback.
- **Alternative BM25 engine:** TigerData `pg_textsearch` (`to_bm25query()` syntax) is an equivalent Okapi BM25 option if you end up in the TimescaleDB ecosystem.