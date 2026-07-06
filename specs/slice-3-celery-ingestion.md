# Slice 3 — Celery ingestion (async, durable)

Move the slice-2 synchronous ingestion into a Celery task. The query path
stays synchronous — do NOT touch it.

Table:
- ingestion_jobs(id, user_id, document_id nullable, source_type, source_uri,
  status, error, retry_count, created_at, updated_at). See docs\04_DATABASE_SCHEMA.md.

Flow:
- POST /ingest {source_type, source_uri} for current_user:
  - create an ingestion_jobs row (status=pending), enqueue a Celery task with
    (job_id, user_id, source_uri), return 202 {job_id, status:"pending"}.
- Celery task ingest_document(job_id, user_id, source_uri):
  - set status=processing.
  - idempotency: if (user_id, content_hash) exists, link document_id, set done.
  - else load -> split -> embed -> insert document + chunks (with user_id),
    set status=done, link document_id.
  - on failure: set status=failed, save error, honor retry_count (max 3,
    exponential backoff). A poison doc ends in failed, not an infinite loop.
- GET /ingest/{job_id}: return status/document_id/error. 404 if not owned.

Config:
- Celery broker + result backend = Redis (REDIS_URL). One worker service in
  docker-compose (command: celery -A ... worker).

Tests:
- POST /ingest returns 202 + job_id; job row created as pending.
- task success -> status done, chunks stored with correct user_id.
- re-ingest same content by same user -> no duplicate chunks (idempotent).
- task failure path -> status failed with an error message.
- GET on another user's job -> 404.
- (Run the task synchronously in tests via task_always_eager.)

What can go wrong: worker can't reach Redis, embed API 429 (retry), job row
and document row out of sync on crash (set status transactionally).