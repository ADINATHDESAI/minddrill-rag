# MindDrill — API Specification

**Version:** 1.1
**Base URL:** `/api/v1`
**Auth:** JWT. Register/login with username + password to get a JWT, then send `Authorization: Bearer <jwt>` on every call. Missing/invalid/expired → `401`. **Every data endpoint is scoped to the authenticated user.**

This spec is the interface contract. Any tool (Claude Code, or another) can build against it.

---

## Conventions

- All request/response bodies are JSON, except streaming endpoints which return **SSE** (`text/event-stream`).
- IDs are UUIDs.
- Non-stream errors use one shape:
  ```json
  { "error": { "code": "string", "message": "human readable" } }
  ```
- Stream errors are sent as an `error` event (see SSE protocol).

---

## 1. Health

### `GET /health`
Liveness check. No auth.

**200**
```json
{ "status": "ok", "version": "1.1" }
```

---

## Auth — register & login

### `POST /auth/register`
Create a user.

**Body**
```json
{ "username": "string", "password": "string" }
```
**201**
```json
{ "user_id": "uuid", "username": "string" }
```
**Errors:** `400` weak/invalid input, `409` username taken.

### `POST /auth/login`
Exchange credentials for a JWT.

**Body**
```json
{ "username": "string", "password": "string" }
```
**200**
```json
{ "access_token": "jwt", "token_type": "bearer", "expires_in": 3600 }
```
**Errors:** `401` bad credentials.

Send the token on every other endpoint: `Authorization: Bearer <jwt>`. Passwords are stored hashed (bcrypt/argon2).

---

## 2. Ingestion (async — Celery)

### `POST /ingest`
Start ingesting a document. Returns immediately with a job id. Scoped to the caller.

**Body**
```json
{
  "source_type": "pdf | markdown | web",
  "source_uri": "https://... or file reference",
  "metadata": { "any": "optional loose fields" }
}
```
(For file upload, use `multipart/form-data` with the file plus `source_type`.)

**202 Accepted**
```json
{ "job_id": "uuid", "status": "pending" }
```

**Errors:** `400` bad source_type, `401` unauthorized.

**Note:** re-ingesting the same content by the same user (same `user_id` + content hash) returns the existing `document_id`; it does not duplicate chunks. Idempotency is **per-user**, so two users uploading the same file get separate, isolated copies.

---

### `GET /ingest/{job_id}`
Poll ingestion status. Only the owner can read a job.

**200**
```json
{
  "job_id": "uuid",
  "status": "pending | processing | done | failed",
  "document_id": "uuid | null",
  "error": "string | null"
}
```

**Errors:** `404` unknown job or not owned by caller.

---

## 3. Sessions

### `POST /sessions`
Create a conversation for the caller.

**Body**
```json
{ "title": "optional string" }
```

**201**
```json
{ "session_id": "uuid", "created_at": "iso-8601" }
```

---

### `GET /sessions/{session_id}/messages`
Return conversation history. Only the owner can read it.

**200**
```json
{
  "session_id": "uuid",
  "messages": [
    { "role": "user | assistant | tool", "content": "string", "created_at": "iso-8601" }
  ]
}
```

**Errors:** `404` unknown session or not owned by caller.

---

## 4. Query — direct RAG (SSE stream)

### `POST /query`
Ask a question answered directly from the caller's documents. Always retrieves, scoped to `current_user`. **Streams** the answer.

**Body**
```json
{ "question": "string", "session_id": "uuid | null" }
```

**Response:** `200`, `Content-Type: text/event-stream`.

Provider failover happens **before** the stream opens. If the user's documents do not support the question, the server sends a `decline` event instead of tokens.

See **SSE Event Protocol** below.

---

## 5. Chat — agent mode with tools (SSE stream)

### `POST /chat`
Open chat. The agent may call tools (calculator, weather, web search, knowledge-base search). Retrieval is one tool here, scoped to the caller. **Streams** the answer and tool activity.

**Body**
```json
{ "message": "string", "session_id": "uuid" }
```

**Response:** `200`, `text/event-stream`. Same protocol, plus `tool_call` and `tool_result` events.

Guardrail: the agent loop has a max tool-step limit. On limit, it stops and returns the best answer so far.

---

## SSE Event Protocol

Each event has a named `event:` and JSON `data:`. Order matters.

| Event         | When                        | `data` payload |
|---------------|-----------------------------|----------------|
| `status`      | optional, during retrieval  | `{ "state": "retrieving" \| "generating" }` |
| `sources`     | once, **before any token**  | `{ "sources": [ { "id": 1, "document_id": "uuid", "chunk_id": "uuid", "score": 0.0 } ] }` |
| `token`       | many, during generation     | `{ "text": "delta" }` |
| `tool_call`   | agent mode only             | `{ "tool": "weather", "args": { } }` |
| `tool_result` | agent mode only             | `{ "tool": "weather", "result": { } }` |
| `decline`     | instead of tokens           | `{ "reason": "documents do not support the question" }` |
| `done`        | last event on success       | `{ "usage": { "input_tokens": 0, "output_tokens": 0 }, "ttft_ms": 0, "latency_ms": 0, "grounded": true }` |
| `error`       | on failure mid-stream       | `{ "code": "string", "message": "string" }` |

**Citations:** the model writes inline markers like `[1]`, `[2]` in the streamed text. They map to the `id` fields in the `sources` event (sent first). The client resolves them; no JSON parsing of the live stream is needed.

**Client disconnect:** if the client closes the connection, the server cancels generation to stop token spend.

---

## Error codes (non-stream)

| HTTP | code                | meaning |
|------|---------------------|---------|
| 400  | `bad_request`       | invalid body or params |
| 401  | `unauthorized`      | missing/invalid/expired token |
| 403  | `forbidden`         | authenticated but not the owner |
| 404  | `not_found`         | unknown job/session (or not owned) |
| 409  | `conflict`          | username already taken |
| 429  | `rate_limited`      | app-level rate limit hit |
| 500  | `internal_error`    | unexpected failure |
| 503  | `providers_unavailable` | all inference providers failed before stream |

---

## Notes for implementers

- `/query` and `/chat` are the only streaming endpoints. Everything else is plain JSON.
- Keep transport (SSE) behind the event protocol so a WebSocket swap later touches nothing else.
- Auth is a single FastAPI dependency that resolves `current_user` from the JWT.
- **Every data endpoint is scoped to `current_user`.** Ingestion, query, chat, and sessions only ever touch the caller's own rows. Both hybrid-retrieval arms filter by `user_id` — enforced in the query, not just the API.
- To return `404` (not `403`) for another user's resource is a fine privacy default — it hides existence. Pick one convention and keep it consistent.