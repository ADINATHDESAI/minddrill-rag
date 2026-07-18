# Slice 7a — Sessions, memory, streaming chat

Tables (migration; see docs/04):
- sessions(id, user_id fk NOT NULL, title, created_at, last_active_at)
- messages(id, session_id fk, role, content, token_count, tool_calls jsonb,
  created_at). Index (session_id, created_at).

Endpoints:
- POST /sessions -> create a session for current_user.
- GET /sessions/{id}/messages -> history (owner only, else 404).
- POST /chat {message, session_id} -> SSE stream (reuse slice-6 streaming).

Memory (short-term only):
- Load recent turns for the session, trim to a token budget (config
  MEMORY_TOKEN_BUDGET). Build the prompt = system + trimmed history + message.
- Persist the user message before generating, and the assistant message after.
- Keep this SEPARATE from RAG retrieval. /chat here does NOT retrieve yet.

Tests:
- create session; post two messages; history returns them in order.
- memory trims when history exceeds the budget (oldest dropped).
- another user's session -> 404 on read.
- /chat streams a reply and persists both messages.

What can go wrong: unbounded history (trim), writing assistant message on a
failed generation (only persist on success), cross-user session access.