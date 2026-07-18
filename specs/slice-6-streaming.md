# Slice 6 — Streaming /query over SSE, with provider failover

Providers (behind the existing LLMProvider interface, stream() is the primitive):
- GeminiProvider (already exists from slice 2): real streaming via google-genai.
- OpenRouterProvider: OpenAI-compatible streaming (openai client, base_url =
  https://openrouter.ai/api/v1, key from settings).

Failover (before the stream opens):
- Try the primary (Gemini). If it errors or 429s BEFORE the first token, fall
  back to OpenRouter. Only once a provider yields its first token do we open the
  SSE response. The client never sees a mid-open failover.

/query becomes SSE (text/event-stream) via sse-starlette EventSourceResponse.
Event order:
- (optional) status: {state:"retrieving"|"generating"}
- sources: ONE event, before any token — the reranked top-5 chunks
  [{id, document_id, chunk_id, score}]
- token: many — {text: delta}
- done: last — {usage, ttft_ms, latency_ms, grounded:true}
- decline: INSTEAD of tokens if grounding fails — {reason}
- error: on mid-stream failure — {code, message}

Grounding gate runs BEFORE streaming: if reranked chunks don't support the
question, emit decline and never open the token stream.

TTFT: record the timestamp of the first token; include ttft_ms in done.
Client disconnect: if the client drops, cancel generation (stop token spend).
Citations: the model writes [1],[2] inline; they map to the sources event ids.

Tests (mock the providers):
- event order: sources before tokens, done last.
- failover: primary raises before first token -> OpenRouter used -> stream ok.
- decline path: weak retrieval -> decline event, no token events.
- disconnect: simulated client drop cancels the generator.

What can go wrong: both providers down (503 before stream), first-token latency,
proxy buffering (disable buffering for text/event-stream).