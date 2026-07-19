# Slice 8 — Langfuse tracing

Instrument both paths (/query and /chat) as one trace per request:
- Use the correlation id (request_id from the logging middleware) as the
  Langfuse trace id, so a log line and a trace point to the same request.
- Spans: retrieve (chunk count, scores), rerank (top_n), generate (provider,
  model, prompt, response). Use @observe on the hand-written query path.
- For the LangGraph agent (/chat), use the Langfuse callback handler so the
  agent + tool steps are traced automatically.

Tee the stream: while yielding tokens, accumulate the full text; on completion,
log the full response + usage (input/output tokens) to the generate span. Never
block the stream to log — log on completion.

Metrics to capture: ttft_ms, total latency_ms, tokens, cost (from usage),
grounded flag, provider used (and whether failover happened).

Privacy: never send secrets or JWTs to Langfuse. Sending prompt/response text
is fine (it's your own data); do not log another user's data into a shared span.

Tests (mock the Langfuse client):
- a /query produces one trace with retrieve, rerank, generate spans.
- the trace id equals the request_id from the logging middleware.
- usage + ttft are attached to the generate span on completion.

What can go wrong: logging blocking the stream (log on completion only), missing
usage on failover, trace id not matching the log id.