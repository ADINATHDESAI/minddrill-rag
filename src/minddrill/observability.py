"""Langfuse tracing seam.

`get_langfuse()` is the single client seam other modules depend on (mirrors
`get_reranker`/`get_embedder`), so tests can override it with a recording fake.
`trace_id_for` derives a Langfuse trace id from the logging middleware's
`request_id`: Langfuse (OTEL-based) requires a 32-char lowercase hex trace id,
which a `uuid4` string is not, so the id is reproducibly derived via
`create_trace_id(seed=...)` rather than minting an unrelated second id. The
same `request_id` is also stamped into trace metadata so a log line and a trace
are both reachable from the one value in `X-Request-ID`.

Never pass secrets, JWTs, or auth headers into any span's input/metadata.
"""

from functools import lru_cache

from langfuse import Langfuse

from minddrill.config import get_settings


@lru_cache
def get_langfuse() -> Langfuse:
    settings = get_settings()
    # `tracing_enabled=False` makes every call a local no-op (no network, no
    # buffering) rather than erroring, so dev/tests run fine with no keys set.
    return Langfuse(
        public_key=settings.langfuse_public_key or None,
        secret_key=settings.langfuse_secret_key or None,
        base_url=settings.langfuse_base_url or None,
        tracing_enabled=settings.langfuse_enabled,
    )


def trace_id_for(request_id: str) -> str:
    """Deterministic Langfuse trace id: same request_id -> same trace id."""
    return Langfuse.create_trace_id(seed=request_id)


__all__ = ["get_langfuse", "trace_id_for"]
