"""Streaming /chat: short-term memory only, no retrieval.

Mirrors the query streaming path (failover resolved before the stream opens),
but the prompt is built from conversation history rather than retrieved
documents. The user turn is persisted before generating; the assistant turn is
persisted only after a successful generation.
"""

import time
import uuid
from collections.abc import AsyncIterator, Sequence

import structlog
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.config import get_settings
from minddrill.db.session import SessionLocal
from minddrill.models.message import Message
from minddrill.models.session import ChatSession
from minddrill.providers.base import LLMProvider
from minddrill.providers.failover import open_stream
from minddrill.rag import events
from minddrill.sessions.memory import (
    build_chat_messages,
    estimate_tokens,
    load_history,
    trim_history,
)

log = structlog.get_logger(__name__)


async def run_chat(
    session: AsyncSession,
    chat_session: ChatSession,
    message: str,
    providers: Sequence[LLMProvider],
) -> AsyncIterator[dict]:
    """Persist the user turn, build memory, resolve failover, return the stream.

    Failover is resolved here, before the returned generator yields, so an
    all-providers-down failure surfaces as `ProvidersUnavailable` (→ 503) rather
    than a broken half-open stream.
    """
    start = time.perf_counter()

    session.add(
        Message(
            session_id=chat_session.id,
            role="user",
            content=message,
            token_count=estimate_tokens(message),
        )
    )
    chat_session.last_active_at = func.now()
    await session.commit()

    history = await load_history(session, chat_session.id)
    trimmed = trim_history(history, get_settings().memory_token_budget)
    prompt = build_chat_messages(trimmed)

    tokens, provider = await open_stream(providers, prompt)
    ttft_ms = int((time.perf_counter() - start) * 1000)
    return _stream_chat(chat_session.id, start, tokens, provider, ttft_ms)


async def _stream_chat(
    session_id: uuid.UUID,
    start: float,
    tokens: AsyncIterator[str],
    provider: LLMProvider,
    ttft_ms: int,
) -> AsyncIterator[dict]:
    yield events.status("generating")
    parts: list[str] = []
    try:
        async for tok in tokens:
            parts.append(tok)
            yield events.token(tok)
    except Exception as exc:  # failure after the stream opened
        log.warning("chat.stream_error", session_id=str(session_id), error=str(exc))
        yield events.error("internal_error", "generation failed mid-stream")
        return
    finally:
        await tokens.aclose()

    # Only reached on a clean generation. A fresh session decouples this write
    # from the request-scoped session's teardown while the stream body is sent.
    text = "".join(parts)
    try:
        async with SessionLocal() as write:
            write.add(
                Message(
                    session_id=session_id,
                    role="assistant",
                    content=text,
                    token_count=estimate_tokens(text),
                )
            )
            await write.commit()
    except Exception as exc:
        # Tokens already reached the client; surface the persistence failure as an
        # error event rather than letting it escape the generator unreported.
        log.warning("chat.persist_error", session_id=str(session_id), error=str(exc))
        yield events.error("internal_error", "failed to persist reply")
        return

    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = getattr(provider, "last_usage", None) or {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    log.info(
        "chat.answered",
        session_id=str(session_id),
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
    )
    yield events.done(usage, ttft_ms, latency_ms, grounded=False)
