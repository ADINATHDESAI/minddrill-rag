"""Provider failover — pick a provider that yields a token before the SSE opens.

The rule the docs commit to: the client never sees a mid-open failover. So we
try providers in order and only commit to one *once it has produced its first
token*. A 429 (or any error) before that first token silently falls through to
the next provider. If every provider fails before a first token, we raise
`ProvidersUnavailable` and the caller returns 503 — no stream is ever opened.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog

from minddrill.providers.base import LLMProvider
from minddrill.providers.gemini import get_llm
from minddrill.providers.openrouter import get_openrouter

log = structlog.get_logger(__name__)


class ProvidersUnavailable(Exception):
    """Every provider failed before yielding a first token."""


async def _prepend(first: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    try:
        yield first
        async for token in rest:
            yield token
    finally:
        # Close the provider generator on normal end, error, or cancellation so
        # its upstream connection is released rather than left to GC.
        await rest.aclose()


async def open_stream(
    providers: Sequence[LLMProvider], messages: Sequence[dict[str, Any]]
) -> tuple[AsyncIterator[str], LLMProvider]:
    """Return (token stream, chosen provider) with the first token already pulled.

    Awaits the first token from each provider in turn; the returned iterator
    replays that first token then drains the rest. A provider that errors, 429s,
    or yields an empty completion before a first token is skipped. Raises
    `ProvidersUnavailable` if no provider produces a first token.
    """
    last_exc: Exception | None = None
    for provider in providers:
        agen = provider.stream(messages)
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:  # empty completion — treat as a failure, fail over
            last_exc = RuntimeError("provider produced no tokens")
            await agen.aclose()
            log.warning(
                "infer.failover", provider=type(provider).__name__, error="empty"
            )
            continue
        except Exception as exc:  # 429 or transport failure before first token
            last_exc = exc
            await agen.aclose()
            log.warning(
                "infer.failover",
                provider=type(provider).__name__,
                error=str(exc),
            )
            continue
        return _prepend(first, agen), provider
    raise ProvidersUnavailable("all inference providers failed") from last_exc


def get_providers() -> list[LLMProvider]:
    """Ordered provider chain for the query path: Gemini primary, OpenRouter fallback."""
    return [get_llm(), get_openrouter()]
