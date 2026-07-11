"""Gemini chat provider.

Streaming-first (CLAUDE.md): `stream` is the primitive; `generate` drains it.
You can collapse a stream into a string cheaply, but not split a string back
into one, so callers build on the stream even when they only need the string.
"""

from collections.abc import AsyncIterator, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from minddrill.config import get_settings

_CHAT_MODEL = "gemini-2.5-flash"

_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _to_lc_messages(messages: Sequence[dict[str, Any]]) -> list[Any]:
    return [_ROLE_TO_MESSAGE[m["role"]](content=m["content"]) for m in messages]


def _content_to_text(content: Any) -> str:
    """Flatten a chunk's content, which may be a string or typed content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return str(content)


def is_rate_limit_error(exc: Exception) -> bool:
    """True if a provider exception looks like an upstream 429 / quota hit."""
    if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
        return True
    name = type(exc).__name__.lower()
    if "resourceexhausted" in name or "ratelimit" in name:
        return True
    return "429" in str(exc)


class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = ChatGoogleGenerativeAI(
            model=_CHAT_MODEL, google_api_key=settings.gemini_api_key
        )

    async def stream(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        async for chunk in self._client.astream(_to_lc_messages(messages)):
            text = _content_to_text(chunk.content)
            if text:
                yield text

    async def generate(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        parts = [token async for token in self.stream(messages, **kwargs)]
        return "".join(parts)


@lru_cache
def get_llm() -> GeminiProvider:
    return GeminiProvider()
