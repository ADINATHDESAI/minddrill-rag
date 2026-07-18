"""OpenRouter chat provider — the fallback when Gemini is rate-limited or down.

Same streaming-first contract as `GeminiProvider`: `stream` is the primitive and
`generate` drains it. OpenRouter speaks the OpenAI wire protocol, so we use the
`openai` async client pointed at their base URL.
"""

from collections.abc import AsyncIterator, Sequence
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from minddrill.config import get_settings

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            base_url=_BASE_URL, api_key=settings.openrouter_api_key or "missing"
        )
        self._model = settings.openrouter_model
        self.last_usage: dict[str, int] | None = None

    async def stream(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        self.last_usage = None
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage is not None:  # final usage-only frame
                self.last_usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def generate(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        parts = [token async for token in self.stream(messages, **kwargs)]
        return "".join(parts)


@lru_cache
def get_openrouter() -> OpenRouterProvider:
    return OpenRouterProvider()
