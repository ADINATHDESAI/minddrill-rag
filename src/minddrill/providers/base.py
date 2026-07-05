"""LLM provider interface.

Inference is streaming-first: `stream` is the primitive; `generate` is defined as
the string wrapper over it. Concrete providers (Gemini free → OpenRouter fallback)
land in the inference slice — this module is the interface only.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """A chat completion provider.

    `messages` is a sequence of role/content dicts. Implementations must make the
    async token stream the primitive and build `generate` on top of it.
    """

    def stream(
        self, messages: Sequence[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Yield response tokens as they arrive. The primitive."""
        ...

    async def generate(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        """Return the full response as a string by consuming `stream`."""
        ...
