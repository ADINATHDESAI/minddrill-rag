"""Chat models for the agent loop.

`create_agent` needs a LangChain `BaseChatModel` that supports `bind_tools`, so
the agent reaches for the model adapters directly rather than the text-stream
`LLMProvider`. Gemini is primary; OpenRouter is the fallback, applied inside the
loop via `ModelFallbackMiddleware`. Both are FastAPI dependencies, so the model
is a swappable seam.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from minddrill.config import get_settings

_CHAT_MODEL = "gemini-2.5-flash"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache
def get_agent_model() -> BaseChatModel:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=_CHAT_MODEL, google_api_key=settings.gemini_api_key
    )


@lru_cache
def get_fallback_model() -> BaseChatModel | None:
    settings = get_settings()
    if not settings.openrouter_api_key:
        return None
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=_OPENROUTER_BASE_URL,
    )
