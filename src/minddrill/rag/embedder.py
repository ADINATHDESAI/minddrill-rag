"""Text embedding behind an interface.

`Embedder` is the seam; `GeminiEmbedder` is the production implementation
(Gemini `text-embedding-004`, 768-dim). LangChain's embeddings client is the
adapter *inside* our interface, not the interface itself. `get_embedder` is the
FastAPI dependency so tests can substitute a deterministic fake.
"""

from functools import lru_cache
from typing import Protocol, runtime_checkable

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from minddrill.config import get_settings

_EMBED_MODEL = "models/gemini-embedding-001"


@runtime_checkable
class Embedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...


class GeminiEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._dim = settings.embed_dim
        self._client = GoogleGenerativeAIEmbeddings(
            model=_EMBED_MODEL,
            google_api_key=settings.gemini_api_key,
            output_dimensionality=self._dim,
        )

    def _check_dim(self, vec: list[float]) -> list[float]:
        if len(vec) != self._dim:
            raise ValueError(
                f"embedding dim {len(vec)} != configured {self._dim}"
            )
        return vec

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vecs = await self._client.aembed_documents(texts)
        return [self._check_dim(v) for v in vecs]

    async def embed_query(self, text: str) -> list[float]:
        return self._check_dim(await self._client.aembed_query(text))


@lru_cache
def get_embedder() -> Embedder:
    return GeminiEmbedder()
