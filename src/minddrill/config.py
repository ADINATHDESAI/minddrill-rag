"""Application settings, loaded once from the environment.

`get_settings()` is the single config seam — everything else (DB engine, auth,
providers, Alembic) reads config through it rather than touching `os.environ`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration.

    Values come from `.env` (or the real environment, which wins). Names are
    case-insensitive, so `DATABASE_URL` populates `database_url`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    redis_url: str
    jwt_secret: str
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    # OpenRouter periodically retires free-tier model slugs; re-check
    # https://openrouter.ai/models?max_price=0 if this starts 404ing.
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    embed_dim: int = 768
    log_level: str = "INFO"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_top_n: int = 5
    rerank_enabled: bool = True
    # Cross-encoder relevance floor for the top reranked chunk. Below it, the
    # retrieved context is treated as not supporting the question and the answer
    # is declined instead of generated.
    grounding_min_score: float = 0.0
    # Token budget for short-term chat memory: recent turns are trimmed to fit
    # this before building the prompt.
    memory_token_budget: int = 2000
    # Ceiling on the agent's model calls per request. On the limit the loop ends
    # and returns the best answer so far instead of spinning tools forever.
    agent_max_steps: int = 6
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""

    @property
    def langfuse_enabled(self) -> bool:
        """Tracing needs both keys; a bare base_url alone can't authenticate."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings`, constructed once and cached."""
    return Settings()
