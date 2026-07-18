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
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings`, constructed once and cached."""
    return Settings()
