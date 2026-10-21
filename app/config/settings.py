import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load app/.env no matter which directory the scripts are started from.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    # Left unset by default: current reasoning models (GPT-5.x, Claude Opus 5.5)
    # reject a custom temperature.
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.5")
    )
    embedding_model: str = Field(default="text-embedding-3-small")


class AnthropicSettings(LLMSettings):
    """Anthropic-specific settings extending LLMSettings."""

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    default_model: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    )
    max_tokens: Optional[int] = 16000  # required by the Anthropic API


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    # TIMESCALE_SERVICE_URL is still accepted so older .env files keep working.
    service_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("DATABASE_URL")
        or os.getenv("TIMESCALE_SERVICE_URL")
    )


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1536
    # Metadata key whose values become DiskANN labels (fast filtered search).
    label_key: str = "category"
    # Postgres text search configuration for the BM25 keyword index.
    text_search_config: str = "english"


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
