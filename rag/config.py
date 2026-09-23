"""Application settings, loaded from environment variables (prefix ``RAG_``) and an optional ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["fake", "openai", "azure"]
StoreName = Literal["memory", "sqlite", "pgvector"]
RerankerName = Literal["none", "overlap"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Providers -------------------------------------------------------
    llm_provider: ProviderName = "fake"
    embedding_provider: ProviderName = "fake"

    # OpenAI-compatible endpoints (OpenAI, Ollama, vLLM, LM Studio, ...)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # Shared generation / embedding options
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    request_timeout_s: float = 60.0
    embedding_batch_size: int = Field(default=64, ge=1)
    fake_embedding_dim: int = Field(default=384, ge=8)

    # --- Vector store ----------------------------------------------------
    vector_store: StoreName = "sqlite"
    sqlite_path: str = "data/rag.db"
    database_url: str | None = None  # postgresql://user:pass@host:5432/db
    pg_table: str = "rag_chunks"

    # --- Chunking --------------------------------------------------------
    chunk_size: int = Field(default=800, ge=50, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=120, ge=0)
    heading_aware: bool = True

    # --- Retrieval -------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    reranker: RerankerName = "none"

    # --- API / security --------------------------------------------------
    # JSON object mapping API key -> list of allowed collections ("*" = all).
    # Example: {"demo-key": ["demo"], "admin-key": ["*"]}
    # When empty, the API runs in open (no-auth) mode, intended for local development only.
    api_keys: dict[str, list[str]] = Field(default_factory=dict)
    max_upload_mb: int = Field(default=25, ge=1)
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("pg_table")
    @classmethod
    def _safe_table(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or not v[0].isalpha():
            raise ValueError("pg_table must be alphanumeric/underscore and start with a letter")
        return v

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.vector_store == "pgvector" and not self.database_url:
            raise ValueError("RAG_DATABASE_URL is required when RAG_VECTOR_STORE=pgvector")
        return self

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
