"""Provider factory: build LLM and embedding providers from settings."""

from __future__ import annotations

from rag.config import Settings
from rag.providers.base import ChatMessage, EmbeddingProvider, LLMProvider, ProviderError
from rag.providers.fake import ExtractiveLLM, HashingEmbeddings
from rag.providers.openai_compat import (
    AzureOpenAIEmbeddings,
    AzureOpenAILLM,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
)

__all__ = [
    "ChatMessage",
    "EmbeddingProvider",
    "LLMProvider",
    "ProviderError",
    "build_embeddings",
    "build_llm",
]


def _secret(value) -> str | None:
    return value.get_secret_value() if value is not None else None


def _require(settings: Settings, *names: str) -> None:
    missing = [f"RAG_{n.upper()}" for n in names if not getattr(settings, n)]
    if missing:
        raise ValueError(f"missing required settings: {', '.join(missing)}")


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    kind = settings.embedding_provider
    if kind == "fake":
        return HashingEmbeddings(dim=settings.fake_embedding_dim)
    if kind == "openai":
        return OpenAICompatibleEmbeddings(
            base_url=settings.openai_base_url,
            model=settings.openai_embedding_model,
            api_key=_secret(settings.openai_api_key),
            timeout=settings.request_timeout_s,
        )
    if kind == "azure":
        _require(
            settings, "azure_openai_endpoint", "azure_openai_api_key", "azure_openai_embedding_deployment"
        )
        return AzureOpenAIEmbeddings(
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_embedding_deployment,
            api_key=_secret(settings.azure_openai_api_key),
            api_version=settings.azure_openai_api_version,
            timeout=settings.request_timeout_s,
        )
    raise ValueError(f"unknown embedding provider: {kind}")


def build_llm(settings: Settings) -> LLMProvider:
    kind = settings.llm_provider
    if kind == "fake":
        return ExtractiveLLM()
    if kind == "openai":
        return OpenAICompatibleLLM(
            base_url=settings.openai_base_url,
            model=settings.openai_chat_model,
            api_key=_secret(settings.openai_api_key),
            timeout=settings.request_timeout_s,
        )
    if kind == "azure":
        _require(settings, "azure_openai_endpoint", "azure_openai_api_key", "azure_openai_chat_deployment")
        return AzureOpenAILLM(
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_chat_deployment,
            api_key=_secret(settings.azure_openai_api_key),
            api_version=settings.azure_openai_api_version,
            timeout=settings.request_timeout_s,
        )
    raise ValueError(f"unknown llm provider: {kind}")
