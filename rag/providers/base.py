"""Provider interfaces. Implement these to plug in any LLM or embedding backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.models import ChatMessage

__all__ = ["ChatMessage", "EmbeddingProvider", "LLMProvider", "ProviderError"]


class EmbeddingProvider(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Return the assistant message content for a chat conversation."""


class ProviderError(RuntimeError):
    pass
