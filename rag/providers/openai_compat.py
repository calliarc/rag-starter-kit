"""HTTP providers for OpenAI-compatible APIs (OpenAI, Ollama, vLLM, LM Studio, ...) and Azure OpenAI.

Implemented directly on ``httpx`` to keep dependencies small and make them easy to test with a mock
transport.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from rag.providers.base import ChatMessage, EmbeddingProvider, LLMProvider, ProviderError

log = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class _HTTPBase:
    def __init__(
        self,
        *,
        headers: dict[str, str],
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(headers=headers, timeout=timeout, transport=transport)
        self._max_retries = max_retries

    def _post(self, url: str, payload: dict[str, Any], params: dict[str, str] | None = None) -> dict:
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(url, json=payload, params=params)
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise ProviderError(f"request to {url} failed: {exc}") from exc
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in _RETRY_STATUS or attempt >= self._max_retries:
                    raise ProviderError(f"{url} returned HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    delay = min(float(retry_after), 30.0)
            log.warning("provider request failed (attempt %d), retrying in %.1fs", attempt + 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise ProviderError("unreachable")  # pragma: no cover


def _parse_embeddings(data: dict) -> list[list[float]]:
    try:
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(map(float, d["embedding"])) for d in items]
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"unexpected embeddings response: {str(data)[:200]}") from exc


def _parse_chat(data: dict) -> str:
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"unexpected chat response: {str(data)[:200]}") from exc


def _messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAICompatibleEmbeddings(_HTTPBase, EmbeddingProvider):
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, **kw: Any) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(headers=headers, **kw)
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _parse_embeddings(self._post(self.url, {"model": self.model, "input": texts}))


class OpenAICompatibleLLM(_HTTPBase, LLMProvider):
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, **kw: Any) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(headers=headers, **kw)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": _messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return _parse_chat(self._post(self.url, payload))


class AzureOpenAIEmbeddings(_HTTPBase, EmbeddingProvider):
    name = "azure"

    def __init__(self, endpoint: str, deployment: str, api_key: str, api_version: str, **kw: Any) -> None:
        super().__init__(headers={"api-key": api_key}, **kw)
        self.url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/embeddings"
        self.params = {"api-version": api_version}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _parse_embeddings(self._post(self.url, {"input": texts}, params=self.params))


class AzureOpenAILLM(_HTTPBase, LLMProvider):
    name = "azure"

    def __init__(self, endpoint: str, deployment: str, api_key: str, api_version: str, **kw: Any) -> None:
        super().__init__(headers={"api-key": api_key}, **kw)
        self.url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        self.params = {"api-version": api_version}

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        payload = {"messages": _messages(messages), "temperature": temperature, "max_tokens": max_tokens}
        return _parse_chat(self._post(self.url, payload, params=self.params))
