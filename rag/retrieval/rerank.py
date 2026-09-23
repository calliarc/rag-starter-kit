"""Re-ranking hook.

Implement ``Reranker`` to plug in a cross-encoder, Cohere/Azure re-rank API or an LLM judge, and pass it to
``RAGService(reranker=...)``. The retriever calls it on the fused candidate list before truncating to top_k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.models import ScoredChunk
from rag.text import tokenize


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return candidates re-ordered (and optionally re-scored) by relevance to ``query``."""


class NoopReranker(Reranker):
    name = "none"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return candidates


class OverlapReranker(Reranker):
    """Cheap lexical re-ranker: blends fused score with query-term coverage of each chunk."""

    name = "overlap"

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        q = set(tokenize(query))
        if not q or not candidates:
            return candidates
        top = max(c.score for c in candidates) or 1.0
        rescored = []
        for c in candidates:
            coverage = len(q & set(tokenize(c.chunk.search_text()))) / len(q)
            score = (1 - self.weight) * (c.score / top) + self.weight * coverage
            rescored.append(c.model_copy(update={"score": score}))
        return sorted(rescored, key=lambda c: -c.score)


def build_reranker(name: str) -> Reranker:
    if name == "none":
        return NoopReranker()
    if name == "overlap":
        return OverlapReranker()
    raise ValueError(f"unknown reranker: {name}")
