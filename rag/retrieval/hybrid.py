"""Hybrid retrieval: BM25 keyword search + vector search, fused with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.models import ScoredChunk
from rag.retrieval.rerank import NoopReranker, Reranker

if TYPE_CHECKING:  # avoid an import cycle: stores.base -> retrieval.keyword
    from rag.providers.base import EmbeddingProvider
    from rag.stores.base import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int = 60, weights: list[float] | None = None
) -> list[ScoredChunk]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)), with 1-based ranks.

    The returned ``ScoredChunk`` records the rank of each chunk in the first two lists (vector, keyword).
    """
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    ranks: dict[str, list[int | None]] = {}
    for li, (lst, w) in enumerate(zip(ranked_lists, weights)):
        for rank, sc in enumerate(lst, start=1):
            cid = sc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            best.setdefault(cid, sc)
            ranks.setdefault(cid, [None] * len(ranked_lists))[li] = rank
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    fused = []
    for cid in order:
        r = ranks[cid]
        fused.append(
            ScoredChunk(
                chunk=best[cid].chunk,
                score=scores[cid],
                vector_rank=r[0] if len(r) > 0 else None,
                keyword_rank=r[1] if len(r) > 1 else None,
            )
        )
    return fused


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        mode: str = "hybrid",
    ) -> None:
        if mode not in {"hybrid", "vector", "keyword"}:
            raise ValueError("mode must be hybrid, vector or keyword")
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker or NoopReranker()
        self.mode = mode

    def retrieve(
        self, collection: str, query: str, top_k: int, *, mode: str | None = None, rerank: bool = True
    ) -> list[ScoredChunk]:
        mode = mode or self.mode
        n = max(self.candidate_k, top_k)
        vector: list[ScoredChunk] = []
        keyword: list[ScoredChunk] = []
        if mode in {"hybrid", "vector"}:
            vector = self.store.vector_search(collection, self.embedder.embed_query(query), n)
        if mode in {"hybrid", "keyword"}:
            keyword = self.store.keyword_search(collection, query, n)
        fused = reciprocal_rank_fusion([vector, keyword], k=self.rrf_k)
        if rerank:
            fused = self.reranker.rerank(query, fused)
        return fused[:top_k]
