"""BM25 keyword index built on ``rank-bm25``."""

from __future__ import annotations

import math

from rank_bm25 import BM25Okapi

from rag.models import Chunk, ScoredChunk
from rag.text import tokenize


class _BM25(BM25Okapi):
    """BM25Okapi with the Lucene-style IDF ``log(1 + (N - n + 0.5) / (n + 0.5))``.

    The classic Okapi IDF is zero or negative for terms present in half or more of the documents,
    which makes keyword search useless on small collections; the Lucene variant is always positive.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [tokenize(c.search_text()) for c in chunks]
        # BM25Okapi cannot handle an empty corpus or all-empty documents.
        self._bm25 = _BM25(corpus) if chunks and any(corpus) else None

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in order[:k] if scores[i] > 0]
