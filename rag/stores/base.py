"""Vector store interface.

A store persists chunks with their embeddings and supports vector search per collection. Keyword (BM25)
search is provided by the base class from ``iter_chunks`` and cached per collection revision; stores may
override ``keyword_search`` with a native implementation.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import numpy as np

from rag.models import Chunk, CollectionInfo, ScoredChunk
from rag.retrieval.keyword import BM25Index


class VectorStore(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self._bm25_cache: dict[str, tuple[str, BM25Index]] = {}
        self._bm25_lock = threading.Lock()

    # --- writes ----------------------------------------------------------
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def delete_document(self, collection: str, doc_id: str) -> int:
        """Delete all chunks of a document. Returns the number of chunks removed."""

    # --- reads -----------------------------------------------------------
    @abstractmethod
    def vector_search(self, collection: str, embedding: list[float], k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    def iter_chunks(self, collection: str) -> list[Chunk]: ...

    @abstractmethod
    def revision(self, collection: str) -> str:
        """Opaque token that changes whenever the collection's contents change."""

    @abstractmethod
    def list_collections(self) -> list[CollectionInfo]: ...

    def keyword_search(self, collection: str, query: str, k: int) -> list[ScoredChunk]:
        rev = self.revision(collection)
        with self._bm25_lock:
            cached = self._bm25_cache.get(collection)
            if cached is None or cached[0] != rev:
                cached = (rev, BM25Index(self.iter_chunks(collection)))
                self._bm25_cache[collection] = cached
        return cached[1].search(query, k)

    def close(self) -> None:  # noqa: B027 - optional hook, most stores need no cleanup
        """Release connections or file handles."""


def cosine_top_k(matrix: np.ndarray, query: list[float], k: int) -> list[tuple[int, float]]:
    """Return (row, cosine similarity) for the top-k rows of ``matrix``."""
    if matrix.size == 0:
        return []
    q = np.asarray(query, dtype=np.float32)
    if q.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"query embedding has dimension {q.shape[0]} but the collection uses {matrix.shape[1]}; "
            "re-ingest after changing embedding models"
        )
    q_norm = np.linalg.norm(q) or 1.0
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    sims = (matrix @ q) / (norms * q_norm)
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return [(int(i), float(sims[i])) for i in idx]
