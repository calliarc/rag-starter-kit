"""In-process store. Fast and dependency-free; data is lost when the process exits."""

from __future__ import annotations

import threading
from collections import defaultdict

import numpy as np

from rag.models import Chunk, CollectionInfo, ScoredChunk
from rag.stores.base import VectorStore, cosine_top_k


class InMemoryStore(VectorStore):
    name = "memory"

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._chunks: dict[str, dict[str, tuple[Chunk, np.ndarray]]] = defaultdict(dict)
        self._revisions: dict[str, int] = defaultdict(int)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        with self._lock:
            for chunk, emb in zip(chunks, embeddings):
                self._chunks[chunk.collection][chunk.id] = (chunk, np.asarray(emb, dtype=np.float32))
                self._revisions[chunk.collection] += 1

    def delete_document(self, collection: str, doc_id: str) -> int:
        with self._lock:
            items = self._chunks.get(collection, {})
            ids = [cid for cid, (c, _) in items.items() if c.doc_id == doc_id]
            for cid in ids:
                del items[cid]
            if ids:
                self._revisions[collection] += 1
            return len(ids)

    def vector_search(self, collection: str, embedding: list[float], k: int) -> list[ScoredChunk]:
        with self._lock:
            items = list(self._chunks.get(collection, {}).values())
        if not items:
            return []
        matrix = np.stack([e for _, e in items])
        return [ScoredChunk(chunk=items[i][0], score=s) for i, s in cosine_top_k(matrix, embedding, k)]

    def iter_chunks(self, collection: str) -> list[Chunk]:
        with self._lock:
            chunks = [c for c, _ in self._chunks.get(collection, {}).values()]
        return sorted(chunks, key=lambda c: (c.doc_name, c.index))

    def revision(self, collection: str) -> str:
        with self._lock:
            return str(self._revisions.get(collection, 0))

    def list_collections(self) -> list[CollectionInfo]:
        with self._lock:
            out = []
            for name, items in sorted(self._chunks.items()):
                if items:
                    docs = {c.doc_id for c, _ in items.values()}
                    out.append(CollectionInfo(name=name, documents=len(docs), chunks=len(items)))
            return out
