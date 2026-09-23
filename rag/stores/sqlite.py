"""SQLite store for single-node local use. Embeddings are stored as float32 blobs and searched with numpy.

Good for up to tens of thousands of chunks per collection. Use pgvector for anything larger or shared.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from rag.models import Chunk, CollectionInfo, ScoredChunk
from rag.stores.base import VectorStore, cosine_top_k

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_name TEXT NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    heading TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(collection, doc_id);
CREATE TABLE IF NOT EXISTS revisions (collection TEXT PRIMARY KEY, rev INTEGER NOT NULL);
"""

_COLS = "id, collection, doc_id, doc_name, idx, text, page, heading, metadata"


def _row_to_chunk(row: tuple) -> Chunk:
    return Chunk(
        id=row[0],
        collection=row[1],
        doc_id=row[2],
        doc_name=row[3],
        index=row[4],
        text=row[5],
        page=row[6],
        heading=row[7],
        metadata=json.loads(row[8] or "{}"),
    )


class SQLiteStore(VectorStore):
    name = "sqlite"

    def __init__(self, path: str = "data/rag.db") -> None:
        super().__init__()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._matrix_cache: dict[str, tuple[str, list[Chunk], np.ndarray]] = {}

    def _bump(self, collection: str) -> None:
        self._conn.execute(
            "INSERT INTO revisions(collection, rev) VALUES (?, 1) "
            "ON CONFLICT(collection) DO UPDATE SET rev = rev + 1",
            (collection,),
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        rows = [
            (
                c.id,
                c.collection,
                c.doc_id,
                c.doc_name,
                c.index,
                c.text,
                c.page,
                c.heading,
                json.dumps(c.metadata),
                np.asarray(e, dtype=np.float32).tobytes(),
            )
            for c, e in zip(chunks, embeddings)
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO chunks({_COLS}, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            for coll in {c.collection for c in chunks}:
                self._bump(coll)

    def delete_document(self, collection: str, doc_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE collection = ? AND doc_id = ?", (collection, doc_id)
            )
            if cur.rowcount:
                self._bump(collection)
            return cur.rowcount

    def revision(self, collection: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT rev FROM revisions WHERE collection = ?", (collection,)
            ).fetchone()
        return str(row[0] if row else 0)

    def _load(self, collection: str) -> tuple[list[Chunk], np.ndarray]:
        rev = self.revision(collection)
        cached = self._matrix_cache.get(collection)
        if cached and cached[0] == rev:
            return cached[1], cached[2]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS}, embedding FROM chunks WHERE collection = ? ORDER BY doc_name, idx",
                (collection,),
            ).fetchall()
        chunks = [_row_to_chunk(r) for r in rows]
        matrix = (
            np.stack([np.frombuffer(r[9], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        self._matrix_cache[collection] = (rev, chunks, matrix)
        return chunks, matrix

    def vector_search(self, collection: str, embedding: list[float], k: int) -> list[ScoredChunk]:
        chunks, matrix = self._load(collection)
        return [ScoredChunk(chunk=chunks[i], score=s) for i, s in cosine_top_k(matrix, embedding, k)]

    def iter_chunks(self, collection: str) -> list[Chunk]:
        return list(self._load(collection)[0])

    def list_collections(self) -> list[CollectionInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT collection, COUNT(DISTINCT doc_id), COUNT(*) FROM chunks "
                "GROUP BY collection ORDER BY collection"
            ).fetchall()
        return [CollectionInfo(name=r[0], documents=r[1], chunks=r[2]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
