"""PostgreSQL + pgvector store (requires ``pip install 'rag-starter-kit[postgres]'``).

The chunks table is created on first write with the dimension of the first embedding, plus an HNSW
index for cosine distance. Keyword search uses the in-process BM25 index from the base class, rebuilt
when the collection revision changes.
"""

from __future__ import annotations

import json
import threading

from rag.models import Chunk, CollectionInfo, ScoredChunk
from rag.stores.base import VectorStore


def _vec(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in embedding) + "]"


class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, dsn: str, table: str = "rag_chunks", pool_size: int = 5) -> None:
        super().__init__()
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pgvector store needs: pip install 'rag-starter-kit[postgres]'") from exc
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        self.table = table
        self.rev_table = f"{table}_revisions"
        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=pool_size, open=True, configure=self._configure_connection
        )
        self._dim: int | None = None
        self._schema_lock = threading.Lock()
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.rev_table} "
                "(collection TEXT PRIMARY KEY, rev BIGINT NOT NULL)"
            )
            self._dim = self._existing_dim(conn)

    @staticmethod
    def _configure_connection(conn) -> None:
        # pgvector >= 0.8: keep scanning the HNSW index until enough rows match the collection filter.
        try:
            conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
            conn.commit()
        except Exception:  # older pgvector: harmless, fall back to default behaviour
            conn.rollback()

    def _ready(self) -> bool:
        """True once the chunks table exists (it may have been created by another process)."""
        if self._dim is None:
            with self._pool.connection() as conn:
                self._dim = self._existing_dim(conn)
        return self._dim is not None

    def _existing_dim(self, conn) -> int | None:
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            (self.table,),
        ).fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None

    def _ensure_table(self, dim: int) -> None:
        self._ready()
        with self._schema_lock:
            if self._dim is not None:
                if self._dim != dim:
                    raise ValueError(
                        f"embedding dimension {dim} does not match table {self.table} ({self._dim}); "
                        "use a new RAG_PG_TABLE or drop the table after changing embedding models"
                    )
                return
            with self._pool.connection() as conn:
                conn.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.table} (
                        id TEXT PRIMARY KEY,
                        collection TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        doc_name TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        page INTEGER,
                        heading TEXT,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({int(dim)}) NOT NULL
                    )"""
                )
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_coll_idx ON {self.table}(collection, doc_id)"
                )
                if dim <= 2000:  # pgvector HNSW limit for the `vector` type; larger dims use exact scan
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw_idx ON {self.table} "
                        "USING hnsw (embedding vector_cosine_ops)"
                    )
                self._dim = self._existing_dim(conn) or dim

    def _bump(self, conn, collection: str) -> None:
        conn.execute(
            f"INSERT INTO {self.rev_table}(collection, rev) VALUES (%s, 1) "
            f"ON CONFLICT (collection) DO UPDATE SET rev = {self.rev_table}.rev + 1",
            (collection,),
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._ensure_table(len(embeddings[0]))
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
                _vec(e),
            )
            for c, e in zip(chunks, embeddings)
        ]
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table}
                    (id, collection, doc_id, doc_name, idx, text, page, heading, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                      text = EXCLUDED.text, page = EXCLUDED.page, heading = EXCLUDED.heading,
                      metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding,
                      doc_name = EXCLUDED.doc_name, idx = EXCLUDED.idx""",
                rows,
            )
            for coll in {c.collection for c in chunks}:
                self._bump(conn, coll)

    def delete_document(self, collection: str, doc_id: str) -> int:
        if not self._ready():
            return 0
        with self._pool.connection() as conn:
            cur = conn.execute(
                f"DELETE FROM {self.table} WHERE collection = %s AND doc_id = %s", (collection, doc_id)
            )
            if cur.rowcount:
                self._bump(conn, collection)
            return cur.rowcount

    _SELECT = "id, collection, doc_id, doc_name, idx, text, page, heading, metadata"

    @staticmethod
    def _to_chunk(r) -> Chunk:
        meta = r[8] if isinstance(r[8], dict) else json.loads(r[8] or "{}")
        return Chunk(
            id=r[0],
            collection=r[1],
            doc_id=r[2],
            doc_name=r[3],
            index=r[4],
            text=r[5],
            page=r[6],
            heading=r[7],
            metadata=meta,
        )

    def vector_search(self, collection: str, embedding: list[float], k: int) -> list[ScoredChunk]:
        if not self._ready():
            return []
        if len(embedding) != self._dim:
            raise ValueError(f"query embedding dimension {len(embedding)} != table dimension {self._dim}")
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {self._SELECT}, 1 - (embedding <=> %s::vector) AS score
                    FROM {self.table} WHERE collection = %s
                    ORDER BY embedding <=> %s::vector LIMIT %s""",
                (_vec(embedding), collection, _vec(embedding), k),
            ).fetchall()
        results = [ScoredChunk(chunk=self._to_chunk(r), score=float(r[9])) for r in rows]
        return sorted(results, key=lambda r: -r.score)  # iterative scans may be slightly out of order

    def iter_chunks(self, collection: str) -> list[Chunk]:
        if not self._ready():
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT} FROM {self.table} WHERE collection = %s ORDER BY doc_name, idx",
                (collection,),
            ).fetchall()
        return [self._to_chunk(r) for r in rows]

    def revision(self, collection: str) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT rev FROM {self.rev_table} WHERE collection = %s", (collection,)
            ).fetchone()
        return str(row[0] if row else 0)

    def list_collections(self) -> list[CollectionInfo]:
        if not self._ready():
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT collection, COUNT(DISTINCT doc_id), COUNT(*) FROM {self.table} "
                "GROUP BY collection ORDER BY collection"
            ).fetchall()
        return [CollectionInfo(name=r[0], documents=r[1], chunks=r[2]) for r in rows]

    def close(self) -> None:
        self._pool.close()
