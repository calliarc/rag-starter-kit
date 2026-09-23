from __future__ import annotations

from rag.config import Settings
from rag.stores.base import VectorStore
from rag.stores.memory import InMemoryStore
from rag.stores.sqlite import SQLiteStore

__all__ = ["InMemoryStore", "SQLiteStore", "VectorStore", "build_store"]


def build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryStore()
    if settings.vector_store == "sqlite":
        return SQLiteStore(settings.sqlite_path)
    if settings.vector_store == "pgvector":
        from rag.stores.pgvector import PgVectorStore

        return PgVectorStore(settings.database_url or "", table=settings.pg_table)
    raise ValueError(f"unknown vector store: {settings.vector_store}")
