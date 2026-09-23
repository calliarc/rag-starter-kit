import os
import uuid

import pytest

from rag.models import Chunk
from rag.stores import InMemoryStore, SQLiteStore


def _chunks(collection: str, doc: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            id=f"{doc}:{i}",
            collection=collection,
            doc_id=doc,
            doc_name=f"{doc}.md",
            index=i,
            text=t,
            page=i + 1,
            heading="H",
            metadata={"k": "v"},
        )
        for i, t in enumerate(texts)
    ]


def _pg_store():
    dsn = os.environ.get("RAG_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set RAG_TEST_DATABASE_URL to run pgvector tests")
    pytest.importorskip("psycopg_pool")
    from rag.stores.pgvector import PgVectorStore

    return PgVectorStore(dsn, table=f"test_{uuid.uuid4().hex[:8]}")


@pytest.fixture(params=["memory", "sqlite", "pgvector"])
def store(request, tmp_path):
    if request.param == "memory":
        s = InMemoryStore()
    elif request.param == "sqlite":
        s = SQLiteStore(str(tmp_path / "rag.db"))
    else:
        s = _pg_store()
    yield s
    if request.param == "pgvector":
        with s._pool.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {s.table}")
            conn.execute(f"DROP TABLE IF EXISTS {s.rev_table}")
    s.close()


def test_store_roundtrip(store):
    store.add(_chunks("a", "doc1", ["red apple", "green pear"]), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.add(_chunks("b", "doc2", ["blue sky"]), [[0.0, 0.0, 1.0]])

    res = store.vector_search("a", [0.9, 0.1, 0.0], 5)
    assert [r.chunk.text for r in res] == ["red apple", "green pear"]
    assert res[0].score > res[1].score
    assert res[0].chunk.page == 1 and res[0].chunk.heading == "H" and res[0].chunk.metadata == {"k": "v"}

    kw = store.keyword_search("a", "pear", 5)
    assert [r.chunk.text for r in kw] == ["green pear"]

    cols = {c.name: (c.documents, c.chunks) for c in store.list_collections()}
    assert cols == {"a": (1, 2), "b": (1, 1)}


def test_revision_changes_and_delete(store):
    r0 = store.revision("a")
    store.add(_chunks("a", "doc1", ["one", "two"]), [[1.0, 0.0], [0.0, 1.0]])
    r1 = store.revision("a")
    assert r1 != r0
    assert store.keyword_search("a", "two", 3)[0].chunk.text == "two"
    assert store.delete_document("a", "doc1") == 2
    assert store.revision("a") != r1
    assert store.vector_search("a", [1.0, 0.0], 3) == []
    assert store.keyword_search("a", "two", 3) == []  # BM25 cache invalidated
    assert store.delete_document("a", "missing") == 0


def test_dimension_mismatch_is_reported(store):
    store.add(_chunks("a", "doc1", ["one"]), [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        store.vector_search("a", [1.0, 0.0], 1)


def test_sqlite_persists_across_instances(tmp_path):
    path = str(tmp_path / "p.db")
    s1 = SQLiteStore(path)
    s1.add(_chunks("a", "doc1", ["persisted"]), [[1.0, 0.0]])
    s1.close()
    s2 = SQLiteStore(path)
    assert s2.iter_chunks("a")[0].text == "persisted"
    s2.close()
