import pytest
from pydantic import ValidationError

from rag.auth import resolve_principal
from rag.config import Settings


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS", '{"k1": ["demo"], "k2": ["*"]}')
    monkeypatch.setenv("RAG_VECTOR_STORE", "memory")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "300")
    s = Settings(_env_file=None)
    assert s.api_keys == {"k1": ["demo"], "k2": ["*"]}
    assert s.auth_enabled and s.chunk_size == 300


def test_invalid_settings():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vector_store="pgvector", database_url=None)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, pg_table="chunks; drop table x")


def test_resolve_principal():
    keys = {"k1": ["demo"], "k2": ["*"]}
    assert resolve_principal(keys, None) is None
    assert resolve_principal(keys, "nope") is None
    p = resolve_principal(keys, "k1")
    assert p.can_access("demo") and not p.can_access("hr")
    assert resolve_principal(keys, "k2").can_access("anything")
    assert resolve_principal({}, None).is_admin  # open dev mode
