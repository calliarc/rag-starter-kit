from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings
from rag.service import RAGService

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "examples" / "docs"
DATASET = ROOT / "evals" / "datasets" / "demo.jsonl"

API_KEYS = {"demo-key": ["demo"], "hr-key": ["hr"], "admin-key": ["*"]}


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        vector_store="memory",
        llm_provider="fake",
        embedding_provider="fake",
        chunk_size=500,
        chunk_overlap=80,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def service(settings: Settings) -> RAGService:
    return RAGService.from_settings(settings)


@pytest.fixture
def demo_service(service: RAGService) -> RAGService:
    service.ingest_path("demo", DOCS)
    return service


@pytest.fixture
def client() -> TestClient:
    app = create_app(make_settings(api_keys=API_KEYS))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def open_client() -> TestClient:
    app = create_app(make_settings())
    with TestClient(app) as c:
        yield c
