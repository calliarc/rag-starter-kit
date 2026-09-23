"""FastAPI application: POST /ingest, POST /query, GET /collections, GET /healthz."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from rag import __version__
from rag.auth import Principal, get_principal, require_collection
from rag.config import Settings, get_settings
from rag.ingestion.chunking import ChunkingOptions
from rag.ingestion.loaders import DocumentParseError, UnsupportedFileType
from rag.models import Answer, CollectionInfo, IngestedDocument, validate_collection_name
from rag.providers.base import ProviderError
from rag.service import RAGService

log = logging.getLogger("rag")


class QueryRequest(BaseModel):
    collection: str
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool = True

    @field_validator("collection")
    @classmethod
    def _collection(cls, v: str) -> str:
        return validate_collection_name(v)


class IngestResponse(BaseModel):
    collection: str
    documents: list[IngestedDocument]


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: str
    llm_provider: str
    embedding_provider: str
    auth_enabled: bool


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = RAGService.from_settings(settings)
        if not settings.auth_enabled:
            log.warning("RAG_API_KEYS is empty: API is running WITHOUT authentication (dev mode)")
        yield
        app.state.service.store.close()

    app = FastAPI(
        title="RAG Starter Kit",
        version=__version__,
        description="Hybrid-search RAG API with citations and collection-level access control.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def svc(request: Request) -> RAGService:
        return request.app.state.service

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            vector_store=settings.vector_store,
            llm_provider=settings.llm_provider,
            embedding_provider=settings.embedding_provider,
            auth_enabled=settings.auth_enabled,
        )

    @app.get("/collections", response_model=CollectionsResponse, tags=["collections"])
    def collections(
        principal: Principal = Depends(get_principal), service: RAGService = Depends(svc)
    ) -> CollectionsResponse:
        visible = [c for c in service.list_collections() if principal.can_access(c.name)]
        return CollectionsResponse(collections=visible)

    @app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
    def ingest(
        collection: str = Form(...),
        files: list[UploadFile] = File(...),
        chunk_size: int | None = Form(default=None),
        chunk_overlap: int | None = Form(default=None),
        heading_aware: bool | None = Form(default=None),
        principal: Principal = Depends(get_principal),
        service: RAGService = Depends(svc),
    ) -> IngestResponse:
        try:
            validate_collection_name(collection)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        require_collection(principal, collection)

        options = None
        if chunk_size is not None or chunk_overlap is not None or heading_aware is not None:
            base = service.chunking
            try:
                options = ChunkingOptions(
                    chunk_size=chunk_size if chunk_size is not None else base.chunk_size,
                    chunk_overlap=chunk_overlap if chunk_overlap is not None else base.chunk_overlap,
                    heading_aware=heading_aware if heading_aware is not None else base.heading_aware,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

        limit = settings.max_upload_mb * 1024 * 1024
        results: list[IngestedDocument] = []
        for upload in files:
            data = upload.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(
                    413,
                    f"{upload.filename} exceeds {settings.max_upload_mb} MB",
                )
            try:
                results.append(
                    service.ingest_bytes(collection, upload.filename or "document", data, chunking=options)
                )
            except UnsupportedFileType as exc:
                raise HTTPException(415, str(exc)) from exc
            except DocumentParseError as exc:
                raise HTTPException(422, str(exc)) from exc
            except ProviderError as exc:
                log.exception("embedding provider failed")
                raise HTTPException(502, "embedding provider error") from exc
        return IngestResponse(collection=collection, documents=results)

    @app.post("/query", response_model=Answer, tags=["query"])
    def query(
        body: QueryRequest,
        principal: Principal = Depends(get_principal),
        service: RAGService = Depends(svc),
    ) -> Answer:
        require_collection(principal, body.collection)
        try:
            return service.answer(body.collection, body.question, body.top_k, rerank=body.rerank)
        except ProviderError as exc:
            log.exception("provider failed")
            raise HTTPException(502, "model provider error") from exc

    return app
