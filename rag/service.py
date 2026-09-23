"""The RAG pipeline: ingest -> chunk -> embed -> store, and retrieve -> generate -> cite."""

from __future__ import annotations

import logging
from pathlib import Path, PurePath

from rag.config import Settings
from rag.ingestion.chunking import Chunker, ChunkingOptions, make_doc_id
from rag.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from rag.models import (
    Answer,
    Citation,
    CollectionInfo,
    IngestedDocument,
    ScoredChunk,
    validate_collection_name,
)
from rag.prompts import NO_ANSWER, build_messages, extract_citation_refs
from rag.providers import EmbeddingProvider, LLMProvider, build_embeddings, build_llm
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import Reranker, build_reranker
from rag.stores import VectorStore, build_store

log = logging.getLogger(__name__)

SNIPPET_CHARS = 300


def to_citation(ref: int, sc: ScoredChunk) -> Citation:
    c = sc.chunk
    snippet = c.text if len(c.text) <= SNIPPET_CHARS else c.text[:SNIPPET_CHARS].rstrip() + "..."
    return Citation(
        ref=ref,
        document=c.doc_name,
        doc_id=c.doc_id,
        chunk_id=c.id,
        chunk_index=c.index,
        page=c.page,
        heading=c.heading,
        score=round(sc.score, 6),
        snippet=snippet,
    )


class RAGService:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        store: VectorStore,
        chunking: ChunkingOptions | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 512,
        embedding_batch_size: int = 64,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.store = store
        self.chunking = chunking or ChunkingOptions()
        self.top_k = top_k
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.embedding_batch_size = embedding_batch_size
        self.retriever = HybridRetriever(
            store, embedder, candidate_k=candidate_k, rrf_k=rrf_k, reranker=reranker
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        reranker: Reranker | None = None,
    ) -> RAGService:
        return cls(
            embedder=embedder or build_embeddings(settings),
            llm=llm or build_llm(settings),
            store=store or build_store(settings),
            chunking=ChunkingOptions(settings.chunk_size, settings.chunk_overlap, settings.heading_aware),
            reranker=reranker or build_reranker(settings.reranker),
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
            rrf_k=settings.rrf_k,
            llm_temperature=settings.llm_temperature,
            llm_max_tokens=settings.llm_max_tokens,
            embedding_batch_size=settings.embedding_batch_size,
        )

    # ------------------------------------------------------------------ ingest
    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            out.extend(self.embedder.embed(texts[i : i + self.embedding_batch_size]))
        if len(out) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected number of vectors")
        return out

    def ingest_bytes(
        self, collection: str, name: str, data: bytes, *, chunking: ChunkingOptions | None = None
    ) -> IngestedDocument:
        validate_collection_name(collection)
        name = PurePath(name.replace("\\", "/")).name or "document"
        doc = load_document(name, data)
        chunks = Chunker(chunking or self.chunking).chunk(doc, collection)
        embeddings = self._embed_batched([c.search_text() for c in chunks]) if chunks else []
        doc_id = make_doc_id(collection, name)
        self.store.delete_document(collection, doc_id)  # re-ingest replaces the previous version
        if chunks:
            self.store.add(chunks, embeddings)
        log.info("ingested %s into %s: %d chunks", name, collection, len(chunks))
        pages = doc.metadata.get("pages")
        return IngestedDocument(document=name, doc_id=doc_id, chunks=len(chunks), pages=pages)

    def ingest_path(
        self, collection: str, path: str | Path, *, chunking: ChunkingOptions | None = None
    ) -> list[IngestedDocument]:
        path = Path(path)
        files = (
            sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if path.is_dir()
            else [path]
        )
        return [self.ingest_bytes(collection, f.name, f.read_bytes(), chunking=chunking) for f in files]

    # ------------------------------------------------------------------- query
    def retrieve(
        self, collection: str, question: str, top_k: int | None = None, *, rerank: bool = True
    ) -> list[ScoredChunk]:
        validate_collection_name(collection)
        return self.retriever.retrieve(collection, question, top_k or self.top_k, rerank=rerank)

    def answer(
        self, collection: str, question: str, top_k: int | None = None, *, rerank: bool = True
    ) -> Answer:
        contexts = self.retrieve(collection, question, top_k, rerank=rerank)
        return self.generate(question, contexts)

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Answer:
        """Generate a cited answer from already-retrieved contexts."""
        retrieved = [to_citation(i, sc) for i, sc in enumerate(contexts, start=1)]
        if not contexts:
            return Answer(answer=NO_ANSWER, citations=[], retrieved=[])
        text = self.llm.chat(
            build_messages(question, contexts),
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )
        refs = [r for r in extract_citation_refs(text) if 1 <= r <= len(contexts)]
        citations = [retrieved[r - 1] for r in refs]
        return Answer(answer=text, citations=citations, retrieved=retrieved)

    def list_collections(self) -> list[CollectionInfo]:
        return self.store.list_collections()
