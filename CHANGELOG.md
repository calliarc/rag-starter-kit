# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-23

First working release.

### Added

- Document ingestion for PDF, DOCX, HTML and Markdown (plus plain text)
- Configurable chunking (size, overlap, heading-aware for Markdown/HTML/DOCX) and pluggable embeddings
- Hybrid search: BM25 keyword + vector search fused with reciprocal rank fusion, with a re-ranking hook
- Answers with source citations (document, chunk, page, section)
- Evaluation suite for retrieval quality and answer accuracy: hit@k, MRR, recall@k, groundedness, citation precision, answer F1, optional LLM judge
- Access control per document collection (API key → allowed collections)
- Docker Compose for local development (API + PostgreSQL/pgvector, optional Ollama)
- Offline "fake" providers so the whole stack, tests and evals run without any API key

[Unreleased]: https://github.com/calliarc/rag-starter-kit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/calliarc/rag-starter-kit/releases/tag/v0.1.0
