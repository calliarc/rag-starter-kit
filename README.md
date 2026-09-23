# RAG Starter Kit

Production-ready retrieval-augmented generation (RAG) template for chatting with company documents, with evals included.

[![CI](https://github.com/calliarc/rag-starter-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/calliarc/rag-starter-kit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/calliarc/rag-starter-kit?include_prereleases&sort=semver)](https://github.com/calliarc/rag-starter-kit/releases)
[![Built by CalliArc](https://img.shields.io/badge/built%20by-CalliArc-0a66c2)](https://www.calliarc.com/)

> **Status:** v0.1.0, the first working release. The API and settings may still change before 1.0.

## Features

- Document ingestion for PDF, DOCX, HTML and Markdown (plus plain text)
- Configurable chunking (size, overlap, heading-aware for Markdown/HTML/DOCX) and pluggable embeddings
- Hybrid search: BM25 keyword + vector search fused with reciprocal rank fusion, with a re-ranking hook
- Answers with source citations (document, chunk, page, section)
- Evaluation suite for retrieval quality and answer accuracy: hit@k, MRR, recall@k, groundedness, citation precision, answer F1, optional LLM judge
- Access control per document collection (API key → allowed collections)
- Docker Compose for local development (API + PostgreSQL/pgvector, optional Ollama)
- Offline "fake" providers so the whole stack, tests and evals run without any API key

## Tech stack

- Python 3.10+
- FastAPI, pydantic-settings
- Azure OpenAI, any OpenAI-compatible API (OpenAI, Ollama, vLLM, LM Studio) or offline fake providers
- pgvector (PostgreSQL), SQLite or in-memory vector stores
- rank-bm25, pypdf, python-docx, BeautifulSoup
- Docker

## Getting started

### 1. Local, fully offline (fake providers)

No API keys, no database. Uses deterministic hashing embeddings, an extractive "LLM" and a SQLite store in `data/`.

```bash
git clone https://github.com/calliarc/rag-starter-kit.git
cd rag-starter-kit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # defaults: fake providers, SQLite, API key "change-me-demo-key"

python -m rag ingest examples/docs --collection demo
python -m rag query "How many vacation days can I carry over?" --collection demo
python -m rag serve             # http://127.0.0.1:8000/docs
```

The fake providers are for demos, tests and CI. They show the whole pipeline working, but answers are extracted sentences, not generated text.

### 2. Azure OpenAI

Create a chat deployment (for example `gpt-4o-mini`) and an embedding deployment (for example `text-embedding-3-small`), then set in `.env`:

```dotenv
RAG_LLM_PROVIDER=azure
RAG_EMBEDDING_PROVIDER=azure
RAG_AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
RAG_AZURE_OPENAI_API_KEY=<key>
RAG_AZURE_OPENAI_API_VERSION=2024-10-21
RAG_AZURE_OPENAI_CHAT_DEPLOYMENT=<chat-deployment-name>
RAG_AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>
```

Re-ingest after you switch embedding providers. Vectors from different models cannot be compared.

### 3. Ollama (local open-source models)

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

```dotenv
RAG_LLM_PROVIDER=openai
RAG_EMBEDDING_PROVIDER=openai
RAG_OPENAI_BASE_URL=http://localhost:11434/v1
RAG_OPENAI_CHAT_MODEL=llama3.1:8b
RAG_OPENAI_EMBEDDING_MODEL=nomic-embed-text
```

vLLM, LM Studio or OpenAI itself work the same way: point `RAG_OPENAI_BASE_URL` at the server and set `RAG_OPENAI_API_KEY` if it needs one.

### 4. Docker Compose (API + pgvector)

```bash
cp .env.example .env            # choose providers here; the store is set to pgvector by compose
docker compose up --build
# optional local models:  docker compose --profile ollama up --build
#   then set RAG_OPENAI_BASE_URL=http://ollama:11434/v1 in .env
```

The API listens on `http://127.0.0.1:8000`, Postgres on `127.0.0.1:5432` (user/password `rag`, local development only).

## API

All endpoints except `/healthz` require an `X-API-Key` header when `RAG_API_KEYS` is set. Interactive docs are at `/docs`.

| Method | Path           | Description                                                                       |
| ------ | -------------- | --------------------------------------------------------------------------------- |
| GET    | `/healthz`     | Liveness, version and active providers                                            |
| GET    | `/collections` | Collections visible to the API key, with document and chunk counts                |
| POST   | `/ingest`      | Multipart upload: `collection`, one or more `files`, optional `chunk_size`, `chunk_overlap`, `heading_aware` |
| POST   | `/query`       | JSON `{collection, question, top_k?, rerank?}` → answer with citations            |

```bash
KEY=change-me-demo-key

curl -s http://localhost:8000/healthz

curl -s -H "X-API-Key: $KEY" http://localhost:8000/ingest \
  -F collection=demo \
  -F files=@examples/docs/security-policy.pdf \
  -F files=@examples/docs/employee-handbook.md

curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" http://localhost:8000/query \
  -d '{"collection": "demo", "question": "What is the minimum password length?", "top_k": 3}'
```

Response (fake providers, shortened):

```json
{
  "answer": "Passwords and authentication All accounts must use a password of at least 14 characters. [1]",
  "citations": [
    {
      "ref": 1,
      "document": "security-policy.pdf",
      "doc_id": "b26891556ba8c0c9",
      "chunk_id": "b26891556ba8c0c9:0",
      "chunk_index": 0,
      "page": 1,
      "heading": null,
      "score": 0.032787,
      "snippet": "Fernhill Instruments Information Security Policy Sample content for ..."
    }
  ],
  "retrieved": ["... every chunk passed to the LLM, same shape as citations ..."]
}
```

`citations` holds the sources the answer actually cites (`[n]` markers). `retrieved` holds every chunk that was given to the model.

### Access control

`RAG_API_KEYS` is a JSON object that maps each API key to the collections it may read and write:

```dotenv
RAG_API_KEYS={"sales-team-key": ["sales", "public"], "hr-key": ["hr"], "admin-key": ["*"]}
```

Keys are compared in constant time. A key gets `401` if it is missing or unknown and `403` for collections outside its list. `/collections` only lists what the key can access. If `RAG_API_KEYS` is empty, the API runs **without authentication** and logs a warning. Use that for local development only.

### CLI

```bash
python -m rag ingest <file-or-dir> -c <collection>
python -m rag query "<question>" -c <collection> [--top-k 5] [--json]
python -m rag collections
python -m rag serve [--host 0.0.0.0] [--port 8000]
```

## Evals

Datasets are JSONL files, with one example per line:

```json
{"id": "log-retention", "collection": "demo", "question": "How long are security logs retained?",
 "expected_sources": ["security-policy.pdf#p2"], "expected_answer": "Security logs are retained for 400 days."}
```

`expected_sources` lists document names. Add `#p<N>` to require a specific PDF page. `expected_answer` is optional.

```bash
# self-contained: ingest docs into a fresh in-memory store, then evaluate
python -m rag.evals --dataset evals/datasets/demo.jsonl --docs examples/docs

# compare retrieval modes, write a JSON report, fail CI below thresholds
python -m rag.evals -d evals/datasets/demo.jsonl --docs examples/docs --mode vector -o evals/reports/vector.json
python -m rag.evals -d evals/datasets/demo.jsonl --docs examples/docs --min-hit-rate 0.9 --min-mrr 0.8

# evaluate your deployed index (the store from your settings) and add an LLM judge
python -m rag.evals -d my-dataset.jsonl --llm-judge
```

| Metric               | Meaning                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `hit@k`              | Share of questions where an expected source appears in the top k chunks                                   |
| `mrr`                | Mean reciprocal rank of the first relevant chunk                                                          |
| `recall@k`           | Share of expected sources found in the top k                                                              |
| `groundedness`       | Share of answer sentences that cite a retrieved source *and* whose content words (≥ 60 %) appear in that source |
| `citation_coverage`  | Share of answer sentences with a valid citation                                                           |
| `citation_precision` | Share of cited chunks that come from an expected source                                                   |
| `answer_f1`, `answer_recall` | Token overlap with `expected_answer`                                                              |
| `answer_rate`        | Share of questions that were answered rather than refused                                                 |
| `llm_judge`          | Optional (`--llm-judge`): groundedness score from 0 to 1 given by the configured LLM                      |

Results on the bundled demo corpus with the offline fake providers (k=5, hybrid): hit@5 1.00, MRR 1.00, groundedness 1.00, answer F1 0.66. The demo corpus is small and easy, so use it as a smoke test and build a dataset from your own documents for real measurements.

## Architecture

```
                 ┌─────────── POST /ingest ───────────┐
 files ──► loaders (pdf/docx/html/md) ──► Chunker ──► EmbeddingProvider ──► VectorStore
                                         size/overlap   fake | openai | azure  memory | sqlite | pgvector
                                         heading-aware

                 ┌─────────── POST /query ────────────┐
 question ──► HybridRetriever ──► RRF fusion ──► Reranker hook ──► prompt with numbered sources
              ├─ vector search (store)                                   │
              └─ BM25 (rank-bm25, cached per collection revision)        ▼
                                                   LLMProvider ──► answer with [n] ──► citations
```

```
rag/
  api.py            FastAPI app (create_app factory)
  auth.py           API key → collection access control
  config.py         Settings (pydantic-settings, RAG_* env vars / .env)
  service.py        RAGService: ingest, retrieve, generate
  prompts.py        prompt building and citation parsing
  ingestion/        loaders.py (PDF, DOCX, HTML, Markdown, text), chunking.py
  providers/        base.py interfaces, openai_compat.py (OpenAI-compatible + Azure), fake.py (offline)
  stores/           base.py interface, memory.py, sqlite.py, pgvector.py
  retrieval/        keyword.py (BM25), hybrid.py (RRF), rerank.py (hook + built-ins)
  evals/            dataset.py, metrics.py, runner.py, CLI (python -m rag.evals)
evals/datasets/     example eval dataset
examples/docs/      fictional sample documents (Markdown, HTML, PDF, DOCX)
tests/              offline pytest suite (pgvector tests run when RAG_TEST_DATABASE_URL is set)
```

### Extending

- **New LLM or embedding backend:** subclass `EmbeddingProvider` / `LLMProvider` in `rag/providers/base.py`.
- **New vector store:** subclass `VectorStore` in `rag/stores/base.py`. BM25 keyword search comes free from `iter_chunks`. Override `keyword_search` to use a native engine.
- **Re-ranking:** subclass `Reranker` in `rag/retrieval/rerank.py` (cross-encoder, Cohere, LLM, ...) and pass it in with `RAGService.from_settings(settings, reranker=MyReranker())`. The built-in `RAG_RERANKER=overlap` is a cheap lexical re-ranker.

### Notes for production

- The pgvector store creates its table on the first write, sized to the embedding dimension, with an HNSW cosine index (up to 2,000 dimensions). If you change the embedding model, use a new `RAG_PG_TABLE`.
- BM25 runs in process and is rebuilt when a collection changes. This works well up to roughly 100k chunks per collection. Beyond that, override `keyword_search` with PostgreSQL full-text search.
- The prompt treats retrieved text as untrusted data. Retrieval can still surface adversarial content, so keep collections scoped to what each key should see.
- Re-uploading a file with the same name into the same collection replaces the earlier version.

## Development

```bash
pip install -e ".[dev,postgres]"
pytest                      # offline: fake providers, memory + SQLite stores
ruff check . && ruff format --check .

# include the pgvector tests
docker compose up -d db
RAG_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest

python examples/make_binary_samples.py   # regenerate the sample PDF/DOCX
```

CI (GitHub Actions) runs lint, the test suite on Python 3.10–3.12 against a pgvector service, an eval quality gate and a Docker build smoke test.

## Roadmap

- [x] Initial release (v0.1.0)
- [x] Documentation and examples
- [x] CI and automated tests
- [ ] Azure AI Search vector store
- [ ] Cross-encoder re-ranker
- [ ] Streaming answers
- [ ] PostgreSQL full-text keyword search for large collections

Have an idea? [Open an issue](https://github.com/calliarc/rag-starter-kit/issues).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE) © 2026 CalliArc

---

Built and maintained by [CalliArc](https://www.calliarc.com/). Need help with AI development? [Talk to our team](https://www.calliarc.com/services/artificial-intelligence-development/).
