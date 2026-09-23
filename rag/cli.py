"""Command line: ``python -m rag {serve,ingest,query,collections}``. Uses the same settings as the API."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rag.config import get_settings


def _service():
    from rag.service import RAGService

    return RAGService.from_settings(get_settings())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag", description="RAG Starter Kit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")

    p = sub.add_parser("ingest", help="ingest a file or directory into a collection")
    p.add_argument("path")
    p.add_argument("--collection", "-c", required=True)

    p = sub.add_parser("query", help="ask a question against a collection")
    p.add_argument("question")
    p.add_argument("--collection", "-c", required=True)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true", help="print the full JSON response")

    sub.add_parser("collections", help="list collections")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("rag.api:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
        return 0

    service = _service()
    try:
        if args.cmd == "ingest":
            for doc in service.ingest_path(args.collection, args.path):
                print(f"{doc.document}: {doc.chunks} chunks")
        elif args.cmd == "query":
            ans = service.answer(args.collection, args.question, args.top_k)
            if args.json:
                print(ans.model_dump_json(indent=2))
            else:
                print(ans.answer, "\n")
                for c in ans.citations:
                    loc = f", page {c.page}" if c.page else ""
                    sec = f" ({c.heading})" if c.heading else ""
                    print(f"  [{c.ref}] {c.document}{loc}{sec} chunk {c.chunk_index}")
        elif args.cmd == "collections":
            print(json.dumps([c.model_dump() for c in service.list_collections()], indent=2))
    finally:
        service.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
