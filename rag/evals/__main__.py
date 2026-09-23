"""CLI: ``python -m rag.evals --dataset evals/datasets/demo.jsonl --docs examples/docs --collection demo``.

With ``--docs`` the documents are ingested into a fresh in-memory store first, so the run is self-contained
and reproducible. Without it, the eval runs against the store configured in your settings (e.g. pgvector).
Providers always come from settings (``RAG_LLM_PROVIDER`` / ``RAG_EMBEDDING_PROVIDER``; ``fake`` offline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import get_settings
from rag.evals.dataset import load_dataset
from rag.evals.runner import EvalReport, run_eval
from rag.providers import build_llm
from rag.service import RAGService
from rag.stores import InMemoryStore


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def print_report(report: EvalReport, verbose: bool = False) -> None:
    print(f"\nRAG eval  |  {report.examples} examples  |  k={report.k}  |  mode={report.mode}\n")
    width = max(len(name) for name in report.summary)
    for name, value in report.summary.items():
        print(f"  {name:<{width}}  {_fmt(value)}")
    misses = [r for r in report.results if r.hit == 0.0]
    if misses:
        print(f"\n  retrieval misses ({len(misses)}):")
        for r in misses:
            print(f"    - {r.id}: {r.question}")
    if verbose:
        print()
        for r in report.results:
            print(f"[{r.id}] {r.question}\n  answer: {r.answer}\n  retrieved: {', '.join(r.retrieved)}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag.evals", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", "-d", required=True, help="JSONL dataset path")
    ap.add_argument("--docs", help="ingest this file/directory into a fresh in-memory store first")
    ap.add_argument("--collection", "-c", help="default collection for examples without one")
    ap.add_argument("-k", type=int, default=5, help="retrieval depth (default 5)")
    ap.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    ap.add_argument(
        "--llm-judge", action="store_true", help="also score groundedness with the configured LLM"
    )
    ap.add_argument("--output", "-o", help="write the full JSON report here")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--min-hit-rate", type=float, help="exit 1 if hit@k is below this value")
    ap.add_argument("--min-mrr", type=float, help="exit 1 if MRR is below this value")
    ap.add_argument("--min-groundedness", type=float, help="exit 1 if groundedness is below this value")
    args = ap.parse_args(argv)

    settings = get_settings()
    examples = load_dataset(args.dataset)

    store = InMemoryStore() if args.docs else None
    service = RAGService.from_settings(settings, store=store)
    if args.docs:
        collections = {ex.collection or args.collection for ex in examples}
        if None in collections:
            ap.error("some examples have no collection; pass --collection")
        for coll in sorted(collections):
            service.ingest_path(coll, args.docs)

    judge = None
    if args.llm_judge:
        if settings.llm_provider == "fake":
            ap.error("--llm-judge needs a real LLM (set RAG_LLM_PROVIDER to openai or azure)")
        judge = build_llm(settings)

    report = run_eval(service, examples, collection=args.collection, k=args.k, mode=args.mode, judge=judge)
    print_report(report, verbose=args.verbose)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.output}")

    failures = []
    checks = [
        (args.min_hit_rate, f"hit@{args.k}"),
        (args.min_mrr, "mrr"),
        (args.min_groundedness, "groundedness"),
    ]
    for threshold, key in checks:
        value = report.summary.get(key)
        if threshold is not None and (value is None or value < threshold):
            failures.append(f"{key}={_fmt(value)} < {threshold}")
    if failures:
        print("\nFAILED thresholds: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
