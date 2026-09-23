"""Run an eval dataset against a ``RAGService`` and aggregate the metrics."""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel

from rag.evals import metrics as m
from rag.evals.dataset import EvalExample
from rag.providers.base import LLMProvider
from rag.service import RAGService


class ExampleResult(BaseModel):
    id: str
    question: str
    collection: str
    hit: float | None
    reciprocal_rank: float | None
    recall: float | None
    first_relevant_rank: int | None
    groundedness: float | None
    citation_coverage: float | None
    citation_precision: float | None
    answer_f1: float | None
    answer_recall: float | None
    llm_judge: float | None = None
    answered: bool
    answer: str
    retrieved: list[str]
    cited: list[str]


class EvalReport(BaseModel):
    k: int
    mode: str
    examples: int
    summary: dict[str, float | None]
    results: list[ExampleResult]


def _avg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _label(document: str, page: int | None) -> str:
    return f"{document}#p{page}" if page else document


def run_eval(
    service: RAGService,
    examples: list[EvalExample],
    *,
    collection: str | None = None,
    k: int = 5,
    mode: str = "hybrid",
    judge: LLMProvider | None = None,
) -> EvalReport:
    results: list[ExampleResult] = []
    for ex in examples:
        coll = ex.collection or collection
        if not coll:
            raise ValueError(f"example {ex.id}: no collection given (set it in the dataset or --collection)")
        contexts = service.retriever.retrieve(coll, ex.question, k, mode=mode)
        answer = service.generate(ex.question, contexts)
        expected = ex.sources()
        texts = {i: sc.chunk.text for i, sc in enumerate(contexts, start=1)}
        g, coverage = m.groundedness(answer.answer, texts)
        has_expected = bool(expected)
        rank = m.first_relevant_rank(answer.retrieved, expected) if has_expected else None
        judge_score = None
        if judge is not None and not m.is_refusal(answer.answer):
            judge_score = m.llm_judge_groundedness(judge, ex.question, answer.answer, list(texts.values()))
        results.append(
            ExampleResult(
                id=ex.id,
                question=ex.question,
                collection=coll,
                hit=m.hit_at_k(answer.retrieved, expected, k) if has_expected else None,
                reciprocal_rank=m.reciprocal_rank(answer.retrieved, expected) if has_expected else None,
                recall=m.source_recall_at_k(answer.retrieved, expected, k) if has_expected else None,
                first_relevant_rank=rank,
                groundedness=g,
                citation_coverage=coverage,
                citation_precision=m.citation_precision(answer.citations, expected) if has_expected else None,
                answer_f1=m.token_f1(answer.answer, ex.expected_answer) if ex.expected_answer else None,
                answer_recall=m.token_recall(answer.answer, ex.expected_answer)
                if ex.expected_answer
                else None,
                llm_judge=judge_score,
                answered=not m.is_refusal(answer.answer),
                answer=answer.answer,
                retrieved=[_label(c.document, c.page) for c in answer.retrieved],
                cited=[_label(c.document, c.page) for c in answer.citations],
            )
        )

    summary = {
        f"hit@{k}": _avg([r.hit for r in results]),
        "mrr": _avg([r.reciprocal_rank for r in results]),
        f"recall@{k}": _avg([r.recall for r in results]),
        "groundedness": _avg([r.groundedness for r in results]),
        "citation_coverage": _avg([r.citation_coverage for r in results]),
        "citation_precision": _avg([r.citation_precision for r in results]),
        "answer_f1": _avg([r.answer_f1 for r in results]),
        "answer_recall": _avg([r.answer_recall for r in results]),
        "answer_rate": _avg([1.0 if r.answered else 0.0 for r in results]),
    }
    if judge is not None:
        summary["llm_judge"] = _avg([r.llm_judge for r in results])
    return EvalReport(k=k, mode=mode, examples=len(results), summary=summary, results=results)
