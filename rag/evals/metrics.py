"""Retrieval and answer metrics. All functions are pure and deterministic."""

from __future__ import annotations

import json
import re
from collections import Counter

from rag.evals.dataset import ExpectedSource
from rag.models import Citation
from rag.prompts import NO_ANSWER, extract_citation_refs, split_cited_sentences, strip_citations
from rag.providers.base import ChatMessage, LLMProvider
from rag.text import tokenize


def _is_relevant(c: Citation, expected: list[ExpectedSource]) -> bool:
    return any(e.matches(c.document, c.page) for e in expected)


def first_relevant_rank(retrieved: list[Citation], expected: list[ExpectedSource]) -> int | None:
    for rank, c in enumerate(retrieved, start=1):
        if _is_relevant(c, expected):
            return rank
    return None


def hit_at_k(retrieved: list[Citation], expected: list[ExpectedSource], k: int) -> float:
    rank = first_relevant_rank(retrieved[:k], expected)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(retrieved: list[Citation], expected: list[ExpectedSource]) -> float:
    rank = first_relevant_rank(retrieved, expected)
    return 1.0 / rank if rank else 0.0


def source_recall_at_k(retrieved: list[Citation], expected: list[ExpectedSource], k: int) -> float:
    if not expected:
        return 0.0
    found = sum(1 for e in expected if any(e.matches(c.document, c.page) for c in retrieved[:k]))
    return found / len(expected)


def citation_precision(citations: list[Citation], expected: list[ExpectedSource]) -> float | None:
    if not citations:
        return None
    return sum(1 for c in citations if _is_relevant(c, expected)) / len(citations)


def is_refusal(answer: str) -> bool:
    return answer.strip().rstrip(".").lower() == NO_ANSWER.rstrip(".").lower()


def groundedness(
    answer: str, contexts: dict[int, str], threshold: float = 0.6
) -> tuple[float | None, float | None]:
    """String/citation-based groundedness.

    The answer is split into sentences. A sentence is *supported* when it cites a retrieved source
    (``contexts`` maps citation number -> full source text) and at least ``threshold`` of its content
    tokens appear in the cited source text. Returns ``(groundedness, citation_coverage)``, both in
    [0, 1], or ``(None, None)`` for refusals.
    """
    if is_refusal(answer) or not answer.strip():
        return None, None
    sentences = [s for s in split_cited_sentences(answer) if tokenize(strip_citations(s))]
    if not sentences:
        return None, None
    supported = cited = 0
    for sent in sentences:
        refs = [r for r in extract_citation_refs(sent) if r in contexts]
        if not refs:
            continue
        cited += 1
        tokens = set(tokenize(strip_citations(sent)))
        source_tokens = set().union(*(set(tokenize(contexts[r])) for r in refs))
        if tokens and len(tokens & source_tokens) / len(tokens) >= threshold:
            supported += 1
    return supported / len(sentences), cited / len(sentences)


def token_f1(prediction: str, reference: str) -> float:
    p = tokenize(strip_citations(prediction))
    r = tokenize(reference)
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Share of reference tokens present in the prediction (lenient for verbose extractive answers)."""
    r = tokenize(reference)
    if not r:
        return 0.0
    p = set(tokenize(strip_citations(prediction)))
    return sum(1 for t in r if t in p) / len(r)


JUDGE_PROMPT = """You grade whether an ANSWER is fully supported by the SOURCES.
Return only JSON: {"score": <number between 0 and 1>, "reason": "<short reason>"}.
1 = every claim is supported by the sources; 0 = unsupported or contradicted."""


def llm_judge_groundedness(llm: LLMProvider, question: str, answer: str, sources: list[str]) -> float | None:
    """Optional LLM-as-judge groundedness score. Returns None if the judge output cannot be parsed."""
    joined = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    msg = f"SOURCES:\n{joined}\n\nQUESTION: {question}\n\nANSWER: {answer}"
    raw = llm.chat([ChatMessage("system", JUDGE_PROMPT), ChatMessage("user", msg)], temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        score = float(json.loads(m.group(0))["score"]) if m else float(raw.strip())
    except (ValueError, KeyError, TypeError):
        return None
    return max(0.0, min(1.0, score))
