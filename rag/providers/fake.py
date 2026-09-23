"""Deterministic offline providers for tests, CI and demos. No network, no API keys.

* ``HashingEmbeddings`` - feature-hashed bag of words + character trigrams, L2-normalised.
* ``ExtractiveLLM`` - answers by extracting the source sentences that best overlap the question and
  citing them with ``[n]`` markers. It understands the prompt format produced by ``rag.prompts``.
"""

from __future__ import annotations

import hashlib
import math

from rag.prompts import NO_ANSWER, parse_prompt
from rag.providers.base import ChatMessage, EmbeddingProvider, LLMProvider
from rag.text import split_sentences, tokenize


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    value = int.from_bytes(h, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbeddings(EmbeddingProvider):
    name = "fake"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        for tok in tokens:
            i, sign = _bucket("w:" + tok, self.dim)
            vec[i] += sign * 1.0
            padded = f"#{tok}#"
            for j in range(len(padded) - 2):
                i, sign = _bucket("c:" + padded[j : j + 3], self.dim)
                vec[i] += sign * 0.25
        for a, b in zip(tokens, tokens[1:]):
            i, sign = _bucket(f"b:{a}_{b}", self.dim)
            vec[i] += sign * 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class ExtractiveLLM(LLMProvider):
    name = "fake"

    def __init__(self, max_sentences: int = 2, min_overlap: int = 1) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def chat(self, messages: list[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        question, sources = parse_prompt(user)
        q_tokens = set(tokenize(question))
        if not q_tokens or not sources:
            return NO_ANSWER

        # Score every source sentence by question-term overlap, with a bonus when the source's section
        # heading matches the question and a small preference for higher-ranked sources.
        candidates: list[tuple[float, int, int, str, set[str]]] = []
        for ref, header, text in sources:
            heading_overlap = len(q_tokens & set(tokenize(header)))
            for pos, sent in enumerate(split_sentences(text)):
                s_tokens = set(tokenize(sent))
                overlap = len(q_tokens & s_tokens)
                if overlap < self.min_overlap:
                    continue
                score = overlap + 0.5 * heading_overlap + 1.0 / (ref + 1) - 0.01 * len(s_tokens)
                candidates.append((score, ref, pos, sent, s_tokens & q_tokens))
        if not candidates:
            return NO_ANSWER
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        # Greedy coverage: add a further sentence only if it is nearly as relevant as the best one and
        # covers question terms that are not covered yet.
        best = candidates[0][0]
        picked = [candidates[0]]
        covered = set(candidates[0][4])
        for cand in candidates[1:]:
            if len(picked) >= self.max_sentences:
                break
            if cand[0] >= 0.6 * best and cand[4] - covered:
                picked.append(cand)
                covered |= cand[4]
        parts = []
        for _, ref, _, sent, _ in picked:
            sent = sent.rstrip()
            if not sent.endswith((".", "!", "?")):
                sent += "."
            parts.append(f"{sent} [{ref}]")
        return " ".join(parts)
