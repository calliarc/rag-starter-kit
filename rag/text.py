"""Small text utilities shared by keyword search, fake providers and eval metrics."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does did for from had has have how i if in into
    is it its of on or our so than that the their them then there these they this to was we were what
    when where which who whom why will with would you your yours about after before also any all may
    might must not no should shall such via per each other more most over under up out off very just
    only own same s t""".split()
)


def stem(token: str) -> str:
    """Very small suffix stripper so that e.g. "lasts"/"last" and "calibrated"/"calibrating" match."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def tokenize(text: str, *, drop_stopwords: bool = True, stemming: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
