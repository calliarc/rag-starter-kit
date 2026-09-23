"""Configurable chunking.

Chunks are packed from paragraphs, falling back to sentences and then words for oversized units, up to
``chunk_size`` characters. Consecutive chunks share roughly ``chunk_overlap`` characters of trailing
context. With ``heading_aware`` enabled, chunks never cross a heading boundary and each chunk carries
its heading path (``Guide > Setup``) for citations and better retrieval.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag.models import Chunk, ParsedDocument, Section
from rag.text import split_sentences


@dataclass(frozen=True)
class ChunkingOptions:
    chunk_size: int = 800
    chunk_overlap: int = 120
    heading_aware: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size < 50:
            raise ValueError("chunk_size must be >= 50")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")


def make_doc_id(collection: str, name: str) -> str:
    return hashlib.sha256(f"{collection}/{name}".encode()).hexdigest()[:16]


Unit = tuple[str, int]  # (text, paragraph number)


def _split_units(text: str, size: int) -> list[Unit]:
    """Break text into units no longer than ``size``: paragraphs -> sentences -> words -> characters."""
    units: list[Unit] = []
    for p_no, para in enumerate(re.split(r"\n\s*\n", text)):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append((para, p_no))
            continue
        for sent in split_sentences(para):
            if len(sent) <= size:
                units.append((sent, p_no))
                continue
            cur = ""
            for w in sent.split(" "):
                while len(w) > size:  # pathological long token
                    if cur:
                        units.append((cur, p_no))
                        cur = ""
                    units.append((w[:size], p_no))
                    w = w[size:]
                if cur and len(cur) + 1 + len(w) > size:
                    units.append((cur, p_no))
                    cur = w
                else:
                    cur = f"{cur} {w}" if cur else w
            if cur:
                units.append((cur, p_no))
    return units


def _join(units: list[Unit]) -> str:
    """Join units: a space inside a paragraph, a blank line between paragraphs."""
    out = ""
    for i, (text, p_no) in enumerate(units):
        if i:
            out += " " if units[i - 1][1] == p_no else "\n\n"
        out += text
    return out


def _pack(units: list[Unit], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[Unit] = []
    for unit in units:
        if current and len(_join([*current, unit])) > size:
            chunks.append(_join(current))
            # carry trailing units into the next chunk as overlap
            carry: list[Unit] = []
            for prev in reversed(current):
                if len(_join([prev, *carry])) > overlap:
                    break
                carry.insert(0, prev)
            current = carry if len(_join([*carry, unit])) <= size else []
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


def split_text(text: str, size: int, overlap: int) -> list[str]:
    return _pack(_split_units(text, size), size, overlap)


class Chunker:
    def __init__(self, options: ChunkingOptions | None = None) -> None:
        self.options = options or ChunkingOptions()

    def _groups(self, doc: ParsedDocument) -> list[Section]:
        if self.options.heading_aware:
            return doc.sections
        # Merge sections per page, keeping headings inline as plain text.
        merged: list[Section] = []
        for s in doc.sections:
            leaf = s.heading.split(" > ")[-1] if s.heading else None
            text = f"{leaf}\n\n{s.text}" if leaf else s.text
            if merged and merged[-1].page == s.page:
                merged[-1] = Section(text=f"{merged[-1].text}\n\n{text}", page=s.page)
            else:
                merged.append(Section(text=text, page=s.page))
        return merged

    def chunk(self, doc: ParsedDocument, collection: str) -> list[Chunk]:
        doc_id = make_doc_id(collection, doc.name)
        chunks: list[Chunk] = []
        for section in self._groups(doc):
            for piece in split_text(section.text, self.options.chunk_size, self.options.chunk_overlap):
                idx = len(chunks)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}:{idx}",
                        collection=collection,
                        doc_id=doc_id,
                        doc_name=doc.name,
                        index=idx,
                        text=piece,
                        page=section.page,
                        heading=section.heading if self.options.heading_aware else None,
                        metadata={"title": doc.title} if doc.title else {},
                    )
                )
        return chunks
