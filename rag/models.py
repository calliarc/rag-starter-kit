"""Core data models shared across ingestion, storage, retrieval and the API."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_collection_name(name: str) -> str:
    if not COLLECTION_RE.match(name or ""):
        raise ValueError(
            "collection must be 1-64 chars of letters, digits, '_' or '-' and start with a letter or digit"
        )
    return name


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class Section(BaseModel):
    """A logical block of a parsed document (a PDF page, a markdown section, ...)."""

    text: str
    page: int | None = None
    heading: str | None = None


class ParsedDocument(BaseModel):
    name: str
    title: str | None = None
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    collection: str
    doc_id: str
    doc_name: str
    index: int
    text: str
    page: int | None = None
    heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def search_text(self) -> str:
        """Text used for embedding and keyword indexing (heading adds useful context)."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


class Citation(BaseModel):
    ref: int = Field(description="Citation marker number used in the answer, e.g. [1]")
    document: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    page: int | None = None
    heading: str | None = None
    score: float
    snippet: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: list[Citation] = Field(default_factory=list)


class CollectionInfo(BaseModel):
    name: str
    documents: int
    chunks: int


class IngestedDocument(BaseModel):
    document: str
    doc_id: str
    chunks: int
    pages: int | None = None
