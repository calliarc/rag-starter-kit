"""Eval dataset format (JSONL, one example per line).

    {"id": "pto-1",
     "question": "How many vacation days do full-time employees get?",
     "expected_sources": ["employee-handbook.md"],          # doc names; "file.pdf#p2" pins a page
     "expected_answer": "22 days of paid time off per year",  # optional, used for answer_f1
     "collection": "demo"}                                  # optional, falls back to --collection

Lines that are blank or start with ``//`` are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ExpectedSource(BaseModel):
    document: str
    page: int | None = None

    @classmethod
    def parse(cls, value: str) -> ExpectedSource:
        doc, sep, frag = value.partition("#")
        if sep and frag.lower().startswith("p") and frag[1:].isdigit():
            return cls(document=doc, page=int(frag[1:]))
        return cls(document=value)

    def matches(self, document: str, page: int | None) -> bool:
        return document == self.document and (self.page is None or self.page == page)


class EvalExample(BaseModel):
    id: str
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    collection: str | None = None

    def sources(self) -> list[ExpectedSource]:
        return [ExpectedSource.parse(s) for s in self.expected_sources]


def load_dataset(path: str | Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        data.setdefault("id", f"line-{lineno}")
        examples.append(EvalExample.model_validate(data))
    if not examples:
        raise ValueError(f"{path}: dataset is empty")
    return examples
