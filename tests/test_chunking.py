import pytest

from rag.ingestion.chunking import Chunker, ChunkingOptions, split_text
from rag.models import ParsedDocument, Section
from rag.text import split_sentences


def _long_text(n_sentences: int = 60) -> str:
    return " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(n_sentences))


def test_chunks_respect_size_and_overlap():
    text = _long_text()
    chunks = split_text(text, size=200, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share trailing content
    for a, b in zip(chunks, chunks[1:]):
        assert a[-30:] in b


def test_zero_overlap_has_no_repeated_units():
    chunks = split_text(_long_text(30), size=150, overlap=0)
    units = [u for c in chunks for u in split_sentences(c)]
    assert len(units) == len(set(units))


def test_pathological_long_word_is_split():
    chunks = split_text("x" * 1000, size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_heading_aware_chunks_do_not_cross_sections():
    doc = ParsedDocument(
        name="guide.md",
        sections=[Section(text="alpha " * 20, heading="A"), Section(text="beta " * 20, heading="B")],
    )
    aware = Chunker(ChunkingOptions(chunk_size=500, chunk_overlap=0)).chunk(doc, "c")
    assert [c.heading for c in aware] == ["A", "B"]
    merged = Chunker(ChunkingOptions(chunk_size=500, chunk_overlap=0, heading_aware=False)).chunk(doc, "c")
    assert len(merged) == 1 and merged[0].heading is None
    assert "alpha" in merged[0].text and "beta" in merged[0].text


def test_chunk_ids_are_stable_and_scoped_to_collection():
    doc = ParsedDocument(name="a.md", sections=[Section(text="hello world")])
    a1 = Chunker().chunk(doc, "one")
    a2 = Chunker().chunk(doc, "one")
    b = Chunker().chunk(doc, "two")
    assert a1[0].id == a2[0].id
    assert a1[0].doc_id != b[0].doc_id
    assert a1[0].id.endswith(":0")


def test_page_numbers_are_kept():
    doc = ParsedDocument(name="x.pdf", sections=[Section(text="p1", page=1), Section(text="p2", page=2)])
    assert [c.page for c in Chunker().chunk(doc, "c")] == [1, 2]


@pytest.mark.parametrize("size,overlap", [(10, 0), (100, 100), (100, 150)])
def test_invalid_options(size, overlap):
    with pytest.raises(ValueError):
        ChunkingOptions(chunk_size=size, chunk_overlap=overlap)


def test_paragraphs_and_sentences_are_joined_naturally():
    text = "First para sentence one. Sentence two.\n\nSecond paragraph."
    assert split_text(text, size=200, overlap=0) == [text]
    long_para = "Alpha beta gamma. " * 12
    chunks = split_text(long_para.strip(), size=100, overlap=0)
    assert all("\n" not in c for c in chunks)  # sentences of one paragraph are space-joined
