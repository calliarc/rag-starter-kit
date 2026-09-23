import pytest

from rag.models import Chunk, ScoredChunk
from rag.retrieval import BM25Index, OverlapReranker, Reranker, reciprocal_rank_fusion
from rag.service import RAGService


def _chunk(i: int, text: str = "", doc: str = "d.md") -> Chunk:
    return Chunk(id=f"c{i}", collection="t", doc_id="d", doc_name=doc, index=i, text=text or f"chunk {i}")


def _sc(i: int) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(i), score=1.0)


def test_rrf_scores_and_ranks():
    vector = [_sc(1), _sc(2), _sc(3)]
    keyword = [_sc(3), _sc(1)]
    fused = reciprocal_rank_fusion([vector, keyword], k=60)
    assert [f.chunk.id for f in fused] == ["c1", "c3", "c2"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].vector_rank == 1 and fused[0].keyword_rank == 2
    assert fused[2].keyword_rank is None


def test_bm25_prefers_matching_chunk_and_handles_empty():
    idx = BM25Index([_chunk(0, "the cat sat"), _chunk(1, "dogs bark loudly"), _chunk(2, "birds sing")])
    res = idx.search("barking dog", 2)
    assert res and res[0].chunk.id == "c1"
    assert BM25Index([]).search("anything", 3) == []
    assert idx.search("the of and", 3) == []  # only stopwords


def test_hybrid_retrieval_finds_expected_documents(demo_service: RAGService):
    hits = demo_service.retrieve("demo", "How long are security logs retained?", 3)
    assert hits[0].chunk.doc_name == "security-policy.pdf"
    assert hits[0].chunk.page == 2
    hits = demo_service.retrieve("demo", "battery life of the Tidewater sensor", 3)
    assert hits[0].chunk.doc_name == "product-faq.html"


def test_retrieval_is_scoped_to_collection(demo_service: RAGService):
    assert demo_service.retrieve("other", "security logs", 5) == []


def test_overlap_reranker_reorders():
    candidates = [
        ScoredChunk(chunk=_chunk(0, "unrelated text"), score=0.03),
        ScoredChunk(chunk=_chunk(1, "parental leave is 18 weeks"), score=0.02),
    ]
    out = OverlapReranker(weight=0.8).rerank("parental leave weeks", candidates)
    assert out[0].chunk.id == "c1"


def test_custom_reranker_hook_is_called(settings):
    class Reverse(Reranker):
        called = False

        def rerank(self, query, candidates):
            Reverse.called = True
            return list(reversed(candidates))

    service = RAGService.from_settings(settings, reranker=Reverse())
    service.ingest_bytes("t", "a.md", b"# A\nalpha beta gamma\n\n# B\nalpha delta\n")
    normal = service.retrieve("t", "alpha", 2, rerank=False)
    reranked = service.retrieve("t", "alpha", 2)
    assert Reverse.called
    assert [c.chunk.id for c in reranked] == [c.chunk.id for c in reversed(normal)]
