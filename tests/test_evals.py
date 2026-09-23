import json

import pytest

from rag.evals import load_dataset, run_eval
from rag.evals.__main__ import main
from rag.evals.dataset import ExpectedSource
from rag.evals.metrics import (
    groundedness,
    hit_at_k,
    is_refusal,
    llm_judge_groundedness,
    reciprocal_rank,
    source_recall_at_k,
    token_f1,
)
from rag.models import Citation
from rag.prompts import NO_ANSWER
from rag.providers.base import LLMProvider
from tests.conftest import DATASET, DOCS


def _cit(doc: str, page=None) -> Citation:
    return Citation(
        ref=1, document=doc, doc_id="d", chunk_id="c", chunk_index=0, page=page, score=1, snippet=""
    )


def test_retrieval_metrics():
    retrieved = [_cit("a.md"), _cit("b.pdf", 2), _cit("c.md")]
    exp = [ExpectedSource.parse("b.pdf#p2")]
    assert hit_at_k(retrieved, exp, 1) == 0.0
    assert hit_at_k(retrieved, exp, 2) == 1.0
    assert reciprocal_rank(retrieved, exp) == 0.5
    assert reciprocal_rank(retrieved, [ExpectedSource.parse("b.pdf#p9")]) == 0.0
    both = [ExpectedSource.parse("a.md"), ExpectedSource.parse("z.md")]
    assert source_recall_at_k(retrieved, both, 3) == 0.5


def test_groundedness_checks_citations_and_support():
    ctx = {1: "Security logs are retained for 400 days.", 2: "Parental leave is 18 weeks."}
    assert groundedness("Security logs are retained for 400 days. [1]", ctx) == (1.0, 1.0)
    # cited but wrong source -> not supported
    assert groundedness("Security logs are retained for 400 days [2].", ctx) == (0.0, 1.0)
    # second sentence uncited
    g, cov = groundedness("Logs are retained for 400 days [1]. The moon is made of cheese.", ctx)
    assert g == 0.5 and cov == 0.5
    assert groundedness(NO_ANSWER, ctx) == (None, None)
    assert is_refusal(NO_ANSWER)


def test_token_f1():
    assert token_f1("26 days of vacation [1]", "26 days of vacation") == 1.0
    assert token_f1("nothing relevant", "26 days") == 0.0


def test_llm_judge_parsing():
    class Judge(LLMProvider):
        def __init__(self, reply):
            self.reply = reply

        def chat(self, messages, **kw):
            return self.reply

    assert llm_judge_groundedness(Judge('{"score": 0.8, "reason": "ok"}'), "q", "a", ["s"]) == 0.8
    assert llm_judge_groundedness(Judge('```json\n{"score": 7}\n```'), "q", "a", ["s"]) == 1.0
    assert llm_judge_groundedness(Judge("no idea"), "q", "a", ["s"]) is None


def test_dataset_loading(tmp_path):
    examples = load_dataset(DATASET)
    assert len(examples) >= 10
    assert all(ex.question and ex.expected_sources for ex in examples)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "x"}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_dataset(bad)


def test_run_eval_on_demo_corpus(demo_service):
    report = run_eval(demo_service, load_dataset(DATASET), k=5)
    s = report.summary
    assert s["hit@5"] >= 0.9
    assert s["mrr"] >= 0.8
    assert s["groundedness"] >= 0.8
    assert s["answer_rate"] == 1.0


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = main(
        [
            "-d",
            str(DATASET),
            "--docs",
            str(DOCS),
            "-k",
            "5",
            "-o",
            str(out),
            "--min-hit-rate",
            "0.9",
            "--min-mrr",
            "0.8",
        ]
    )
    assert code == 0
    assert "hit@5" in capsys.readouterr().out
    report = json.loads(out.read_text())
    assert report["examples"] == len(report["results"])
    # impossible threshold fails the run (useful as a CI gate)
    assert main(["-d", str(DATASET), "--docs", str(DOCS), "--min-hit-rate", "1.01"]) == 1
