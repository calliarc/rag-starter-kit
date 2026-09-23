import json
import math

import httpx
import pytest

from rag.models import Chunk, ScoredChunk
from rag.prompts import NO_ANSWER, build_messages, extract_citation_refs, parse_prompt
from rag.providers import build_embeddings, build_llm
from rag.providers.base import ChatMessage, ProviderError
from rag.providers.fake import ExtractiveLLM, HashingEmbeddings
from rag.providers.openai_compat import (
    AzureOpenAIEmbeddings,
    AzureOpenAILLM,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
)
from tests.conftest import make_settings


def _ctx(i: int, text: str, page=None) -> ScoredChunk:
    c = Chunk(id=f"c{i}", collection="t", doc_id="d", doc_name=f"doc{i}.md", index=i, text=text, page=page)
    return ScoredChunk(chunk=c, score=1.0)


def test_hashing_embeddings_are_deterministic_and_normalised():
    emb = HashingEmbeddings(dim=64)
    a, b = emb.embed(["parental leave policy", "parental leave policy"])
    assert a == b and len(a) == 64
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)
    near, far = emb.embed(["parental leave weeks", "sensor battery life"])

    def cos(x, y):
        return sum(p * q for p, q in zip(x, y))

    assert cos(a, near) > cos(a, far)


def test_prompt_roundtrip_and_citation_parsing():
    msgs = build_messages("What?", [_ctx(1, "Alpha text.", page=3), _ctx(2, "Beta text.")])
    assert msgs[0].role == "system" and "untrusted" in msgs[0].content
    question, sources = parse_prompt(msgs[1].content)
    assert question == "What?"
    assert sources[0][0] == 1 and "page 3" in sources[0][1] and sources[0][2] == "Alpha text."
    assert extract_citation_refs("a [2] b [1, 3] c [2]") == [2, 1, 3]


def test_extractive_llm_cites_best_sentence():
    llm = ExtractiveLLM()
    msgs = build_messages(
        "How many weeks of parental leave?",
        [_ctx(1, "Office opens at nine. Lunch is at noon."), _ctx(2, "Parental leave lasts 18 weeks.")],
    )
    out = llm.chat(msgs)
    assert out.startswith("Parental leave lasts 18 weeks.") and "[2]" in out
    assert llm.chat(build_messages("zebra quantum", [_ctx(1, "Office opens at nine.")])) == NO_ANSWER


def _recorder(responses):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        status, body = responses[min(len(calls) - 1, len(responses) - 1)]
        return httpx.Response(status, json=body)

    return calls, httpx.MockTransport(handler)


EMB_BODY = {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
CHAT_BODY = {"choices": [{"message": {"content": " Answer [1] "}}]}


def test_openai_compatible_embeddings_and_chat():
    calls, transport = _recorder([(200, EMB_BODY)])
    emb = OpenAICompatibleEmbeddings("http://localhost:11434/v1/", "nomic-embed-text", transport=transport)
    assert emb.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]  # sorted by index
    assert str(calls[0].url) == "http://localhost:11434/v1/embeddings"
    assert "authorization" not in calls[0].headers  # Ollama needs no key
    assert json.loads(calls[0].content) == {"model": "nomic-embed-text", "input": ["a", "b"]}

    calls, transport = _recorder([(200, CHAT_BODY)])
    llm = OpenAICompatibleLLM("https://api.example.com/v1", "m", api_key="sk-test", transport=transport)
    assert llm.chat([ChatMessage("user", "hi")], max_tokens=5) == "Answer [1]"
    assert calls[0].headers["authorization"] == "Bearer sk-test"
    assert json.loads(calls[0].content)["max_tokens"] == 5


def test_azure_openai_urls_and_headers():
    calls, transport = _recorder([(200, EMB_BODY)])
    emb = AzureOpenAIEmbeddings(
        "https://res.openai.azure.com/", "emb", "k", "2024-10-21", transport=transport
    )
    emb.embed(["x", "y"])
    assert calls[0].url.path == "/openai/deployments/emb/embeddings"
    assert calls[0].url.params["api-version"] == "2024-10-21"
    assert calls[0].headers["api-key"] == "k"

    calls, transport = _recorder([(200, CHAT_BODY)])
    llm = AzureOpenAILLM("https://res.openai.azure.com", "chat", "k", "2024-10-21", transport=transport)
    assert llm.chat([ChatMessage("user", "hi")]) == "Answer [1]"
    assert calls[0].url.path == "/openai/deployments/chat/chat/completions"


def test_retries_then_fails(monkeypatch):
    monkeypatch.setattr("rag.providers.openai_compat.time.sleep", lambda s: None)
    calls, transport = _recorder([(429, {"error": "slow down"}), (200, CHAT_BODY)])
    llm = OpenAICompatibleLLM("http://x/v1", "m", transport=transport)
    assert llm.chat([ChatMessage("user", "hi")]) == "Answer [1]"
    assert len(calls) == 2

    calls, transport = _recorder([(401, {"error": "bad key"})])
    llm = OpenAICompatibleLLM("http://x/v1", "m", transport=transport)
    with pytest.raises(ProviderError):
        llm.chat([ChatMessage("user", "hi")])
    assert len(calls) == 1  # 4xx other than 429 is not retried


def test_factory_builds_providers_and_validates_azure():
    s = make_settings()
    assert isinstance(build_embeddings(s), HashingEmbeddings)
    assert isinstance(build_llm(s), ExtractiveLLM)
    s = make_settings(llm_provider="openai", embedding_provider="openai", openai_base_url="http://h/v1")
    assert isinstance(build_llm(s), OpenAICompatibleLLM)
    with pytest.raises(ValueError, match="RAG_AZURE_OPENAI_ENDPOINT"):
        build_llm(make_settings(llm_provider="azure"))
