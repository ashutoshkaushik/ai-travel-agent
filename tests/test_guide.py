"""Phase 8 tests: chunking, embedding cache, city filter, confidence bands, citations (no API calls)."""

import json

from fakes import ScriptedLLM, tool_call
from langchain.messages import AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from travel_planner.agents.guide_agent import (
    ENTRY_NOTE,
    GuideAnswer,
    build_guide_agent,
    final_answer_text,
    make_guide_tool,
    verify_citations,
)
from travel_planner.cities import SUPPORTED_CITIES
from travel_planner.rag import CachedEmbeddings, GuideIndex, confidence_band, load_chunks


# ---------- ingest ----------

def test_every_city_has_five_sections_with_metadata():
    chunks = load_chunks()
    assert len(chunks) == len(SUPPORTED_CITIES) * 5
    assert {c.metadata["city"] for c in chunks} == set(SUPPORTED_CITIES)
    tokyo_visa = next(c for c in chunks if c.id == "tokyo/visa-and-entry")
    assert tokyo_visa.page_content.startswith("Tokyo > Visa and entry\n")  # heading path prepended


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0
        self.inner = DeterministicFakeEmbedding(size=8)

    def embed_documents(self, texts):
        self.calls += len(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_embeddings_are_cached_on_disk(tmp_path):
    inner = CountingEmbeddings()
    cached = CachedEmbeddings(inner, "fake-model", cache_dir=tmp_path)
    first = cached.embed_documents(["a", "b"])
    second = cached.embed_documents(["a", "b", "c"])
    assert second[:2] == first
    assert inner.calls == 3  # "a" and "b" were embedded once, not twice


# ---------- retrieval ----------

def test_city_filter_only_returns_that_city():
    index = GuideIndex(DeterministicFakeEmbedding(size=16))
    hits = index.search("visa", city="Rome", k=5)
    assert hits and all(doc.metadata["city"] == "Rome" for doc, _ in hits)


def test_confidence_bands():
    assert confidence_band(0.10) == "stop"
    assert confidence_band(0.35) == "confirm"
    assert confidence_band(0.60) == "proceed"


class StubIndex:
    """Returns one fixed passage at a chosen similarity score."""

    def __init__(self, score: float):
        self.score = score

    def search(self, query, city=None, k=3):
        doc = Document(id="tokyo/visa-and-entry", page_content="Tokyo > Visa and entry\nVisa-free for 90 days.",
                       metadata={"city": "Tokyo", "section": "Visa and entry"})
        return [(doc, self.score)]


def test_weak_match_returns_no_passages():
    result = make_guide_tool(StubIndex(0.1)).invoke({"query": "vaccinations", "city": "Tokyo"})
    assert result["confidence"] == "stop" and result["passages"] == []


def test_middle_band_warns_the_model():
    result = make_guide_tool(StubIndex(0.35)).invoke({"query": "visa", "city": "Tokyo"})
    assert result["confidence"] == "confirm" and "ONLY if" in result["note"]
    assert result["passages"][0]["chunk_id"] == "tokyo/visa-and-entry"


# ---------- citations ----------

def tool_msg(chunk_ids):
    return ToolMessage(content=json.dumps({"passages": [{"chunk_id": c} for c in chunk_ids]}),
                       tool_call_id="c1", name="search_travel_guide")


def test_citations_must_have_been_retrieved():
    answer = GuideAnswer(answer="x", citations=["tokyo/visa-and-entry", "paris/money"], answered_from_guides=True)
    problems = verify_citations(answer, [tool_msg(["tokyo/visa-and-entry"])])
    assert problems == ["Cited 'paris/money', which was never retrieved."]


def test_claiming_guides_without_citations_is_flagged():
    answer = GuideAnswer(answer="x", citations=[], answered_from_guides=True)
    assert verify_citations(answer, []) == ["Claims to answer from the guides but cites nothing."]


def test_entry_answers_always_carry_the_official_sources_note():
    answer = GuideAnswer(answer="Visa-free for 90 days.", citations=["tokyo/visa-and-entry"], answered_from_guides=True)
    assert final_answer_text(answer).endswith(ENTRY_NOTE)
    other = GuideAnswer(answer="Tap a Suica card.", citations=["tokyo/getting-around"], answered_from_guides=True)
    assert ENTRY_NOTE not in final_answer_text(other)


def test_agent_answers_with_verified_citations():
    llm = ScriptedLLM(messages=iter([
        tool_call("search_travel_guide", {"query": "visa", "city": "Tokyo"}),
        tool_call("GuideAnswer", {"answer": "Visa-free for 90 days.", "citations": ["tokyo/visa-and-entry"],
                                  "answered_from_guides": True}, "c2"),
    ]))
    result = build_guide_agent(StubIndex(0.6), llm).invoke({"messages": [{"role": "user", "content": "visa?"}]})
    assert result["structured_response"].citations == ["tokyo/visa-and-entry"]
    assert verify_citations(result["structured_response"], result["messages"]) == []
