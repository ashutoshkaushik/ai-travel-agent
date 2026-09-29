"""Phase 8: the travel-guide agent. Agentic RAG: retrieval is a tool the agent decides to call
and every answer must cite the guide chunks it used.
"""

import json
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from travel_planner.cities import SUPPORTED_CITIES
from travel_planner.guardrails import ToolOutputGuard, unfence
from travel_planner.llm import agent_middleware, primary_model
from travel_planner.prompts import GUIDE_PROMPT
from travel_planner.rag import GuideIndex, confidence_band, default_index
from travel_planner.tools._common import ok

CityName = Literal[tuple(SUPPORTED_CITIES)]


class GuideAnswer(BaseModel):
    """The answer to a travel question, grounded in the guides."""

    answer: str = Field(description="The answer, using only facts from the retrieved passages")
    citations: list[str] = Field(description="chunk_id of every passage the answer relies on")
    answered_from_guides: bool = Field(description="False if the guides don't cover the question")


def make_guide_tool(index: GuideIndex):
    @tool(parse_docstring=True)
    def search_travel_guide(query: str, city: CityName) -> dict:
        """Search the travel guides for one city: entry and visas, getting around, etiquette, money, where to stay.

        Args:
            query: The traveler's question, rephrased as a search query if helpful.
            city: The destination city the question is about.
        """
        hits = index.search(query, city)
        best = hits[0][1] if hits else 0.0
        band = confidence_band(best)
        if band == "stop":
            return ok(confidence="stop", best_score=round(best, 3), passages=[],
                      note="Nothing in the guides matches this question. Say it isn't covered.")
        note = ("Strong match." if band == "proceed" else
                "Weak match: answer ONLY if a passage directly answers the question; otherwise say it isn't covered.")
        return ok(confidence=band, best_score=round(best, 3), note=note, passages=[
            {"chunk_id": doc.id, "section": doc.metadata["section"], "text": doc.page_content, "score": round(score, 3)}
            for doc, score in hits
        ])

    return search_travel_guide


def build_guide_agent(index: GuideIndex | None = None, llm: BaseChatModel | None = None):
    return create_agent(
        llm or primary_model(),
        tools=[make_guide_tool(index or default_index())],
        system_prompt=GUIDE_PROMPT,
        response_format=ToolStrategy(GuideAnswer, handle_errors=True),
        middleware=[ToolOutputGuard()] + ([] if llm else agent_middleware()),
    )


def retrieved_chunk_ids(messages: list) -> set[str]:
    ids = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "search_travel_guide":
            ids |= {p["chunk_id"] for p in json.loads(unfence(m.content)).get("passages", [])}
    return ids


def verify_citations(answer: GuideAnswer, messages: list) -> list[str]:
    """Every citation must be a chunk that was actually retrieved in this conversation."""
    retrieved = retrieved_chunk_ids(messages)
    problems = [f"Cited {c!r}, which was never retrieved." for c in answer.citations if c not in retrieved]
    if answer.answered_from_guides and not answer.citations:
        problems.append("Claims to answer from the guides but cites nothing.")
    return problems


ENTRY_NOTE = "Entry rules change: confirm with official government sources before you travel."


def final_answer_text(answer: GuideAnswer) -> str:
    """Required notes are added by code, not requested in the prompt (the model skipped it in testing)."""
    text = answer.answer
    if any(c.endswith("/visa-and-entry") for c in answer.citations) and ENTRY_NOTE not in text:
        text += f" {ENTRY_NOTE}"
    return text
