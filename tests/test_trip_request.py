"""Phase 4 tests: validation rules (pure Python) and the intake graph (scripted extractor, no API cost)."""

from datetime import date, timedelta

import pytest
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_planner.agents.trip_intake import MAX_ASK_ROUNDS, build_trip_intake
from travel_planner.models import TripRequest, TripRequestDraft, validate_draft
from travel_planner.tools.hotels import _is_private_hotel

DEPART = date.today() + timedelta(days=70)
VALID = dict(origin="SFO", city="tokyo", depart_date=DEPART.isoformat(),
             return_date=(DEPART + timedelta(days=6)).isoformat(), adults=2, budget_usd=4000)


def draft(**overrides) -> TripRequestDraft:
    return TripRequestDraft(**(VALID | overrides))


# ---------- validation rules ----------

def test_valid_draft_becomes_request():
    request, problems = validate_draft(draft())
    assert problems == []
    assert request.city == "Tokyo"  # normalized from "tokyo"
    assert request.nights == 6
    assert request.destination_airport == "NRT"


def test_kyoto_flies_into_osaka():
    request, _ = validate_draft(draft(city="Kyoto"))
    assert request.destination_airport == "KIX"


def test_missing_fields_listed_all_at_once():
    _, problems = validate_draft(draft(depart_date=None, budget_usd=None))
    assert len(problems) == 2
    assert any("departure date" in p for p in problems)
    assert any("budget" in p for p in problems)


@pytest.mark.parametrize("overrides, expected", [
    (dict(city="Dubai"), "I can plan trips to: Tokyo"),
    (dict(origin="LHR"), "not a supported US departure airport"),
    (dict(return_date=(DEPART + timedelta(days=20)).isoformat()), "I can plan 2 to 14 nights"),
    (dict(return_date=(DEPART + timedelta(days=1)).isoformat()), "I can plan 2 to 14 nights"),
    (dict(depart_date="2020-01-01", return_date="2020-01-05"), "not in the future"),
    (dict(depart_date="Dec 6"), "isn't a valid date"),
    (dict(adults=6), "1 to 4 adults"),
    (dict(budget_usd=0), "more than $0"),
    (dict(origin="JFK", city="New York"), "same city"),
])
def test_invalid_drafts_explain_why(overrides, expected):
    request, problems = validate_draft(draft(**overrides))
    assert request is None
    assert any(expected in p for p in problems), problems


def test_trip_request_cannot_be_built_invalid():
    with pytest.raises(ValueError):
        TripRequest(**(VALID | {"city": "Atlantis"}))


# ---------- hotel quality rules ----------

@pytest.mark.parametrize("name, allowed", [
    ("Hamacho Hotel", True),
    ("Dormitory Kinshu", False),
    ("BEE-HIVE巣鴨 (Male Only)", False),
    ("Joe Backpackers in Tokyo", False),
    ("Nine Hours Capsule Hotel", False),
    ("Tokyo Central Youth Hostel", False),
    ("Camping Village Fabulous", False),
])
def test_shared_room_listings_are_excluded(name, allowed):
    assert _is_private_hotel({"name": name}) is allowed


# ---------- intake graph ----------

def scripted_extractor(*drafts: TripRequestDraft):
    """Fake extractor: returns the next draft each time it's called, ignoring the messages."""
    queue = list(drafts)
    return RunnableLambda(lambda _messages: queue.pop(0))


def test_complete_request_needs_no_questions():
    agent = build_trip_intake(scripted_extractor(draft()), checkpointer=InMemorySaver())
    result = agent.invoke({"messages": [HumanMessage("full details")], "ask_rounds": 0},
                          {"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" not in result
    assert result["request"]["city"] == "Tokyo"
    assert result["messages"][-1].content.startswith("Got it: SFO → Tokyo")


def test_asks_for_missing_details_then_completes():
    agent = build_trip_intake(
        scripted_extractor(draft(depart_date=None, return_date=None, origin=None), draft()),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t2"}}
    first = agent.invoke({"messages": [HumanMessage("Tokyo in December")], "ask_rounds": 0}, config)
    question = first["__interrupt__"][0].value["question"]
    assert "departure date" in question and "departure airport" in question  # one question, all gaps

    final = agent.invoke(Command(resume={"answer": "From SFO, Dec 6-12"}), config)
    assert final["request"] is not None
    assert final["ask_rounds"] == 1
    assert any(m.content == "From SFO, Dec 6-12" for m in final["messages"])  # answer is in the conversation


def test_gives_up_after_max_rounds():
    agent = build_trip_intake(
        scripted_extractor(*[draft(city="Dubai")] * (MAX_ASK_ROUNDS + 1)),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t3"}}
    agent.invoke({"messages": [HumanMessage("Dubai please")], "ask_rounds": 0}, config)
    for _ in range(MAX_ASK_ROUNDS - 1):
        agent.invoke(Command(resume={"answer": "still Dubai"}), config)
    final = agent.invoke(Command(resume={"answer": "Dubai!"}), config)
    assert final["request"] is None
    assert "couldn't pin down" in final["messages"][-1].content
