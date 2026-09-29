"""Phase 5 tests: cost arithmetic, evidence checks, and the create_agent loop (scripted LLM + fake tools)."""

from datetime import date, timedelta

import pytest
from fakes import ScriptedLLM, tool_call
from langchain.messages import AIMessage
from langchain.tools import tool

from travel_planner.agents.research_agent import build_research_agent, research_request_message, tool_results_from
from travel_planner.models import FlightChoice, HotelChoice, SightChoice, TripPlan, TripRequest, cost_breakdown, verify_plan

DEPART = date.today() + timedelta(days=70)
REQUEST = TripRequest(origin="SFO", city="Tokyo", depart_date=DEPART, return_date=DEPART + timedelta(days=6),
                      adults=2, budget_usd=3000)

# What the fake tools "return": the evidence plans are checked against
FLIGHTS = {"status": "ok", "offers": [{"offer_id": "off_1", "airline": "Duffel Airways", "price": 1200.0}]}
HOTELS = {"status": "ok", "hotels": [{"hotel_id": "h_1", "name": "Hamacho Hotel", "price_per_night": 127, "total_price": 762}]}
SIGHTS = {"status": "ok", "sights": [{"name": "Sensō-ji", "price_usd": 0.0}, {"name": "Tokyo Tower", "price_usd": 9.54}]}

PLAN_ARGS = {
    "flight": {"offer_id": "off_1", "airline": "Duffel Airways", "price_total": 1200.0, "nonstop": True},
    "hotel": {"hotel_id": "h_1", "name": "Hamacho Hotel", "price_per_night": 127, "total_price": 762, "rating": 4.6},
    "sights": [{"name": "Sensō-ji", "day": 1, "price_usd": 0}, {"name": "Tokyo Tower", "day": 2, "price_usd": 9.54}],
    "rationale": "Nonstop, well-rated hotel, classic sights.",
}


def plan(**overrides) -> TripPlan:
    return TripPlan(**(PLAN_ARGS | overrides))


# ---------- Python owns the arithmetic ----------

def test_cost_breakdown():
    costs = cost_breakdown(plan(), REQUEST)
    assert costs.sights == 19.08  # 9.54 per person x 2 adults
    assert costs.food_estimate == 45 * 2 * 6  # Tokyo rate x adults x nights
    assert costs.total == pytest.approx(1200 + 762 + 19.08 + 540)
    assert costs.remaining == pytest.approx(3000 - costs.total)
    assert not costs.over_budget


def test_over_budget_detected():
    tight = REQUEST.model_copy(update={"budget_usd": 2000})
    assert cost_breakdown(plan(), tight).over_budget


# ---------- evidence checks catch invented or mis-copied values ----------

def test_plan_matching_tool_results_passes():
    assert verify_plan(plan(), [FLIGHTS, HOTELS, SIGHTS]) == []


def test_miscopied_sight_price_is_caught():
    """Regression: an agent once kept the sight names but set every price to $0."""
    free_tower = plan(sights=[SightChoice(name="Tokyo Tower", day=2, price_usd=0)])
    problems = verify_plan(free_tower, [FLIGHTS, HOTELS, SIGHTS])
    assert problems == ["Sight 'Tokyo Tower' price 0.0 doesn't match the search result's 9.54."]


def test_invented_and_miscopied_values_are_caught():
    bad = plan(
        flight=FlightChoice(offer_id="off_1", airline="Duffel Airways", price_total=1100.0, nonstop=True),  # wrong price
        hotel=HotelChoice(hotel_id="h_999", name="Imaginary Inn", price_per_night=50, total_price=300, rating=5),
        sights=[SightChoice(name="Mount Fuji Summit", day=1, price_usd=0)],
    )
    problems = verify_plan(bad, [FLIGHTS, HOTELS, SIGHTS])
    assert len(problems) == 3
    assert any("1100.0" in p for p in problems)
    assert any("Imaginary Inn" in p for p in problems)
    assert any("Mount Fuji Summit" in p for p in problems)


# ---------- the create_agent loop ----------

@tool
def fake_search_flights(origin: str) -> dict:
    """Fake flight search."""
    return FLIGHTS


@tool
def fake_search_hotels(city: str) -> dict:
    """Fake hotel search."""
    return HOTELS


@tool
def fake_search_top_sights(city: str) -> dict:
    """Fake sight search."""
    return SIGHTS


FAKE_TOOLS = [fake_search_flights, fake_search_hotels, fake_search_top_sights]
SEARCH_TURN = AIMessage(content="", tool_calls=[
    {"name": "fake_search_flights", "args": {"origin": "SFO"}, "id": "c1"},
    {"name": "fake_search_hotels", "args": {"city": "Tokyo"}, "id": "c2"},
    {"name": "fake_search_top_sights", "args": {"city": "Tokyo"}, "id": "c3"},
])


def test_agent_returns_verified_structured_plan():
    agent = build_research_agent(ScriptedLLM(messages=iter([SEARCH_TURN, tool_call("TripPlan", PLAN_ARGS, "c4")])), FAKE_TOOLS)
    result = agent.invoke({"messages": [research_request_message(REQUEST)]})
    assert isinstance(result["structured_response"], TripPlan)
    assert verify_plan(result["structured_response"], tool_results_from(result["messages"])) == []


def test_invalid_structured_output_is_sent_back_and_fixed():
    """ToolStrategy(handle_errors=True): a schema violation becomes feedback, and the LLM retries."""
    missing_hotel = {k: v for k, v in PLAN_ARGS.items() if k != "hotel"}
    agent = build_research_agent(ScriptedLLM(messages=iter([
        SEARCH_TURN,
        tool_call("TripPlan", missing_hotel, "c4"),  # invalid: no hotel
        tool_call("TripPlan", PLAN_ARGS, "c5"),      # corrected
    ])), FAKE_TOOLS)
    result = agent.invoke({"messages": [research_request_message(REQUEST)]})
    assert result["structured_response"].hotel.name == "Hamacho Hotel"
    feedback = [m.content for m in result["messages"] if getattr(m, "tool_call_id", None) == "c4"]
    assert feedback and "hotel" in feedback[0].lower()  # the validation error the LLM saw


def test_constraints_reach_the_agent():
    msg = research_request_message(REQUEST, constraints=["Hotel must be under $100/night."])
    assert "Hard constraints" in msg.content and "under $100/night" in msg.content
