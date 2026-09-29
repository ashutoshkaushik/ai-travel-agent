"""Phase 6 tests: every path through the planner, with a scripted research function (no LLM, no APIs)."""

from datetime import date, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_planner.agents.trip_planner import (
    MAX_RETRIES,
    build_trip_planner,
    hotel_price_cap,
    initial_state,
    parse_escalation,
    parse_review,
)
from travel_planner.models import SightChoice, TripPlan, TripRequest, cost_breakdown

DEPART = date.today() + timedelta(days=70)
NIGHTS = 6


def request(budget: int) -> TripRequest:
    return TripRequest(origin="SFO", city="Tokyo", depart_date=DEPART, return_date=DEPART + timedelta(days=NIGHTS),
                       adults=2, budget_usd=budget)


def plan(per_night: float, flight: float = 1200.0) -> TripPlan:
    return TripPlan(
        flight={"offer_id": "off_1", "airline": "Duffel Airways", "price_total": flight, "nonstop": True},
        hotel={"hotel_id": f"h_{per_night}", "name": f"Hotel {per_night}", "price_per_night": per_night,
               "total_price": per_night * NIGHTS, "rating": 4.5},
        sights=[{"name": "Sensō-ji", "day": 1, "price_usd": 0}],
        rationale="test",
    )


def backed(p: TripPlan) -> tuple[TripPlan, list[dict]]:
    """A research result whose tool evidence matches the plan exactly."""
    return p, [
        {"status": "ok", "offers": [{"offer_id": p.flight.offer_id, "airline": p.flight.airline, "price": p.flight.price_total,
                                     "slices": [{"stops": 0}]}]},
        {"status": "ok", "hotels": [{"hotel_id": p.hotel.hotel_id, "name": p.hotel.name, "rating": p.hotel.rating,
                                     "price_per_night": p.hotel.price_per_night, "total_price": p.hotel.total_price}]},
        {"status": "ok", "sights": [{"name": s.name, "price_usd": s.price_usd} for s in p.sights]},
    ]


# Base cost with a $0 hotel: flight 1200 + food 45*2*6 = 540 -> 1740, so hotel budget = budget - 1740


class ScriptedResearch:
    """Returns pre-written (plan, problems) results and records the constraints it was given."""

    def __init__(self, *results):
        self.results = list(results)
        self.constraints_seen = []

    def __call__(self, req, constraints, config):
        self.constraints_seen.append(list(constraints))
        return self.results.pop(0)


def start(research: ScriptedResearch, budget: int, thread: str):
    planner = build_trip_planner(research, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    return planner, config, planner.invoke(initial_state(request(budget)), config)


def paused_kind(result) -> str:
    return result["__interrupt__"][0].value["kind"]


# ---------- pure helpers ----------

def test_hotel_price_cap_math():
    costs = cost_breakdown(plan(150), request(2500))  # 1740 + 900 = 2640 -> $140 over
    assert hotel_price_cap(plan(150), costs, NIGHTS) == 126  # 150 - 140/6 = 126.67 -> 126


def test_hotel_price_cap_none_when_hotel_cannot_fix_it():
    costs = cost_breakdown(plan(150), request(1500))  # flight + food alone exceed the budget
    assert hotel_price_cap(plan(150), costs, NIGHTS) is None


@pytest.mark.parametrize("answer, expected", [
    ("$3,500", ("raise", 3500)), ("3.5k", ("raise", 3500)), ("accept", ("accept", None)),
    ("cancel please", ("cancel", None)), ("I don't know", ("unclear", None)),
])
def test_parse_escalation(answer, expected):
    assert parse_escalation(answer) == expected


def test_parse_review():
    assert parse_review("Looks good!") == ("approve", None)
    assert parse_review("fewer museums") == ("modify", "fewer museums")


# ---------- graph paths ----------

def test_within_budget_goes_straight_to_review_then_approved():
    planner, config, result = start(ScriptedResearch(backed(plan(100))), budget=4000, thread="ok")
    assert paused_kind(result) == "review"
    final = planner.invoke(Command(resume={"answer": "approve"}), config)
    assert final["status"] == "approved"


def test_over_budget_replans_with_computed_hotel_cap():
    research = ScriptedResearch(backed(plan(150)), backed(plan(110)))
    planner, config, result = start(research, budget=2500, thread="replan")
    assert paused_kind(result) == "review"  # second plan fit: 1740 + 660 = 2400
    assert research.constraints_seen[0] == []
    assert "at most $126" in research.constraints_seen[1][0]
    assert result["retries"] == 1


def test_unfixable_budget_escalates_immediately_without_wasting_retries():
    research = ScriptedResearch(backed(plan(150)))
    planner, config, result = start(research, budget=1500, thread="esc")
    assert paused_kind(result) == "escalation"
    assert len(research.constraints_seen) == 1  # no pointless retry


def test_escalation_raise_budget_then_review():
    planner, config, _ = start(ScriptedResearch(backed(plan(150))), budget=1500, thread="raise")
    result = planner.invoke(Command(resume={"answer": "$2,700"}), config)
    assert paused_kind(result) == "review"
    assert result["request"]["budget_usd"] == 2700


def test_escalation_accept_and_cancel():
    planner, config, _ = start(ScriptedResearch(backed(plan(150))), budget=1500, thread="accept")
    assert paused_kind(planner.invoke(Command(resume={"answer": "accept"}), config)) == "review"

    planner, config, _ = start(ScriptedResearch(backed(plan(150))), budget=1500, thread="cancel")
    assert planner.invoke(Command(resume={"answer": "cancel"}), config)["status"] == "cancelled"


def test_escalation_unclear_answer_asks_again():
    planner, config, _ = start(ScriptedResearch(backed(plan(150))), budget=1500, thread="unclear")
    result = planner.invoke(Command(resume={"answer": "hmm"}), config)
    assert paused_kind(result) == "escalation"


def test_unclear_replies_are_capped():
    """Regression: an unparseable reply used to re-ask forever (4,000+ pauses in a real run)."""
    planner, config, _ = start(ScriptedResearch(backed(plan(150))), budget=1500, thread="unclear-cap")
    result = None
    for _ in range(3):
        result = planner.invoke(Command(resume={"answer": "hmm"}), config)
    assert result["status"] == "cancelled"
    assert "__interrupt__" not in result


def test_replan_keeps_flight_and_sights():
    """Regression: a replan once swapped the hotel but added 4 sights, eating the savings."""
    research = ScriptedResearch(backed(plan(150)), backed(plan(110)))
    start(research, budget=2500, thread="keep")
    constraint = research.constraints_seen[1][0]
    assert "Change ONLY the hotel" in constraint
    assert "off_1" in constraint and "Sensō-ji (day 1)" in constraint


def test_hotel_over_the_cap_escalates_instead_of_retrying():
    """Regression: the agent returned a $63 hotel when the cap was $60; code must notice."""
    research = ScriptedResearch(backed(plan(150)), backed(plan(140)), backed(plan(100)))  # 140 > cap of 126
    planner, config, result = start(research, budget=2500, thread="cap-violated")
    assert paused_kind(result) == "escalation"
    assert len(research.constraints_seen) == 2  # did not burn the last retry


def test_a_hotel_within_the_cap_always_fits():
    """With flight and sights held fixed by code, the computed cap guarantees one replan is enough."""
    pricier = plan(110, 1300)  # the LLM tried a different flight; only its hotel search is new evidence
    research = ScriptedResearch(backed(plan(150, 1200)), (pricier, backed(pricier)[1][1:2]))
    planner, config, result = start(research, budget=2400, thread="cap-fits")
    assert paused_kind(result) == "review"
    assert result["plan"]["flight"]["price_total"] == 1200  # code kept the original flight
    assert len(research.constraints_seen) == 2


def test_review_feedback_starts_new_round_with_feedback_as_constraint():
    research = ScriptedResearch(backed(plan(100)), backed(plan(90)))
    planner, config, _ = start(research, budget=4000, thread="modify")
    result = planner.invoke(Command(resume={"answer": "Hotel closer to Shibuya please"}), config)
    assert paused_kind(result) == "review"
    assert "Shibuya" in research.constraints_seen[1][0]


def test_bad_evidence_retries_then_fails():
    unbacked = (plan(100), [])  # a plan with no tool results behind it
    research = ScriptedResearch(*[unbacked] * (MAX_RETRIES + 1))
    planner, config, result = start(research, budget=4000, thread="bad")
    assert result["status"] == "failed"
    assert "Copy them exactly" in research.constraints_seen[1][-1]


def test_items_kept_from_an_earlier_attempt_count_as_evidence():
    """Regression: a replan that kept the flight and sights (without re-searching) was flagged as invented."""
    first = backed(plan(150))
    cheaper = plan(110)
    only_hotel_searched = [{"status": "ok", "hotels": [{"hotel_id": cheaper.hotel.hotel_id, "total_price": cheaper.hotel.total_price}]}]
    research = ScriptedResearch(first, (cheaper, only_hotel_searched))
    planner, config, result = start(research, budget=2500, thread="kept")
    assert result["problems"] == []
    assert paused_kind(result) == "review"


def test_evidence_retry_keeps_the_hotel_cap():
    """Regression: an evidence retry used to replace the hotel-cap constraint instead of adding to it."""
    research = ScriptedResearch(backed(plan(150)), (plan(110), []), backed(plan(110)))
    start(research, budget=2500, thread="keep-cap")
    third = research.constraints_seen[2]
    assert any("at most $126" in c for c in third)
    assert any("Copy them exactly" in c for c in third)


def test_hotel_only_replan_carries_sights_over_in_code():
    """Regression: replans kept sight names but zeroed their prices; now code carries them over."""
    zeroed = plan(110).model_copy(update={"sights": [SightChoice(name="Sensō-ji", day=1, price_usd=99.0)]})
    research = ScriptedResearch(backed(plan(150)), (zeroed, backed(plan(110))[1]))  # tools had the real price
    planner, config, result = start(research, budget=2500, thread="carry")
    assert result["plan"]["sights"] == plan(150).model_dump()["sights"]
    assert result["problems"] == []


def test_hotel_only_replan_carries_the_flight_over_in_code():
    """Regression: a replan paired one offer's price with a neighboring offer's id."""
    from travel_planner.models import FlightChoice

    mixed = plan(110).model_copy(update={"flight": FlightChoice(offer_id="off_1", airline="Iberia", price_total=999.0, nonstop=True)})
    research = ScriptedResearch(backed(plan(150)), (mixed, backed(plan(110))[1]))
    planner, config, result = start(research, budget=2500, thread="carry-flight")
    assert result["plan"]["flight"] == plan(150).model_dump()["flight"]
    assert result["problems"] == []


def test_budget_replan_keeps_the_travelers_feedback():
    """Regression: 'rated 4.5 or higher' was dropped when the budget replan added a price cap."""
    research = ScriptedResearch(backed(plan(100)), backed(plan(150)), backed(plan(110)))
    planner, config, _ = start(research, budget=2500, thread="feedback-kept")
    planner.invoke(Command(resume={"answer": "hotel rated 4.5 or higher"}), config)  # -> $150 hotel, over budget
    last = research.constraints_seen[-1]
    assert any("4.5 or higher" in c for c in last) and any("at most $" in c for c in last)


def test_review_offers_the_options_research_found():
    research = ScriptedResearch((plan(100), backed(plan(100))[1] + backed(plan(80))[1]))
    planner, config, result = start(research, budget=4000, thread="options")
    options = result["__interrupt__"][0].value["options"]
    assert [h["price_per_night"] for h in options["hotels"]] == [80, 100]  # cheapest first, current pick included


def test_selecting_options_rebuilds_the_plan_from_evidence_and_approves():
    research = ScriptedResearch((plan(100), backed(plan(100))[1] + backed(plan(80))[1]))
    planner, config, _ = start(research, budget=4000, thread="select")
    final = planner.invoke(Command(resume={"answer": "approve", "selection": {"offer_id": "off_1", "hotel_id": "h_80"}}), config)
    assert final["status"] == "approved"
    assert final["plan"]["hotel"]["price_per_night"] == 80
    assert final["costs"]["hotel"] == 480
    assert final["details"]["hotel"]["hotel_id"] == "h_80"  # evidence for the itinerary and map



def test_escalation_is_recorded_as_a_readable_decision():
    planner, config, _ = start(ScriptedResearch(backed(plan(150))), budget=1500, thread="decision-log")
    final = planner.invoke(Command(resume={"answer": "$2,700"}), config)
    decisions = [m for m in final["messages"] if m.additional_kwargs.get("decision")]
    assert decisions[0].content.startswith("Over budget by $")
    assert decisions[1].content == "Raised budget to $2,700"


def test_hotel_is_repriced_when_the_flight_lands_the_next_day():
    overnight = plan(100)
    evidence = backed(overnight)[1]
    evidence[0]["offers"][0]["slices"] = [
        {"from": "SFO", "to": "NRT", "depart": f"{DEPART}T06:58:00", "arrive": f"{DEPART + timedelta(days=1)}T03:02:00", "stops": 0},
        {"from": "NRT", "to": "SFO", "depart": f"{DEPART + timedelta(days=NIGHTS)}T11:00:00", "arrive": f"{DEPART + timedelta(days=NIGHTS)}T09:00:00", "stops": 0},
    ]
    planner, config, result = start(ScriptedResearch((overnight, evidence)), budget=4000, thread="reprice")
    assert result["plan"]["hotel"]["total_price"] == 500.0  # 600 for 6 nights -> 5 nights used
    assert any("5 nights instead of 6" in m.content for m in result["messages"])
