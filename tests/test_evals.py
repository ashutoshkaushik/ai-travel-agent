"""The eval harness itself: dataset shape, code scorers, cache keys, and feedback → case."""

import json
from pathlib import Path

from langchain_core.load import dumps
from langchain.messages import AIMessage, HumanMessage

from evals.from_feedback import case_from
from evals.scorers import RESULT_CHECKS, STEP_CHECKS, path_matches, score
from travel_planner.llm_cache import stable_key

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.jsonl"


def record(**overrides) -> dict:
    base = {"id": "x", "category": "planner", "turns": ["plan"], "intent": "plan_trip", "path": [], "pauses": [],
            "replies": [], "questions": [], "tool_calls": [], "llm_calls": 1, "evidence": {}, "trip_status": None,
            "error": None, "seconds": 0.1}
    return {**base, **overrides}


PLAN = {"flight": {"offer_id": "off_1", "airline": "Duffel Airways", "price_total": 852.88, "nonstop": True},
        "hotel": {"hotel_id": "h1", "name": "MOB HOUSE", "price_per_night": 99.0, "total_price": 495.0, "rating": 4.3},
        "sights": [{"name": "Eiffel Tower", "day": 1, "price_usd": 16.91}], "rationale": "."}
EVIDENCE = {"search_flights": [{"offers": [{"offer_id": "off_1", "price": 852.88}]}],
            "search_hotels": [{"hotels": [{"hotel_id": "h1", "price_per_night": 99}]}],
            "search_top_sights": [{"sights": [{"name": "Eiffel Tower"}]}]}


def review(total=2000.0, budget=5000, plan=PLAN):
    return {"kind": "review", "plan": plan, "costs": {"total": total, "budget": budget}}


def test_dataset_is_well_formed():
    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    assert len(cases) >= 25
    assert len({c["id"] for c in cases}) == len(cases)
    assert {"planner", "budget", "intake", "guide", "booking", "guardrails"} <= {c["category"] for c in cases}
    for c in cases:
        for assertion in c["assertions"]:
            assert assertion.partition(":")[0] in RESULT_CHECKS | STEP_CHECKS, (c["id"], assertion)
    assert sum(c["category"] == "guardrails" for c in cases) >= 3  # the prompt-injection cases


def test_path_match_is_an_ordered_subsequence():
    assert path_matches(["a", "x", "b", "c"], ["a", "b", "c"])
    assert not path_matches(["b", "a"], ["a", "b"])


def test_grounded_plan_passes_the_result_checks():
    r = record(pauses=[review()], evidence=EVIDENCE)
    for name in ("within_budget", "budget_honest", "prices_match_evidence", "no_invented_sights"):
        assert RESULT_CHECKS[name](r, "")[0], name


def test_invented_price_and_sight_are_caught():
    bad = {**PLAN, "flight": {**PLAN["flight"], "price_total": 700.0}, "sights": [{"name": "Moon Base", "day": 1}]}
    r = record(pauses=[review(plan=bad)], evidence=EVIDENCE)
    assert not RESULT_CHECKS["prices_match_evidence"](r, "")[0]
    assert not RESULT_CHECKS["no_invented_sights"](r, "")[0]


def test_over_budget_plan_shown_for_approval_is_dishonest_but_escalating_is_honest():
    assert not RESULT_CHECKS["budget_honest"](record(pauses=[review(total=6000)]), "")[0]
    escalated = {"kind": "escalation", "costs": {"total": 6000, "budget": 5000}}
    assert RESULT_CHECKS["budget_honest"](record(pauses=[escalated]), "")[0]


def test_citations_must_have_been_retrieved():
    evidence = {"search_travel_guide": [{"passages": [{"chunk_id": "tokyo-visa-0"}]}]}
    good = record(replies=["Yes.\n📖 Sources: tokyo-visa-0"], evidence=evidence)
    bad = record(replies=["Yes.\n📖 Sources: tokyo-visa-0, made-up-9"], evidence=evidence)
    assert RESULT_CHECKS["citations_valid"](good, "")[0]
    assert not RESULT_CHECKS["citations_valid"](bad, "")[0]


def test_a_booking_tool_call_always_fails_the_steps_level():
    case = {"expected_path": [], "expected_intent": "any", "assertions": [], "max_tool_calls": 5}
    steps = score(record(tool_calls=["book_hotel"]), case)["steps"]
    assert ("never books without approval", False, "a book_* call ran") in steps


def test_cache_key_ignores_random_message_ids_and_usage():
    def prompt(msg_id, usage):
        ai = AIMessage("hi", id=msg_id, usage_metadata=usage)
        return dumps([HumanMessage("plan", id=msg_id), ai])
    a = prompt("uuid-1", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    b = prompt("uuid-2", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "total_cost": 0})
    assert stable_key(a, "model") == stable_key(b, "model")
    assert stable_key(a, "model") != stable_key(dumps([HumanMessage("other")]), "model")


def test_a_rejection_becomes_a_regression_case():
    event = {"ts": "t", "decision": "reject", "action": "book_hotel", "reason": "Heathrow is too far from central London",
             "args": {"hotel_name": "The Broadway Hotel - London Heathrow"},
             "request": {"city": "London", "origin": "SFO", "depart_date": "2027-03-10", "return_date": "2027-03-17",
                         "adults": 2, "budget_usd": 3500}}
    case = case_from(event)
    assert case["category"] == "feedback"
    assert "Heathrow is too far" in case["turns"][0] and "London from SFO" in case["turns"][0]
    assert "plan_avoids:The Broadway Hotel - London Heathrow" in case["assertions"]
    assert case_from({**event, "request": None}) is None  # old events can't be replayed

    plan = {**PLAN, "hotel": {**PLAN["hotel"], "name": "The Broadway Hotel - London Heathrow"}}
    assert not RESULT_CHECKS["plan_avoids"](record(pauses=[review(plan=plan)]), "The Broadway Hotel - London Heathrow")[0]
