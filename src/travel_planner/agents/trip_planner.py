"""Phase 6: the trip planner. An agent does the research; plain code checks, replans, and gates.

    START -> research -> budget_check --(fits)------------------------> review (interrupt)
                ^             |                                          |-- approve --> approved -> END
                |             |--(over, tries left)--> replan --+        +-- feedback --> research
                |             |--(bad evidence, tries left)-----+
                |             |--(over, no tries / can't fix)--> escalate (interrupt)
                |             |                                    |-- new budget --> budget_check
                |             |                                    |-- accept -----> review
                |             |                                    +-- cancel -----> cancelled -> END
                |             +--(bad evidence, no tries)--> failed -> END
                +--------------------------------------------- replan

The research agent (create_agent) decides WHAT to pick. Everything that must hold every time
(arithmetic, the retry limit, when to ask a human) is code: "LLM decides, code guarantees".
"""

import math
import operator
import re
from typing import Annotated, Callable, Literal, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Checkpointer, interrupt

from travel_planner.agents.research_agent import build_research_agent, research_request_message, tool_results_from
from travel_planner.display import decision_messages, describe_costs, describe_plan
from travel_planner.itinerary import fit_hotel_to_stay, stay_dates
from travel_planner.limits import RunLimitExceeded
from travel_planner.models import (
    CostBreakdown,
    FlightChoice,
    HotelChoice,
    TripPlan,
    TripRequest,
    cost_breakdown,
    verify_plan,
)

MAX_RETRIES = 2  # research attempts after the first, per planning round
MIN_HOTEL_PER_NIGHT = 40  # below this, a cheaper hotel can't realistically fix the budget
MAX_UNCLEAR_REPLIES = 2  # human loops need a stop condition too

# research_fn(request, constraints, config) -> (plan or None, tool results). Injectable for tests.
ResearchFn = Callable[[TripRequest, list[str], RunnableConfig], tuple[TripPlan | None, list[dict]]]


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # the traveler-facing log
    request: dict  # TripRequest as JSON
    plan: dict | None  # latest TripPlan as JSON
    costs: dict | None  # latest CostBreakdown as JSON
    evidence: Annotated[list[dict], operator.add]  # every tool result from every attempt (appended)
    problems: list[str]  # evidence problems from the latest research attempt
    constraints: list[str]  # hard constraints for the next research attempt
    retries: int  # retries used in this planning round
    over_budget_accepted: bool
    escalation: str | None  # the traveler's choice at the last escalation
    unclear_replies: int  # consecutive escalation replies we couldn't parse
    hotel_cap: int | None  # the nightly cap the last replan demanded (checked in code)
    keep_sights: list[dict] | None  # sights carried over by code in a hotel-only replan
    keep_flight: dict | None  # flight carried over by code in a hotel-only replan
    details: dict | None  # evidence for the approved choices (flight times, hotel coordinates) for the UI
    status: str  # "planning" | "approved" | "cancelled" | "failed"


def agent_research(request: TripRequest, constraints: list[str], config: RunnableConfig) -> tuple[TripPlan | None, list[dict]]:
    """Default research step: run the Phase 5 create_agent; return its plan and the tool results it saw."""
    agent = build_research_agent()
    result = agent.invoke({"messages": [research_request_message(request, constraints)]},
                          {**config, "recursion_limit": 20})
    return result.get("structured_response"), tool_results_from(result["messages"])


# ---------- pure helpers (unit-tested directly) ----------

def hotel_price_cap(plan: TripPlan, costs: CostBreakdown, nights: int) -> int | None:
    """The nightly hotel price that would bring the trip within budget, or None if that's unrealistic."""
    overshoot = -costs.remaining
    cap = math.floor(plan.hotel.price_per_night - overshoot / nights)
    return cap if cap >= MIN_HOTEL_PER_NIGHT else None


def parse_escalation(answer: str) -> tuple[str, int | None]:
    """Traveler reply to an escalation -> ("raise", new_budget) | ("accept", None) | ("cancel", None) | ("unclear", None)."""
    text = answer.strip().lower()
    amount = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", text)
    if amount:
        value = float(amount.group(1).replace(",", "")) * (1000 if amount.group(2) else 1)
        return "raise", int(value)
    words = set(re.findall(r"[a-z]+", text))  # whole words only: "know" must not match "no"
    if words & {"accept", "continue", "proceed", "fine"}:
        return "accept", None
    if words & {"cancel", "stop", "quit", "no"}:
        return "cancel", None
    return "unclear", None


APPROVE_WORDS = {"approve", "approved", "yes", "y", "ok", "okay", "looks good", "lgtm", "book it", "go"}


def parse_review(answer: str) -> tuple[str, str | None]:
    """Traveler reply to a plan review -> ("approve", None) or ("modify", feedback)."""
    text = answer.strip().lower().rstrip("!.")
    return ("approve", None) if text in APPROVE_WORDS else ("modify", answer.strip())


def flight_detail(state: dict, plan: TripPlan | None = None) -> dict | None:
    """The evidence for the plan's flight (times, slices), from every tool result so far."""
    plan = plan or TripPlan(**state["plan"])
    offers, _, _ = evidence_index(state.get("evidence", []))
    return offers.get(plan.flight.offer_id)


def stay_nights(state: dict, plan: TripPlan) -> int:
    """Nights the hotel is actually used (check-in on the arrival day), for the replan math."""
    check_in, check_out = stay_dates(TripRequest(**state["request"]), flight_detail(state, plan))
    return (check_out - check_in).days


def evidence_index(evidence: list[dict]) -> tuple[dict, dict, dict]:
    """offers, hotels, and sights from every tool result so far, keyed by id / name (latest wins)."""
    offers, hotels, sights = {}, {}, {}
    for r in evidence:
        offers.update({o["offer_id"]: o for o in r.get("offers", [])})
        hotels.update({h["hotel_id"]: h for h in r.get("hotels", [])})
        sights.update({s["name"]: s for s in r.get("sights", [])})
    return offers, hotels, sights


def review_options(evidence: list[dict], plan: TripPlan, flights: int = 5, hotels: int = 6) -> dict:
    """The alternatives the research actually found, cheapest first, always including the current pick."""
    offers, hotel_map, _ = evidence_index(evidence)

    def top(items: dict, key, chosen_id: str, n: int) -> list[dict]:
        ranked = sorted(items.values(), key=key)[:n]
        if chosen_id in items and all(chosen_id != (i.get("offer_id") or i.get("hotel_id")) for i in ranked):
            ranked.append(items[chosen_id])
        return ranked

    return {"flights": top(offers, lambda o: o.get("price", 0), plan.flight.offer_id, flights),
            "hotels": top(hotel_map, lambda h: h.get("price_per_night", 0), plan.hotel.hotel_id, hotels)}


def apply_selection(plan: TripPlan, selection: dict, evidence: list[dict]) -> TripPlan:
    """Rebuild the plan with the traveler's chosen flight and hotel, copied from the evidence by code."""
    offers, hotels, _ = evidence_index(evidence)
    update = {}
    if (offer := offers.get(selection.get("offer_id"))) is not None:
        update["flight"] = FlightChoice(offer_id=offer["offer_id"], airline=offer["airline"], price_total=offer["price"],
                                        nonstop=all(s["stops"] == 0 for s in offer.get("slices", [])))
    if (hotel := hotels.get(selection.get("hotel_id"))) is not None:
        update["hotel"] = HotelChoice(hotel_id=hotel["hotel_id"], name=hotel["name"], price_per_night=hotel["price_per_night"],
                                      total_price=hotel["total_price"], rating=hotel.get("rating") or 0)
    return plan.model_copy(update=update)


# ---------- the graph ----------

def build_trip_planner(research_fn: ResearchFn = agent_research, checkpointer: Checkpointer = None):

    def research(state: PlannerState, config: RunnableConfig) -> dict:
        request = TripRequest(**state["request"])
        try:
            plan, tool_results = research_fn(request, state.get("constraints", []), config)
        except RunLimitExceeded as e:
            # A clean, explained stop instead of a crash: the per-run token/$ cap was reached
            return {"status": "stopped", "messages": [AIMessage(
                f"⏹️ I stopped planning because {e}. Nothing was booked. Try a simpler request or a new session.",
                additional_kwargs={"notice": True})]}
        if plan and state.get("keep_sights") is not None:
            # Hotel-only replan: code carries the verified flight and sights over instead of trusting
            # the LLM to re-type them. In testing it zeroed sight prices, and once paired one
            # offer's price with a neighboring offer's id.
            plan = TripPlan(**{**plan.model_dump(), "flight": state["keep_flight"], "sights": state["keep_sights"]})
        # Verify against evidence from ALL attempts: items kept from an earlier plan are still backed
        problems = verify_plan(plan, state.get("evidence", []) + tool_results) if plan else ["The research agent did not return a plan."]
        attempt = state.get("retries", 0) + 1
        notes = []
        if plan is None:
            notes.append(f"🔍 Research attempt {attempt} failed: {' '.join(problems)}")
        else:
            notes.append(f"🔍 Research attempt {attempt}: {plan.flight.airline} ${plan.flight.price_total:,.2f}, "
                         f"{plan.hotel.name} ${plan.hotel.price_per_night:,.0f}/night, {len(plan.sights)} sights")
            if not problems:
                # The hotel was searched for the whole trip before the flight was known. Re-price it for
                # the nights actually used (check-in on the arrival day), after verification.
                offers, _, _ = evidence_index(state.get("evidence", []) + tool_results)
                plan, stay_note = fit_hotel_to_stay(plan, request, offers.get(plan.flight.offer_id))
                if stay_note:
                    notes.append(f"🛏️ {stay_note}")
        # A notice is something the traveler should see in the main chat (the router forwards it)
        return {"plan": plan.model_dump() if plan else None, "problems": problems, "evidence": tool_results,
                "messages": [AIMessage(n, additional_kwargs={"notice": True} if n.startswith("🛏️") else {}) for n in notes]}

    def budget_check(state: PlannerState) -> dict:
        """Pure Python: recompute costs every time (the budget may have been raised)."""
        if state["plan"] is None:
            return {"costs": None}
        costs = cost_breakdown(TripPlan(**state["plan"]), TripRequest(**state["request"]))
        return {"costs": costs.model_dump(), "messages": [AIMessage(describe_costs(costs))]}

    def route_after_budget(state: PlannerState) -> Literal["replan", "escalate", "review", "failed", "stopped"]:
        if state.get("status") == "stopped":
            return "stopped"
        tries_left = state.get("retries", 0) < MAX_RETRIES
        if state["problems"] or state["plan"] is None:
            return "replan" if tries_left else "failed"
        costs = CostBreakdown(**state["costs"])
        if not costs.over_budget or state.get("over_budget_accepted"):
            return "review"
        request, plan = TripRequest(**state["request"]), TripPlan(**state["plan"])
        if state.get("hotel_cap") and plan.hotel.price_per_night > state["hotel_cap"]:
            return "escalate"  # the agent couldn't find a hotel under the cap: retrying won't help
        if tries_left and hotel_price_cap(plan, costs, stay_nights(state, plan)) is not None:
            return "replan"
        return "escalate"

    def replan(state: PlannerState) -> dict:
        """Turn the failure into a concrete, checkable constraint for the next attempt."""
        if state["problems"] or state["plan"] is None:
            # Keep the constraints already in force (e.g. a hotel cap) and add the correction
            constraints = [c for c in state.get("constraints", []) if not c.startswith("Your previous plan used")]
            constraints += ["Your previous plan used ids, names, or prices that did not match the tool results. "
                            "Copy them exactly from tool results. Problems: " + " ".join(state["problems"])]
        else:
            request, plan, costs = TripRequest(**state["request"]), TripPlan(**state["plan"]), CostBreakdown(**state["costs"])
            cap = hotel_price_cap(plan, costs, stay_nights(state, plan))
            sights = "; ".join(f"{s.name} (day {s.day})" for s in plan.sights)
            # The traveler's own feedback stays in force: a budget replan adds a cap, it never replaces
            # what they asked for (in testing, "rated 4.5 or higher" was silently dropped here).
            feedback = [c for c in state.get("constraints", []) if c.startswith("Traveler feedback")]
            constraints = feedback + [
                f"The previous plan was ${-costs.remaining:,.0f} over budget. Change ONLY the hotel: choose one "
                f"with price_per_night of at most ${cap} (search with max_price_per_night={cap}) that still "
                f"satisfies any traveler feedback above. If none exists, return the best match anyway. "
                f"Keep flight offer {plan.flight.offer_id} and exactly these sights: {sights}."]
            return {"constraints": constraints, "hotel_cap": cap, "keep_sights": [s.model_dump() for s in plan.sights],
                    "keep_flight": plan.flight.model_dump(),
                    "retries": state.get("retries", 0) + 1,
                    "messages": [AIMessage(f"🔁 Replanning: hotel must be at most ${cap}/night; flight and sights unchanged.")]}
        return {"constraints": constraints, "retries": state.get("retries", 0) + 1,
                "messages": [AIMessage(f"🔁 Replanning: {constraints[0]}")]}

    def escalate(state: PlannerState) -> dict:
        costs = CostBreakdown(**state["costs"])
        suggested = math.ceil(costs.total / 100) * 100
        question = (
            f"I couldn't fit this trip in your ${costs.budget:,} budget. The best plan I found costs "
            f"${costs.total:,.2f} (${-costs.remaining:,.2f} over).\n"
            f"{describe_plan(TripPlan(**state['plan']), costs, TripRequest(**state['request']), flight_detail(state))}\n"
            f"Reply with a new budget (e.g. '${suggested:,}'), 'accept' to continue over budget, or 'cancel'."
        )
        # Everything above interrupt() re-runs on resume, so it must stay side-effect free.
        reply = interrupt({"question": question, "kind": "escalation", "costs": state["costs"],
                           "suggested_budget": suggested})["answer"]
        action, amount = parse_escalation(reply)
        short = (f"Over budget by ${-costs.remaining:,.2f} (the best plan costs ${costs.total:,.2f}, your budget is "
                 f"${costs.budget:,}). Raise the budget, continue over budget, or cancel?")
        answer = {"raise": f"Raised budget to ${amount:,}" if amount else "", "accept": "Continue over budget",
                  "cancel": "Cancel the trip"}.get(action, f"Unclear reply: {reply}")
        log = decision_messages(short, answer)
        if action == "raise":
            request = TripRequest(**state["request"]).model_copy(update={"budget_usd": amount})
            return {"request": request.model_dump(mode="json"), "escalation": action, "unclear_replies": 0,
                    "hotel_cap": None, "messages": log}
        if action == "accept":
            return {"over_budget_accepted": True, "escalation": action, "unclear_replies": 0, "messages": log}
        if action == "cancel":
            return {"status": "cancelled", "escalation": action, "messages": log}
        unclear = state.get("unclear_replies", 0) + 1
        if unclear > MAX_UNCLEAR_REPLIES:
            return {"status": "cancelled", "escalation": "cancel", "unclear_replies": unclear,
                    "messages": log + [AIMessage("I couldn't understand the replies, so I've stopped here.")]}
        return {"escalation": action, "unclear_replies": unclear,
                "messages": log + [AIMessage("I didn't catch that; let me ask again.")]}

    def route_after_escalate(state: PlannerState) -> Literal["budget_check", "review", "cancelled", "escalate"]:
        return {"raise": "budget_check",  # re-check the same plan against the new budget
                "accept": "review",
                "cancel": "cancelled"}.get(state["escalation"], "escalate")  # unclear: ask again

    def review(state: PlannerState) -> dict:
        plan, costs = TripPlan(**state["plan"]), CostBreakdown(**state["costs"])
        request = TripRequest(**state["request"])
        question = (f"Here's your trip plan:\n{describe_plan(plan, costs, request, flight_detail(state))}\n"
                    "Reply 'approve', or tell me what to change.")
        short = "Review the plan: approve it, pick other flight or hotel options, or ask for changes."
        # The payload carries everything a UI needs to let the traveler compare and pick options
        reply = interrupt({"question": question, "kind": "review", "request": state["request"],
                           "plan": state["plan"], "costs": state["costs"],
                           "options": review_options(state.get("evidence", []), plan)})
        if reply.get("selection"):
            # The traveler picked a flight and hotel from the options: code rebuilds the plan from the
            # evidence (so it is verified by construction), recomputes costs, and approves it.
            chosen = apply_selection(plan, reply["selection"], state.get("evidence", []))
            offers, _, _ = evidence_index(state.get("evidence", []))
            chosen, _ = fit_hotel_to_stay(chosen, request, offers.get(chosen.flight.offer_id))
            chosen_costs = cost_breakdown(chosen, request)
            answer = (f"Picked {chosen.flight.airline} (${chosen.flight.price_total:,.2f}) and {chosen.hotel.name} "
                      f"(${chosen.hotel.total_price:,.0f}); approved at ${chosen_costs.total:,.2f}")
            return {"plan": chosen.model_dump(), "costs": chosen_costs.model_dump(), "status": "approved",
                    "over_budget_accepted": chosen_costs.over_budget, "messages": decision_messages(short, answer)}
        decision, feedback = parse_review(reply["answer"])
        log = decision_messages(short, "Approved the plan" if decision == "approve" else f"Change request: {feedback}")
        if decision == "approve":
            return {"status": "approved", "messages": log}
        # A new planning round: the traveler's words become a hard constraint, retries reset
        return {"constraints": [f"Traveler feedback on the previous plan: {feedback}"], "retries": 0,
                "over_budget_accepted": False, "hotel_cap": None, "keep_sights": None, "keep_flight": None, "messages": log}

    def route_after_review(state: PlannerState) -> Literal["approved", "research"]:
        return "approved" if state.get("status") == "approved" else "research"

    def approved(state: PlannerState) -> dict:
        offers, hotels, _ = evidence_index(state.get("evidence", []))
        plan = TripPlan(**state["plan"])
        details = {"flight": offers.get(plan.flight.offer_id), "hotel": hotels.get(plan.hotel.hotel_id)}
        return {"details": details, "messages": [AIMessage("✅ Plan approved.")]}

    def cancelled(state: PlannerState) -> dict:
        return {"messages": [AIMessage("Trip planning cancelled.")]}

    def stopped(state: PlannerState) -> dict:
        return {}  # the research node already explained why

    def failed(state: PlannerState) -> dict:
        return {"status": "failed",
                "messages": [AIMessage(f"❌ I couldn't produce a reliable plan after {MAX_RETRIES + 1} attempts: {' '.join(state['problems'])}")]}

    graph = StateGraph(PlannerState)
    for name, fn in [("research", research), ("budget_check", budget_check), ("replan", replan),
                     ("escalate", escalate), ("review", review), ("approved", approved),
                     ("cancelled", cancelled), ("failed", failed), ("stopped", stopped)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "research")
    graph.add_edge("research", "budget_check")
    graph.add_conditional_edges("budget_check", route_after_budget, ["replan", "escalate", "review", "failed", "stopped"])
    graph.add_edge("replan", "research")
    graph.add_conditional_edges("escalate", route_after_escalate, ["budget_check", "review", "cancelled", "escalate"])
    graph.add_conditional_edges("review", route_after_review, ["approved", "research"])
    for terminal in ("approved", "cancelled", "failed", "stopped"):
        graph.add_edge(terminal, END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(request: TripRequest) -> dict:
    return {"request": request.model_dump(mode="json"), "plan": None, "costs": None, "problems": [],
            "evidence": [], "constraints": [], "retries": 0, "over_budget_accepted": False, "escalation": None,
            "unclear_replies": 0, "hotel_cap": None, "keep_sights": None, "keep_flight": None, "details": None,
            "status": "planning", "messages": []}
