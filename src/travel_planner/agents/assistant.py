"""Phase 9: the travel assistant. One front door, the router pattern.

    START -> load_profile -> input_guard -> classify_intent --plan_trip--------> intake -> planner -> save_profile -> END
                                            --book (approved)---> booking ---------------------------> END
                                            --travel_question---> guide -----------------------------> END
                                            --other / no plan---> fallback --------------------------> END

- The classifier is ONE cheap structured-output call. It routes; it doesn't do the work.
- Each specialist is a sub-agent built in an earlier phase, called from a node. Their interrupts
  (clarifying questions, plan review, booking approval) bubble up through this graph.
- ONE checkpointer, at this graph: sub-agents inherit it, so every pause resumes correctly,
  even after the process restarts (SqliteSaver).
- Memory, all three kinds: working (this turn's context), session (the checkpointed thread),
  and long-term (the traveler profile in the store: read on entry, written after approval).
"""

from typing import Annotated, Literal, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field

from travel_planner.agents.booking_agent import booking_request_message, build_booking_agent
from travel_planner.agents.guide_agent import CityName, build_guide_agent, final_answer_text
from travel_planner.agents.trip_intake import build_trip_intake
from travel_planner.agents.trip_planner import build_trip_planner, initial_state
from travel_planner.cities import SUPPORTED_CITIES
from travel_planner.llm import structured_model
from travel_planner.display import decision_messages, describe_plan, is_decision
from travel_planner.guardrails import MONEY, check_input, guard_output
from travel_planner.models import CostBreakdown, TripPlan, TripRequest
from travel_planner.prompts import CLASSIFIER_PROMPT

Intent = Literal["plan_trip", "book", "travel_question", "other"]


class IntentClassification(BaseModel):
    """The single best intent for the traveler's latest message."""

    intent: Intent
    city: CityName | None = Field(description="Supported destination city named in the message, else null")
    reasoning: str = Field(description="One short sentence explaining the choice")


class AssistantState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str | None
    intent_city: str | None
    profile: dict  # long-term memory, loaded on entry
    request: dict | None  # current trip (TripRequest JSON)
    plan: dict | None
    costs: dict | None
    trip_status: str | None  # None | approved | partially_booked | booked | cancelled | failed | stopped
    details: dict | None  # evidence for the approved flight/hotel (times, coordinates) for itinerary and map
    guard: str | None  # why the input guardrail blocked the latest message (None = allowed)


def profile_namespace(config: RunnableConfig) -> tuple[str, str]:
    return ("profiles", config["configurable"].get("user_id", "default"))


def trip_context(state: AssistantState) -> str:
    if not state.get("request"):
        return "none yet"
    return f"{TripRequest(**state['request']).summary()} (status: {state.get('trip_status') or 'planning'})"


def forwarded(messages: list) -> list:
    """What a sub-agent's run should add to the main conversation: every human decision (question +
    answer) and every notice, in order, without duplicates. Internal steps stay in the sub-agent."""
    seen, out = set(), []
    for m in messages:
        if (is_decision(m) or m.additional_kwargs.get("notice")) and (m.type, m.content) not in seen:
            seen.add((m.type, m.content))
            out.append(m)
    return out


def booking_decisions(result: dict) -> list:
    """Turn the approval middleware's outcome into a readable decision pair for the chat."""
    calls = {c["id"]: c for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls}
    edited = result.get("hitl_edited_tool_calls") or {}
    lines = []
    for m in result["messages"]:
        if not (isinstance(m, ToolMessage) and m.name in ("book_flight", "book_hotel")):
            continue
        what = "flight" if m.name == "book_flight" else "hotel"
        if m.status == "error" and "rejected" in m.content:
            lines.append(f"Rejected the {what}: {m.content.split('reason:', 1)[-1].strip()}")
        elif m.tool_call_id in edited:
            lines.append(f"Edited and approved the {what}")
        else:
            lines.append(f"Approved the {what}" + (" booking" if '"status": "ok"' in m.content else ""))
    if not lines or not calls:
        return []
    return decision_messages("Approve each booking before anything is written?", "; ".join(lines))


def allowed_amounts(state: AssistantState) -> set[float]:
    """Every dollar amount a reply may mention: the traveler's own numbers, the plan's (tool-verified)
    prices, and code-computed costs. Anything else is an unverified price."""
    amounts = {float(a.replace(",", "")) for m in state["messages"] if isinstance(m, HumanMessage)
               for a in MONEY.findall(m.content)}
    if state.get("request"):
        amounts.add(float(state["request"]["budget_usd"]))
    plan = state.get("plan")
    if plan:
        amounts |= {plan["flight"]["price_total"], plan["hotel"]["price_per_night"], plan["hotel"]["total_price"]}
        amounts |= {s["price_usd"] for s in plan["sights"] if s.get("price_usd") is not None}
    if state.get("costs"):
        amounts |= {abs(v) for v in state["costs"].values() if isinstance(v, (int, float))}
    return amounts


def output_checked(state: AssistantState, message: AIMessage) -> list:
    """Output guardrail: redact PII the user never gave and prices nothing backs; say so if anything changed."""
    user_text = " ".join(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    text, notes = guard_output(message.content, user_text, allowed_amounts(state))
    if not notes:
        return [message]
    return [AIMessage(text), AIMessage("🛡️ Guardrail: " + "; ".join(sorted(set(notes))) + ".",
                                       additional_kwargs={"notice": True})]


def latest_human_text(state: AssistantState) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def build_assistant(
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
    *,
    classifier: Runnable | None = None,
    intake=None,
    planner=None,
    booking=None,
    guide=None,
):
    """All components are injectable so tests can run the whole router with no API calls.
    Sub-agents are compiled WITHOUT a checkpointer: they inherit this graph's."""
    if classifier is None:
        classifier = structured_model(IntentClassification)  # retry + fallback model (llm.py)
    intake = intake or build_trip_intake()
    planner = planner or build_trip_planner()
    booking = booking or build_booking_agent()
    guide = guide or build_guide_agent()

    def load_profile(state: AssistantState, config: RunnableConfig, runtime: Runtime) -> dict:
        item = runtime.store.get(profile_namespace(config), "travel") if runtime.store else None
        return {"profile": item.value if item else {}}

    def input_guard(state: AssistantState) -> dict:
        """Code check before any LLM sees the message (guardrails.check_input)."""
        request = state.get("request") or {}
        approved = state.get("trip_status") in ("approved", "partially_booked")
        return {"guard": check_input(latest_human_text(state), request.get("city"), approved)}

    def route_guard(state: AssistantState) -> Literal["classify_intent", "fallback"]:
        return "fallback" if state.get("guard") else "classify_intent"

    def classify_intent(state: AssistantState) -> dict:
        prompt = CLASSIFIER_PROMPT.format(cities=", ".join(SUPPORTED_CITIES), trip_context=trip_context(state))
        result = classifier.invoke([SystemMessage(prompt), HumanMessage(latest_human_text(state))])
        return {"intent": result.intent, "intent_city": result.city}

    def route_intent(state: AssistantState) -> Literal["intake", "booking", "guide", "fallback"]:
        if state["intent"] == "plan_trip":
            return "intake"
        if state["intent"] == "book":
            bookable = state.get("trip_status") in ("approved", "partially_booked")
            return "booking" if bookable else "fallback"  # precondition enforced in code
        if state["intent"] == "travel_question":
            return "guide"
        return "fallback"

    def run_intake(state: AssistantState, config: RunnableConfig) -> dict:
        # Give the extractor what we already know, so "make it 3 nights" or a returning traveler
        # doesn't have to repeat everything. Long-term memory, used.
        context = []
        if state.get("request"):
            context.append(f"Current trip: {TripRequest(**state['request']).summary()}.")
        profile = state.get("profile") or {}
        if profile.get("home_airport"):
            context.append(f"My saved profile: home airport {profile['home_airport']}, "
                           f"usually {profile.get('usual_travelers', 1)} traveler(s).")
        text = " ".join(context + [latest_human_text(state)])
        result = intake.invoke({"messages": [HumanMessage(text)], "ask_rounds": 0}, config)
        return {"request": result.get("request"), "messages": forwarded(result["messages"]) + [result["messages"][-1]],
                "plan": None, "costs": None, "trip_status": None if result.get("request") else "cancelled"}

    def route_after_intake(state: AssistantState) -> Literal["planner", "__end__"]:
        return "planner" if state.get("request") else END

    def run_planner(state: AssistantState, config: RunnableConfig) -> dict:
        result = planner.invoke(initial_state(TripRequest(**state["request"])), config)
        update = {"request": result["request"], "plan": result["plan"], "costs": result["costs"],
                  "trip_status": result["status"], "details": result.get("details")}
        decisions = forwarded(result["messages"])
        if result["status"] == "approved":
            flight = (result.get("details") or {}).get("flight")
            summary = describe_plan(TripPlan(**result["plan"]), CostBreakdown(**result["costs"]),
                                    TripRequest(**result["request"]), flight)
            approved_msg = AIMessage(f"✅ Plan approved:\n{summary}\nSay 'book it' when you're ready.")
            update["messages"] = decisions + output_checked({**state, **update}, approved_msg)
        else:
            tail = result["messages"][-1]
            update["messages"] = decisions + ([] if tail in decisions else [tail])  # no duplicate stop notice
        return update

    def save_profile(state: AssistantState, config: RunnableConfig, runtime: Runtime) -> dict:
        """Write back only high-signal, stable facts, and only after the traveler approved a plan."""
        if state.get("trip_status") != "approved" or not runtime.store:
            return {}
        request = TripRequest(**state["request"])
        profile = {**state.get("profile", {}), "home_airport": request.origin, "usual_travelers": request.adults,
                   "last_destination": request.city,
                   "trips_planned": state.get("profile", {}).get("trips_planned", 0) + 1}
        runtime.store.put(profile_namespace(config), "travel", profile)
        return {"profile": profile}

    def run_booking(state: AssistantState, config: RunnableConfig) -> dict:
        plan, request = TripPlan(**state["plan"]), TripRequest(**state["request"])
        flight = (state.get("details") or {}).get("flight")
        result = booking.invoke({"messages": [booking_request_message(plan, request, flight)]}, config)
        booked = {m.name for m in result["messages"]
                  if isinstance(m, ToolMessage) and m.name in ("book_flight", "book_hotel") and '"status": "ok"' in m.content}
        status = "booked" if len(booked) == 2 else ("partially_booked" if booked else state["trip_status"])
        return {"messages": booking_decisions(result) + output_checked(state, result["messages"][-1]), "trip_status": status}

    def run_guide(state: AssistantState, config: RunnableConfig) -> dict:
        city = state.get("intent_city") or (state["request"]["city"] if state.get("request") else None)
        if city is None:
            return {"messages": [AIMessage(f"Which city is this about? I have guides for {', '.join(SUPPORTED_CITIES)}.")]}
        result = guide.invoke({"messages": [HumanMessage(f"[City: {city}] {latest_human_text(state)}")]}, config)
        answer = result["structured_response"]
        sources = f"\n📖 Sources: {', '.join(answer.citations)}" if answer.citations else ""
        return {"messages": output_checked(state, AIMessage(final_answer_text(answer) + sources))}

    def fallback(state: AssistantState) -> dict:
        if state.get("guard"):
            return {"messages": [AIMessage(f"🛡️ {state['guard']}")]}
        if state["intent"] == "book":
            text = ("There's no approved plan to book yet. Tell me where and when you'd like to go, "
                    "and I'll plan it first.")
        else:
            text = ("I can plan trips to " + ", ".join(SUPPORTED_CITIES) + ", book an approved plan, "
                    "and answer practical travel questions. Try: 'Plan 5 nights in Rome for 2 in November, $3,000'.")
        return {"messages": [AIMessage(text)]}

    graph = StateGraph(AssistantState)
    for name, fn in [("load_profile", load_profile), ("input_guard", input_guard),
                     ("classify_intent", classify_intent), ("intake", run_intake),
                     ("planner", run_planner), ("save_profile", save_profile), ("booking", run_booking),
                     ("guide", run_guide), ("fallback", fallback)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "load_profile")
    graph.add_edge("load_profile", "input_guard")
    graph.add_conditional_edges("input_guard", route_guard, ["classify_intent", "fallback"])
    graph.add_conditional_edges("classify_intent", route_intent, ["intake", "booking", "guide", "fallback"])
    graph.add_conditional_edges("intake", route_after_intake, ["planner", END])
    graph.add_edge("planner", "save_profile")
    for terminal in ("save_profile", "booking", "guide", "fallback"):
        graph.add_edge(terminal, END)
    return graph.compile(checkpointer=checkpointer, store=store)
