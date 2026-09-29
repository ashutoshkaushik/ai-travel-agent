"""The Travel Assistant page: plan visually, pick options, approve, book, and ask questions.

Everything the traveler does here (the trip form, option pickers, approve / modify / book buttons)
turns into a message or a resume value for the SAME assistant graph the terminal uses
(agents/assistant.py). The page never plans anything itself; it only draws state and sends choices.
"""

import json
import os
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from travel_planner.agents.assistant import build_assistant, profile_namespace
from travel_planner.agents.trip_planner import apply_selection
from travel_planner.bookings import record_feedback
from travel_planner.cities import SUPPORTED_CITIES, US_AIRPORTS
from travel_planner.itinerary import build_itinerary, fit_hotel_to_stay
from travel_planner.models import CostBreakdown, TripPlan, TripRequest, cost_breakdown
from travel_planner.persistence import open_checkpointer, open_store
from travel_planner import guardrails, http_cache
from travel_planner.limits import DEFAULT_RUN_LIMITS, RunLimitExceeded, RunLimits, SessionBudget
from travel_planner.llm import PRIMARY_ATTEMPTS, PRIMARY_MODEL, simulate_outage
from travel_planner.trace import RunStats, pending_interrupt
from travel_planner.usage import record
from lab.nav import PAGES
from travel_planner.display import is_decision
from ui.graphs import actor_of
from ui.text import safe_html, safe_md
from ui.trip_views import cost_bar, itinerary_columns, trip_map

EXAMPLES = {
    "Simple": ["Plan a trip to Paris from JFK, November 5 to 10, 2 adults, budget 5000 dollars"],
    "Needs a question": ["Thinking of Tokyo sometime in December, 2 of us, around 2300 dollars"],
    "Over budget": ["Plan London from SFO, March 10 to 17 2027, 2 adults, budget 1500 dollars"],
    "Travel questions": ["Do I need a visa for Japan?", "Should I tip at restaurants in New York?"],
}

STATUS_BADGE = {"approved": ("ok", "Approved"), "booked": ("ok", "Booked"), "partially_booked": ("human", "Partly booked"),
                "cancelled": ("no", "Cancelled"), "failed": ("no", "Failed")}


# ------------------------------------------------------------------ state and running turns

@st.cache_resource
def get_assistant():
    """One compiled graph for all sessions; conversations are separated by thread_id."""
    return build_assistant(open_checkpointer(), open_store())


@st.cache_resource
def get_store():
    return open_store()


def _init_session() -> None:
    ss = st.session_state
    if "thread_id" not in ss:
        # The trip id lives in the URL too, so a reload (or a round trip to a Lab page) keeps the trip
        ss.thread_id = st.query_params.get("trip") or f"web-{uuid.uuid4().hex[:8]}"
    st.query_params["trip"] = ss.thread_id
    ss.setdefault("user_id", "guest")
    ss.setdefault("trace", [])
    ss.setdefault("turn", 0)
    ss.setdefault("budget", SessionBudget())
    ss.setdefault("break_it", set())


def _config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id, "user_id": st.session_state.user_id}}


def _trip_values() -> dict:
    return get_assistant().get_state(_config()).values


# ------------------------------------------------------------------ replay a recorded run

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
LIVE, REPLAY = "Live agents", "Replay a recorded run"


@st.cache_data
def load_recordings() -> list[dict]:
    """Recorded by scripts/record_replays.py: the four examples, turn by turn. No keys, no cost."""
    return [json.loads(p.read_text()) for p in sorted(RECORDINGS_DIR.glob("*.json"))]


def _replaying() -> bool:
    return st.session_state.get("mode") == REPLAY


def _as_message(m: dict):
    cls = HumanMessage if m["type"] == "human" else AIMessage
    return cls(m["content"], additional_kwargs=m.get("kwargs", {}))


def _replay_next() -> None:
    """Play the next recorded turn: the same trace rows the live run printed, paced from their real timings."""
    ss = st.session_state
    turns = load_recordings()[ss.replay_index]["turns"]
    if ss.replay_step >= len(turns):
        st.session_state.replay_notice = ("That's the end of this recording. Switch to Live agents to keep going "
                                          "with real searches.")
        st.rerun()
    turn = turns[ss.replay_step]
    with st.status(turn["label"], expanded=True) as status:
        for e in turn["events"]:
            time.sleep(min(0.7, max(0.08, e.get("ms", 0) / 1000 * 0.3)))
            status.update(label=safe_md(e["what"][:90]))
            status.write(safe_md(f"{e['what'][:110]}  ·  {e['node']}"))
        status.update(label=safe_md(f"Replayed · recorded in {turn['seconds']:.1f}s · {turn['summary']}"),
                      state="complete", expanded=False)
    ss.trace.append({k: turn[k] for k in ("label", "events", "summary", "seconds")})
    ss.replay_step += 1
    ss.turn += 1
    st.rerun()


def _start_replay(index: int) -> None:
    ss = st.session_state
    ss.replay_index, ss.replay_step, ss.trace = index, 0, []
    _replay_next()


def replay_body() -> None:
    ss = st.session_state
    recordings = load_recordings()
    names = [r["example"] for r in recordings]
    st.info("Replay mode: a recorded run of the real agents, step by step. No API calls and no cost. "
            "Buttons continue the recording; switch to Live agents to plan your own trip.", icon=":material/replay:")
    choice = st.selectbox("Recorded example", range(len(names)), format_func=lambda i: names[i],
                          index=ss.get("replay_index", 0))
    if st.button("Play" if choice != ss.get("replay_index") or not ss.get("replay_step") else "Restart",
                 type="primary", icon=":material/play_arrow:"):
        _start_replay(choice)
    if choice != ss.get("replay_index") or not ss.get("replay_step"):
        return
    turn = recordings[ss.replay_index]["turns"][ss.replay_step - 1]
    if notice := ss.pop("replay_notice", None):
        st.warning(notice, icon=":material/stop_circle:")
    conversation([_as_message(m) for m in turn["messages"] + turn["paused_decisions"]])
    state = turn["state"]
    if turn["pending"]:
        your_turn(turn["pending"])
    elif state.get("plan") and state.get("trip_status") in ("approved", "partially_booked", "booked"):
        trip_section(state)


# Which Agent Lab lesson explains each kind of step (by the node's own name)
LESSON = {
    "classify_intent": "router", "load_profile": "router", "save_profile": "router",
    "intake": "structured", "extract": "structured", "validate": "structured", "ask": "hitl", "confirm": "structured",
    "planner": "planner", "research": "planner", "budget_check": "planner", "replan": "planner",
    "escalate": "hitl", "review": "hitl", "approved": "planner", "model": "loop", "tools": "tools",
    "booking": "approval", "HumanInTheLoopMiddleware.after_model": "approval",
    "guide": "rag", "fallback": "router", "waiting for you": "hitl",
    "input_guard": "guardrails", "retry": "evals", "run cap": "guardrails",
}

TOOL_NOUN = {"search_flights": "flights", "search_hotels": "hotels", "search_top_sights": "sights",
             "get_weather": "the weather", "search_travel_guide": "the travel guides",
             "book_flight": "the flight", "book_hotel": "the hotel"}


def _progress(path: str, node: str, update: dict) -> str:
    """A plain-English line for the live progress view."""
    messages = (update or {}).get("messages", [])
    calls = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    if calls:
        verb = "Booking" if calls[0].startswith("book_") else "Searching"
        nouns = [TOOL_NOUN.get(c, c) for c in calls]
        return f"{verb} {', '.join(nouns[:-1]) + ' and ' + nouns[-1] if len(nouns) > 1 else nouns[0]}…"
    tool_results = [m.name for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        return "Got " + ", ".join(TOOL_NOUN.get(t, t) for t in tool_results)
    return {
        "load_profile": "Loading your saved profile", "classify_intent": "Understanding your request",
        "extract": "Reading the trip details", "validate": "Checking the details against the rules",
        "confirm": "Trip details confirmed", "research": "Drafted a plan", "budget_check": "Checked the budget",
        "replan": "Over budget: looking for a cheaper hotel…", "save_profile": "Saved your profile",
        "approved": "Plan approved", "guide": "Answered from the travel guides",
    }.get(node, _summarize(update))


def _summarize(update: dict) -> str:
    """One readable line for the trace: what this node did."""
    for m in reversed((update or {}).get("messages", [])):
        if isinstance(m, AIMessage) and m.tool_calls:
            return "requests " + ", ".join(c["name"] for c in m.tool_calls)
        if isinstance(m, ToolMessage):
            return f"{m.name} returned"
        if m.content:
            return str(m.content).splitlines()[0][:110]
    keys = [k for k in (update or {}) if k != "messages"]
    return ", ".join(f"{k} = {str(update[k])[:40]}" for k in keys[:2]) if keys else "done"


BREAK_IT = {
    "duffel_503": "Duffel returns 503 twice (watch the retries)",
    "serpapi_429": "SerpApi rate-limits once with 429",
    "model_outage": "Primary model fails twice (watch the fallback model answer)",
    "guards_off": "Guardrails off (the input, tool-output and output checks)",
    "tiny_budget": "Tiny run budget: 3,000 tokens (watch the clean stop)",
}


def _arm_break_it() -> RunLimits:
    """Apply the learner's 'Break it' switches for exactly one turn. Returns this turn's run limits."""
    flags = st.session_state.get("break_it", set())
    if "duffel_503" in flags:
        http_cache.inject_faults("duffel", 2, 503)
    if "serpapi_429" in flags:
        http_cache.inject_faults("serpapi", 1, 429)
    if "model_outage" in flags:
        simulate_outage(PRIMARY_ATTEMPTS)
    guardrails.GUARDS["on"] = "guards_off" not in flags
    return RunLimits(max_tokens=3_000) if "tiny_budget" in flags else DEFAULT_RUN_LIMITS


def _disarm_break_it() -> None:
    http_cache.clear_faults()
    simulate_outage(0)
    guardrails.GUARDS["on"] = True


def _drain_retries(events: list, status) -> None:
    """Show each HTTP retry (real or simulated) as its own trace row, as it happens."""
    while http_cache.RETRY_EVENTS:
        provider, attempt, why, delay, simulated = http_cache.RETRY_EVENTS.popleft()
        what = f"attempt {attempt} got {why}{' (simulated)' if simulated else ''}; retrying in {delay:.2f}s"
        events.append({"node": f"retry › {provider}", "key": "retry", "actor": "code", "what": what, "ms": 0})
        status.write(safe_md(f"Retrying {provider}: {what}"))


def _answered_by(update: dict) -> str | None:
    for m in (update or {}).get("messages", []):
        model = getattr(m, "response_metadata", {}).get("model_name", "")
        if model and not model.startswith(PRIMARY_MODEL):
            return model
    return None


def run_turn(payload, label: str) -> None:
    """Stream one turn of the graph, printing each step as it completes, and record a timed trace."""
    budget: SessionBudget = st.session_state.budget
    if reason := budget.blocked_reason():
        st.error(reason)
        return
    limits = _arm_break_it()
    stats, events, intent, stopped = RunStats(limits=limits), [], None, None
    start = previous = time.perf_counter()
    http_cache.RETRY_EVENTS.clear()
    try:
        with st.status(label, expanded=True) as status:
            try:
                for namespace, chunk in get_assistant().stream(payload, {**_config(), "callbacks": [stats]},
                                                               stream_mode="updates", subgraphs=True):
                    now = time.perf_counter()
                    _drain_retries(events, status)
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            # The pause surfaces once per graph level (sub-agent and router): record it once
                            if not events or events[-1]["node"] != "waiting for you":
                                events.append({"node": "waiting for you", "key": "waiting for you", "actor": "human",
                                               "what": "paused for your decision", "ms": 0})
                                status.write("Waiting for you")
                            continue
                        intent = (update or {}).get("intent") or intent
                        path = " › ".join([p.split(":")[0] for p in namespace] + [node])
                        line = _progress(path, node, update)
                        what = _summarize(update)
                        if model := _answered_by(update):
                            what = f"[answered by fallback model {model}] {what}"
                        ms = round((now - previous) * 1000)
                        events.append({"node": path, "key": node, "actor": actor_of(node), "what": what, "ms": ms})
                        record("node", thread=st.session_state.thread_id, node=node, ms=ms)
                        status.update(label=line)
                        status.write(safe_md(f"{line}  ·  {path}"))
                    previous = now
            except RunLimitExceeded as e:
                stopped = str(e)
                events.append({"node": "run cap", "key": "run cap", "actor": "code", "what": f"stopped: {e}", "ms": 0})
            seconds = time.perf_counter() - start
            done = (f"Stopped at the run cap in {seconds:.1f}s" if stopped else f"Done in {seconds:.1f}s")
            status.update(label=safe_md(f"{done} · {stats.llm_calls} LLM calls · {stats.tool_calls} tool calls · "
                                        f"${stats.cost_usd:.4f}"), state="error" if stopped else "complete", expanded=False)
    finally:
        _disarm_break_it()
    planner_ran = any(e["key"] in ("planner", "research") for e in events)
    budget.add(stats.cost_usd, planner_ran)
    record("turn", thread=st.session_state.thread_id, intent=intent, llm_calls=stats.llm_calls,
           tokens=stats.input_tokens + stats.output_tokens, cost_usd=round(stats.cost_usd, 6), seconds=round(seconds, 2),
           stopped=bool(stopped))
    st.session_state.trace.append({"label": label, "events": events, "summary": stats.summary(), "seconds": seconds})
    if stopped:
        st.session_state.stop_notice = (f"I stopped because {stopped}. Nothing was booked. Try a simpler request, "
                                        "or turn off 'Tiny run budget' in Break it mode.")
    st.session_state.turn += 1
    st.rerun()


def send_message(text: str) -> None:
    if _replaying():
        _replay_next()
    short = text if len(text) <= 70 else text[:70].rstrip() + "…"
    run_turn({"messages": [HumanMessage(text)]}, f"Working on: {short}")


def resume(value: dict, label: str) -> None:
    if _replaying():
        _replay_next()  # a replay follows the recorded choice, whichever button was pressed
    run_turn(Command(resume=value), label)


# ------------------------------------------------------------------ the "your turn" panels

def _flight_label(o: dict) -> str:
    out = (o.get("slices") or [{}])[0]
    stops = "nonstop" if all(s.get("stops", 0) == 0 for s in o.get("slices", [])) else "with stops"
    depart = out.get("depart", "")[11:16]
    return safe_md(f"{o['airline']} · ${o['price']:,.2f} · {stops}" + (f" · departs {depart}" if depart else ""))


def _hotel_label(h: dict) -> str:
    rating = f" · {h['rating']}★" if h.get("rating") else ""
    return safe_md(f"{h['name']} · ${h['price_per_night']:,.0f}/night{rating}")


def review_panel(p: dict) -> None:
    request, plan = TripRequest(**p["request"]), TripPlan(**p["plan"])
    flights, hotels = p["options"]["flights"], p["options"]["hotels"]
    st.markdown("<div class='turn-head'>Your turn: review the plan</div>", unsafe_allow_html=True)
    st.caption(safe_md(f"{request.summary()}. The agent picked the highlighted options; choose others to compare. "
                       "Costs update as you choose (computed in code, no LLM)."))

    left, right = st.columns(2)
    with left:
        offer_ids = [o["offer_id"] for o in flights]
        flight_id = st.radio("Flight (test-mode prices, total for all travelers)", offer_ids,
                             index=offer_ids.index(plan.flight.offer_id) if plan.flight.offer_id in offer_ids else 0,
                             format_func=lambda i: _flight_label(next(o for o in flights if o["offer_id"] == i)),
                             key=f"flight_{st.session_state.turn}")
    with right:
        hotel_ids = [h["hotel_id"] for h in hotels]
        hotel_id = st.radio("Hotel (whole stay)", hotel_ids,
                            index=hotel_ids.index(plan.hotel.hotel_id) if plan.hotel.hotel_id in hotel_ids else 0,
                            format_func=lambda i: _hotel_label(next(h for h in hotels if h["hotel_id"] == i)),
                            key=f"hotel_{st.session_state.turn}")

    selection = {"offer_id": flight_id, "hotel_id": hotel_id}
    flight_detail = next((o for o in flights if o["offer_id"] == flight_id), None)
    hotel_detail = next((h for h in hotels if h["hotel_id"] == hotel_id), None)
    preview = apply_selection(plan, selection, [{"offers": flights}, {"hotels": hotels}])
    # Same re-pricing the planner applies on approval: check in on the day this flight lands
    preview, stay_note = fit_hotel_to_stay(preview, request, flight_detail)
    costs = cost_breakdown(preview, request)
    cost_bar(costs)
    if stay_note:
        st.caption(safe_md(stay_note))
    days_tab, map_tab = st.tabs(["Day by day", "Map"])
    with days_tab:
        itinerary_columns(build_itinerary(request, preview, flight_detail))
    with map_tab:
        trip_map(preview, request.city, hotel_detail)

    a, b = st.columns([1, 2])
    with a:
        label = "Approve this plan" + (" (over budget)" if costs.over_budget else "")
        if st.button(label, type="primary", width="stretch", key=f"approve_{st.session_state.turn}"):
            resume({"answer": "approve", "selection": selection}, "Approving your plan")
    with b:
        change = st.text_input("Or ask the agent to change something", placeholder="e.g. a hotel closer to the center",
                               key=f"change_{st.session_state.turn}")
        if st.button("Ask for changes", width="stretch", key=f"modify_{st.session_state.turn}", disabled=not change):
            resume({"answer": change}, f"Replanning: {change[:60]}")


def escalation_panel(p: dict) -> None:
    costs = CostBreakdown(**p["costs"])
    st.markdown("<div class='turn-head'>Your turn: the budget doesn't fit</div>", unsafe_allow_html=True)
    st.write(safe_md(f"The best plan the agent found costs **${costs.total:,.0f}**, "
                     f"${-costs.remaining:,.0f} over your ${costs.budget:,} budget. A cheaper hotel can't close the gap."))
    cost_bar(costs)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button(safe_md(f"Raise budget to ${p['suggested_budget']:,}"), type="primary", width="stretch"):
            resume({"answer": f"${p['suggested_budget']}"}, "Re-checking with the new budget")
    with c2:
        custom = st.number_input("Or a different budget", min_value=100, step=100, value=int(p["suggested_budget"]))
        if st.button("Use this budget", width="stretch"):
            resume({"answer": f"${custom}"}, "Re-checking with the new budget")
    with c3:
        if st.button("Continue over budget", width="stretch"):
            resume({"answer": "accept"}, "Continuing over budget")
    with c4:
        if st.button("Cancel this trip", width="stretch"):
            resume({"answer": "cancel"}, "Cancelling")


def booking_panel(p: dict) -> None:
    st.markdown("<div class='turn-head'>Your turn: approve each booking</div>", unsafe_allow_html=True)
    st.caption("Nothing has been booked yet: the agent is paused before each write. Bookings are sandbox only.")
    decisions, missing_reason = [], False
    for i, action in enumerate(p["action_requests"]):
        with st.container(border=True):
            st.markdown(f"**{safe_md(action['description'])}**")
            choice = st.radio("Decision", ["Approve", "Edit", "Reject"], horizontal=True,
                              key=f"dec_{st.session_state.turn}_{i}", label_visibility="collapsed")
            if choice == "Approve":
                decisions.append({"type": "approve"})
            elif choice == "Edit":
                field = "passengers" if action["name"] == "book_flight" else "guests"
                value = st.number_input(f"Change {field}", 1, 4, int(action["args"][field]), key=f"edit_{st.session_state.turn}_{i}")
                decisions.append({"type": "edit", "edited_action": {"name": action["name"], "args": {**action["args"], field: value}}})
            else:
                reason = st.text_input("Why? (required, saved as feedback)", key=f"why_{st.session_state.turn}_{i}")
                missing_reason |= not reason.strip()
                decisions.append({"type": "reject", "message": reason.strip()})
    if st.button("Submit decisions", type="primary", disabled=missing_reason):
        for action, d in zip(p["action_requests"], decisions):
            if d["type"] == "reject" and not _replaying():
                record_feedback(action, d["message"], _trip_values().get("request"))
        resume({"decisions": decisions}, "Booking what you approved")
    if missing_reason:
        st.caption("Add a reason for each rejection to submit.")


def question_panel(p: dict) -> None:
    st.markdown("<div class='turn-head'>Your turn: the agent needs a detail</div>", unsafe_allow_html=True)
    st.markdown(safe_html(p["question"]).replace("\n", "<br>"), unsafe_allow_html=True)
    answer = st.text_input("Your answer", key=f"answer_{st.session_state.turn}", placeholder="e.g. From SFO, December 6 to 12")
    if st.button("Send", type="primary", disabled=not answer):
        resume({"answer": answer}, "Continuing with your answer")


def your_turn(p: dict) -> None:
    with st.container(border=True, key="yourturn"):
        if "action_requests" in p:
            booking_panel(p)
        elif p.get("kind") == "review":
            review_panel(p)
        elif p.get("kind") == "escalation":
            escalation_panel(p)
        else:
            question_panel(p)


# ------------------------------------------------------------------ page sections

def plan_form(expanded: bool) -> None:
    with st.expander("Plan a trip", expanded=expanded, icon=":material/flight_takeoff:"):
        default_start = date.today() + timedelta(days=70)
        c1, c2 = st.columns(2)
        origin = c1.selectbox("From", sorted(US_AIRPORTS), index=sorted(US_AIRPORTS).index("SFO"),
                              format_func=lambda c: f"{c} · {US_AIRPORTS[c]}")
        city = c2.selectbox("To", list(SUPPORTED_CITIES))
        c3, c4, c5 = st.columns([2, 1, 1])
        dates = c3.date_input("Dates", (default_start, default_start + timedelta(days=6)), min_value=date.today())
        adults = c4.number_input("Travelers", 1, 4, 2)
        budget = c5.number_input("Budget (USD)", 500, 30000, 4000, step=100)
        if st.button("Plan my trip", type="primary", disabled=len(dates) != 2):
            start, end = dates
            send_message(f"Plan a trip from {origin} to {city}, {start:%B %-d %Y} to {end:%B %-d %Y}, "
                         f"{adults} adult{'s' if adults > 1 else ''}, budget {budget} dollars")
        st.caption("This becomes a normal message, so the same agents handle it as if you had typed it.")


def trip_section(state: dict) -> None:
    request, plan, costs = TripRequest(**state["request"]), TripPlan(**state["plan"]), CostBreakdown(**state["costs"])
    details = state.get("details") or {}
    kind, text = STATUS_BADGE.get(state["trip_status"], ("code", state["trip_status"]))
    st.markdown(f"### Your trip <span class='badge {kind}'>{text}</span>", unsafe_allow_html=True)
    st.caption(safe_md(request.summary()))
    days_tab, map_tab, cost_tab = st.tabs(["Day by day", "Map", "Costs"])
    with days_tab:
        itinerary_columns(build_itinerary(request, plan, details.get("flight")))
    with map_tab:
        trip_map(plan, request.city, details.get("hotel"))
    with cost_tab:
        cost_bar(costs)
        st.caption(safe_md(plan.rationale))
    if state["trip_status"] in ("approved", "partially_booked"):
        if st.button("Book this trip", type="primary", icon=":material/check_circle:"):
            send_message("book it")


def paused_decisions(assistant, config: dict, shown: list) -> list:
    """Decisions and notices still inside a paused sub-agent (e.g. the budget you raised before the plan
    review). The router forwards them to the main chat when the sub-agent finishes; until then, read
    them from the sub-agent's saved state so the chat always explains what you already decided."""
    seen = {(m.type, m.content) for m in shown}
    found, stack = [], [assistant.get_state(config, subgraphs=True)]
    while stack:
        snapshot = stack.pop()
        for task in snapshot.tasks:
            if task.state is not None and hasattr(task.state, "values"):
                for m in task.state.values.get("messages", []):
                    if (is_decision(m) or m.additional_kwargs.get("notice")) and (m.type, m.content) not in seen:
                        seen.add((m.type, m.content))
                        found.append(m)
                stack.append(task.state)
    return found


def conversation(messages: list) -> None:
    for i, m in enumerate(messages):
        if is_decision(m):
            asked = m.additional_kwargs["decision"] == "question"
            with st.container(key=f"decision_{i}"):
                with st.chat_message("assistant" if asked else "user",
                                     avatar=":material/front_hand:" if asked else ":material/how_to_reg:"):
                    st.markdown(f"<div class='decision-label'>{'The agent asked you' if asked else 'You decided'}</div>",
                                unsafe_allow_html=True)
                    st.markdown(safe_md(m.content))
        elif isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.markdown(safe_md(m.content))
        elif isinstance(m, AIMessage) and m.content:
            with st.chat_message("assistant"):
                st.markdown(safe_md(m.content))


def trace_view() -> None:
    if not st.session_state.trace:
        return
    with st.expander("What the agents did (live trace)", icon=":material/timeline:"):
        a, b = st.columns([3, 1])
        a.markdown("<span class='badge agent'>agent: the LLM decides</span><span class='badge code'>code: guarantees"
                   "</span><span class='badge human'>human: you decide</span>", unsafe_allow_html=True)
        newest_first = b.toggle("Newest first", value=False, key="trace_newest_first")
        turns = st.session_state.trace[-6:]
        for t, turn in enumerate(reversed(turns) if newest_first else turns):
            st.markdown(f"<div class='trace-turn'><b>{safe_html(turn['label'])}</b> <span class='muted'>· "
                        f"{turn.get('seconds', 0):.1f}s · {safe_html(turn['summary'])}</span></div>",
                        unsafe_allow_html=True)
            for e_i, e in enumerate(turn["events"]):
                c1, c2, c3, c4, c5 = st.columns([1.1, 3.2, 4.4, 0.9, 1.0], vertical_alignment="center")
                c1.markdown(f"<span class='badge {e['actor']}'>{e['actor']}</span>", unsafe_allow_html=True)
                c2.markdown(f"<span class='trace-node'>{safe_html(e['node'])}</span>", unsafe_allow_html=True)
                c3.markdown(f"<span class='trace-what'>{safe_html(e['what'])}</span>", unsafe_allow_html=True)
                c4.markdown(f"<span class='trace-ms'>{e.get('ms', 0):,} ms</span>", unsafe_allow_html=True)
                lesson = LESSON.get(e.get("key", ""))
                if lesson in PAGES:
                    c5.page_link(PAGES[lesson], label="lesson", icon=":material/school:")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Traveler")
        name = st.text_input("Your name (for the saved profile)", st.session_state.user_id)
        if name.strip() and name.strip() != st.session_state.user_id:
            st.session_state.user_id = name.strip()
        item = get_store().get(profile_namespace(_config()), "travel")
        if item:
            v = item.value
            st.caption(f"Remembered: home airport {v.get('home_airport')}, usually {v.get('usual_travelers')} "
                       f"traveler(s), {v.get('trips_planned', 0)} trip(s) planned, last: {v.get('last_destination')}.")
        else:
            st.caption("No saved profile yet. It's written after you approve a plan.")
        if st.button("Start a new trip", icon=":material/add:", width="stretch"):
            st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
            st.session_state.trace = []
            st.rerun()
        if st.session_state.get("mode") != REPLAY:  # replays make no calls: nothing to break or budget
            with st.expander("Break it (learning mode)", icon=":material/science:"):
                st.caption("Each switch applies to your next message only, then resets. Watch the live trace.")
                chosen = set()
                for key, label in BREAK_IT.items():
                    if st.checkbox(label, key=f"brk_{key}"):
                        chosen.add(key)
                st.session_state.break_it = chosen
            b = st.session_state.budget
            st.caption(safe_md(f"This session: {b.planner_runs}/{b.max_planner_runs} trip plans, "
                               f"${b.spend_usd:.4f} of ${b.max_spend_usd:.2f} LLM budget."))
        st.divider()
        st.markdown("### Try an example")
        for group, texts in EXAMPLES.items():
            st.markdown(f"<div class='section-label'>{group}</div>", unsafe_allow_html=True)
            for t in texts:
                if st.button(t, key=f"ex_{t}", width="stretch"):
                    st.session_state.pending_text = t
                    st.rerun()


def assistant_page() -> None:
    _init_session()
    sidebar()
    st.title("Travel Assistant")
    st.markdown("<div class='lede'>Plan a trip with the form or in your own words. Agents research flights, hotels "
                "and sights; code checks the budget and the evidence; you approve before anything is booked.</div>",
                unsafe_allow_html=True)

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    picked = st.segmented_control("Mode", [LIVE, REPLAY], default=LIVE if has_key else REPLAY, key="mode_pick",
                                  label_visibility="collapsed")
    mode = REPLAY if not has_key else (picked or st.session_state.get("mode", LIVE))
    if mode != st.session_state.get("mode"):
        st.session_state.trace = []  # live and replayed traces don't mix
    st.session_state.mode = mode
    if not has_key:
        st.caption("No OpenAI key is configured here, so live agents are off: replays only.")
    if mode == REPLAY:
        if text := st.session_state.pop("pending_text", None):  # a sidebar example: play its recording
            names = [r["example"] for r in load_recordings()]
            if text in names:
                _start_replay(names.index(text))
        replay_body()
        trace_view()
        st.chat_input("Replay mode: switch to Live agents to type", disabled=True)
        return

    if text := st.session_state.pop("pending_text", None):
        send_message(text)

    assistant, config = get_assistant(), _config()
    state = assistant.get_state(config).values
    pending = pending_interrupt(assistant, config)

    if notice := st.session_state.pop("stop_notice", None):
        st.warning(safe_md(notice), icon=":material/block:")
    shown = state.get("messages", [])
    conversation(shown + (paused_decisions(assistant, config, shown) if pending else []))
    if pending:
        your_turn(pending)
    elif state.get("plan") and state.get("trip_status") in ("approved", "partially_booked", "booked"):
        trip_section(state)
    plan_form(expanded=not state.get("messages"))
    trace_view()

    if prompt := st.chat_input("Plan a trip, change it, say 'book it', or ask a travel question"):
        send_message(prompt)
