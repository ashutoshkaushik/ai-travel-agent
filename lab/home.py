"""'Start here': the one-image idea of the whole app, then where to go next."""

import json
from pathlib import Path

import streamlit as st

from lab.nav import go_button
from ui.text import safe_html

RECORDING = Path(__file__).resolve().parent / "recordings" / "1-plan-a-trip-to-paris-from-jfk--november.json"
ROW_SECONDS = 1.5  # pace of the "Watch it work" replay: ~15 steps ≈ 25 seconds
PLAIN = {"load_profile": "no saved profile yet: a first-time traveler", "input_guard": "passed the input guardrail"}

LADDER = [
    ("Plain LLM", "prompt → answer",
     "Writes fluent text from what it learned in training.",
     "Can't see today's prices, your documents, or do anything. Guesses."),
    ("RAG", "retrieve → prompt → answer",
     "Looks things up in your documents first, then answers with citations.",
     "Still one shot: it can't search again, compute, or act."),
    ("Tool calling", "prompt → one tool → answer",
     "Can call a real API once: a flight search, a weather lookup.",
     "One step. Can't chain steps where each depends on the last."),
    ("Agent", "reason ⇄ act, in a loop",
     "Decides the next step from each result, as many times as needed, and asks you when stuck.",
     "One generalist with every tool: easy to pick the wrong one, hard to guarantee rules."),
    ("Multi-agent system", "router → specialists + code + you",
     "A router sends each request to a focused agent. Code enforces budget and evidence. You approve every booking.",
     "This app."),
]


def ladder_html() -> str:
    rungs = []
    for i, (title, flow, adds, cant) in enumerate(LADDER, start=1):
        this = i == len(LADDER)
        rungs.append(
            f"<div class='rung{' this' if this else ''}'>"
            + ("<span class='here badge agent'>this app</span>" if this else "")
            + f"<div class='n'>{i}</div><div class='t'>{title}</div><div class='flow'>{flow}</div>"
            f"<div class='adds'>{adds}</div>"
            + ("" if this else f"<div class='cant'>Limit: {cant}</div>")
            + "</div>")
    return f"<div class='ladder'>{''.join(rungs)}</div>"


def watch_it_work_html(replay_count: int = 0) -> str:
    """The first recorded turn of the Paris example, as rows that appear one by one (CSS only, no API calls)."""
    recording = json.loads(RECORDING.read_text())
    turn, request = recording["turns"][0], recording["example"]
    rows = [f"<div class='ww-row ww-user' style='animation-delay:0s'><span class='badge human'>you</span>"
            f"<span>{safe_html(request)}</span></div>"]
    events = [e for e in turn["events"] if e["key"] not in ("intake", "planner", "research")] or turn["events"]
    for i, e in enumerate(events, start=1):
        rows.append(f"<div class='ww-row' style='animation-delay:{i * ROW_SECONDS:.1f}s'>"
                    f"<span class='badge {e['actor']}'>{e['actor']}</span><span class='trace-node'>{safe_html(e['node'])}"
                    f"</span><span class='trace-what'>{safe_html(PLAIN.get(e['key'], e['what'])[:120])}</span></div>")
    costs = (turn.get("pending") or {}).get("costs")
    if costs:
        rows.append(f"<div class='ww-row ww-done' style='animation-delay:{(len(events) + 1) * ROW_SECONDS:.1f}s'>"
                    f"<b>Plan ready for your approval:</b> {safe_html(f'${costs['total']:,.0f} of ${costs['budget']:,}')} "
                    f"(flight {safe_html(f'${costs['flight']:,.0f}')}, hotel {safe_html(f'${costs['hotel']:,.0f}')}, "
                    "sights and food). Every price checked against the search results.</div>")
    return f"<div class='watch' data-run='{replay_count}'>{''.join(rows)}</div>"


def home_page() -> None:
    st.markdown("<div class='eyebrow'>A multi-agent travel planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero'>A travel agent that plans, checks its own work, and asks before it books</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='lede'>Tell it where you want to go. Agents research real flights, hotels and sights; "
                "plain code checks the budget and verifies every price; you choose the options and approve each "
                "booking. Every step is visible, and the Agent Lab explains how each piece was built.</div>",
                unsafe_allow_html=True)

    st.markdown("#### Watch it work")
    st.caption("A recorded run of the real agents (no API calls): the classifier routes, agents search, code checks "
               "the budget and the evidence, and it pauses for you.")
    count = st.session_state.get("watch_runs", 0)
    st.markdown(watch_it_work_html(count), unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 2])
    with a:
        go_button("tools", "Start the tour", primary=True, button_key="home_tour")
    with b:
        if st.button("Watch again", icon=":material/replay:", width="stretch"):
            st.session_state.watch_runs = count + 1
            st.rerun()

    st.markdown("#### From an LLM to an agent system, in five steps")
    st.markdown(ladder_html(), unsafe_allow_html=True)
    go_button("compare", "See all five levels run on the same request", button_key="home_compare")

    st.markdown("#### Who does what in this app")
    st.markdown(
        "<div class='steps'>"
        "<div class='agent'><b>Agents decide</b><span>Classify your request, extract trip details, choose flights, "
        "hotels and sights, answer questions from the travel guides.</span></div>"
        "<div class='code'><b>Code guarantees</b><span>Validates the request, does all the budget math, verifies every "
        "price against the search results, caps retries, blocks bookings without approval.</span></div>"
        "<div class='human'><b>You decide</b><span>Answer missing details, pick flight and hotel options, raise the "
        "budget or not, and approve, edit or reject each booking.</span></div>"
        "</div>", unsafe_allow_html=True)

    st.markdown("")
    a, b, c = st.columns(3)
    with a:
        go_button("assistant", "Plan a trip", primary=True)
    with b:
        go_button("tools", "Tour the Agent Lab")
    with c:
        go_button("lessons", "What broke, and how it was fixed")
