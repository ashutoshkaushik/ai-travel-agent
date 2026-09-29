"""Record the four example conversations for the Travel Assistant's "Replay a recorded run" mode.

    uv run python scripts/record_replays.py

Each recording is a list of turns. A turn holds exactly what the page draws: the timed trace
events, the chat messages, the trip state, and the pause (if any). Replays need no API keys and
cost nothing, so the app can be deployed publicly in replay mode. Recording costs about $0.01.

The traveler's side is scripted: answer a question, raise the budget when escalated, approve the
plan the agent picked, then (for the first example) say "book it" and stop at the booking gate.
"""

import json
import sys
import time
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.assistant_page import EXAMPLES, _answered_by, _summarize, paused_decisions  # noqa: E402
from travel_planner.agents.assistant import build_assistant  # noqa: E402
from travel_planner.display import is_decision  # noqa: E402
from travel_planner.trace import RunStats, pending_interrupt  # noqa: E402
from ui.graphs import actor_of  # noqa: E402

OUT = ROOT / "lab" / "recordings"
ANSWERS = {"question": "From SFO, December 10 to 16"}
MAX_TURNS = 5


def message_json(m) -> dict:
    return {"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content,
            "kwargs": {k: v for k, v in m.additional_kwargs.items() if k in ("decision", "notice")}}


def run_turn(graph, config: dict, payload, label: str) -> dict:
    stats, events = RunStats(), []
    start = previous = time.perf_counter()
    for namespace, chunk in graph.stream(payload, {**config, "callbacks": [stats]}, stream_mode="updates",
                                         subgraphs=True):
        now = time.perf_counter()
        for node, update in chunk.items():
            if node == "__interrupt__":
                if not events or events[-1]["node"] != "waiting for you":
                    events.append({"node": "waiting for you", "key": "waiting for you", "actor": "human",
                                   "what": "paused for your decision", "ms": 0})
                continue
            what = _summarize(update)
            if model := _answered_by(update):
                what = f"[answered by fallback model {model}] {what}"
            events.append({"node": " › ".join([p.split(":")[0] for p in namespace] + [node]), "key": node,
                           "actor": actor_of(node), "what": what, "ms": round((now - previous) * 1000)})
        previous = now
    values = graph.get_state(config).values
    shown = [m for m in values.get("messages", []) if isinstance(m, HumanMessage) or
             (isinstance(m, AIMessage) and m.content) or is_decision(m)]
    pending = pending_interrupt(graph, config)
    return {
        "label": label, "events": events, "summary": stats.summary(),
        "seconds": round(time.perf_counter() - start, 2), "cost_usd": round(stats.cost_usd, 5),
        "messages": [message_json(m) for m in shown],
        "paused_decisions": [message_json(m) for m in paused_decisions(graph, config, shown)] if pending else [],
        "state": {k: values.get(k) for k in ("request", "plan", "costs", "trip_status", "details")},
        "pending": pending,
    }


def next_step(pending: dict | None, turns: list, book: bool) -> tuple | None:
    """The scripted traveler: (payload, label) for the next turn, or None to stop."""
    if pending is None:
        approved = turns[-1]["state"].get("trip_status") == "approved"
        return ({"messages": [HumanMessage("book it")]}, "Working on: book it") if approved and book else None
    kind = pending.get("kind") or ("approval" if "action_requests" in pending else "question")
    if kind == "question":
        return Command(resume={"answer": ANSWERS["question"]}), "Continuing with your answer"
    if kind == "escalation":
        return Command(resume={"answer": f"${pending['suggested_budget']}"}), "Re-checking with the new budget"
    if kind == "review":
        plan = pending["plan"]
        selection = {"offer_id": plan["flight"]["offer_id"], "hotel_id": plan["hotel"]["hotel_id"]}
        return Command(resume={"answer": "approve", "selection": selection}), "Approving your plan"
    return None  # the booking gate: a replay never books anything


def record(text: str, book: bool) -> dict:
    graph = build_assistant(InMemorySaver(), InMemoryStore())
    config = {"configurable": {"thread_id": "replay", "user_id": "guest"}}
    short = text if len(text) <= 70 else text[:70].rstrip() + "…"
    turns = [run_turn(graph, config, {"messages": [HumanMessage(text)]}, f"Working on: {short}")]
    while len(turns) < MAX_TURNS and (step := next_step(turns[-1]["pending"], turns, book)):
        turns.append(run_turn(graph, config, *step))
    return {"example": text, "recorded": time.strftime("%Y-%m-%d"), "turns": turns}


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower())[:40].strip("-")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    examples = [t for texts in EXAMPLES.values() for t in texts][:4]
    for i, text in enumerate(examples):
        recording = record(text, book=(i == 0))
        path = OUT / f"{i + 1}-{slug(text)}.json"
        path.write_text(json.dumps(recording, indent=1, default=str))
        cost = sum(t["cost_usd"] for t in recording["turns"])
        print(f"{path.name}: {len(recording['turns'])} turns, ${cost:.4f}")


if __name__ == "__main__":
    main()
