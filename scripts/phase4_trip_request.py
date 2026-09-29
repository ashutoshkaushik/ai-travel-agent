"""Phase 4: free text in, validated TripRequest out. Code decides when to ask.

    uv run python scripts/phase4_trip_request.py "Tokyo in December for 2 people, budget 4k"
    uv run python scripts/phase4_trip_request.py "Tokyo in December for 2, 4k" \
        --answer "Flying from San Francisco, Dec 6 to Dec 12"
    uv run python scripts/phase4_trip_request.py --graph
"""

import argparse
import sys

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from travel_planner.agents.trip_intake import build_trip_intake
from travel_planner.models import TripRequest
from travel_planner.trace import RunStats, run_with_interrupts

agent = build_trip_intake(checkpointer=InMemorySaver())


def main() -> None:
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--answer", action="append", default=[], help="scripted answer to an interrupt (repeatable)")
    args = parser.parse_args()

    config = {"configurable": {"thread_id": "intake-1"}}
    stats = RunStats()
    print(f"🧳 {args.request}")
    run_with_interrupts(agent, {"messages": [HumanMessage(args.request)], "ask_rounds": 0}, config, args.answer, stats)

    final = agent.get_state(config).values
    print(f"\n{'=' * 80}")
    if final.get("request"):
        request = TripRequest(**final["request"])
        print(f"✅ TripRequest: {request!r}")
        print(f"   nights={request.nights}, destination_airport={request.destination_airport}")
    else:
        print("❌ No valid TripRequest")
    print(f"📊 {stats.summary()}")


if __name__ == "__main__":
    main()
