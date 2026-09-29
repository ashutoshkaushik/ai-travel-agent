"""Phase 6: the trip planner. Research agent + budget loop + escalation and review checkpoints.

    uv run python scripts/phase6_planner.py --budget 2500 --answer approve
    uv run python scripts/phase6_planner.py --budget 1000 --answer '$2,500' --answer approve
    uv run python scripts/phase6_planner.py --graph
"""

import argparse
import sys
from datetime import date, timedelta

from langgraph.checkpoint.memory import InMemorySaver

from travel_planner.agents.trip_planner import build_trip_planner, initial_state
from travel_planner.display import describe_plan
from travel_planner.models import CostBreakdown, TripPlan, TripRequest
from travel_planner.trace import RunStats, run_with_interrupts

planner = build_trip_planner(checkpointer=InMemorySaver())


def main() -> None:
    if "--graph" in sys.argv:
        print(planner.get_graph().draw_mermaid())
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Tokyo")
    parser.add_argument("--origin", default="SFO")
    parser.add_argument("--depart", default="2026-12-06")
    parser.add_argument("--nights", type=int, default=6)
    parser.add_argument("--adults", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--answer", action="append", default=[], help="scripted answer to an interrupt (repeatable)")
    args = parser.parse_args()

    depart = date.fromisoformat(args.depart)
    request = TripRequest(origin=args.origin, city=args.city, depart_date=depart,
                          return_date=depart + timedelta(days=args.nights), adults=args.adults, budget_usd=args.budget)
    print(f"🧳 {request.summary()}")

    config = {"configurable": {"thread_id": "planner-1"}}
    stats = RunStats()
    run_with_interrupts(planner, initial_state(request), config, args.answer, stats)

    final = planner.get_state(config).values
    print(f"\n{'=' * 80}\n🏁 Status: {final['status']}")
    if final["status"] == "approved":
        print(describe_plan(TripPlan(**final["plan"]), CostBreakdown(**final["costs"])))
    print(f"📊 {stats.summary()}")


if __name__ == "__main__":
    main()
