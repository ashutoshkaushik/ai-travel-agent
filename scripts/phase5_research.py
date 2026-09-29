"""Phase 5: create_agent research agent. TripRequest in, verified TripPlan + Python-computed costs out.

    uv run python scripts/phase5_research.py
    uv run python scripts/phase5_research.py --city Rome --origin ORD --depart 2026-11-20 --nights 7 --budget 5000
    uv run python scripts/phase5_research.py --graph     # the graph create_agent built for us
"""

import argparse
import sys
from datetime import date, timedelta

from travel_planner.agents.research_agent import build_research_agent, research_request_message, tool_results_from
from travel_planner.models import TripRequest, cost_breakdown, verify_plan
from travel_planner.trace import RunStats, print_update

agent = build_research_agent()


def main() -> None:
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Tokyo")
    parser.add_argument("--origin", default="SFO")
    parser.add_argument("--depart", default="2026-12-06")
    parser.add_argument("--nights", type=int, default=6)
    parser.add_argument("--adults", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4000)
    args = parser.parse_args()

    depart = date.fromisoformat(args.depart)
    request = TripRequest(origin=args.origin, city=args.city, depart_date=depart,
                          return_date=depart + timedelta(days=args.nights), adults=args.adults, budget_usd=args.budget)
    print(f"🧳 {request.summary()}")

    stats = RunStats()
    final_state = {}
    for chunk in agent.stream({"messages": [research_request_message(request)]},
                              config={"recursion_limit": 20, "callbacks": [stats]}, stream_mode="updates"):
        for node_name, update in chunk.items():
            update = update or {}
            print_update(node_name, {k: v for k, v in update.items() if k != "structured_response"}, stats)
            final_state.setdefault("messages", []).extend(update.get("messages", []))
            if "structured_response" in update:
                final_state["structured_response"] = update["structured_response"]

    plan = final_state["structured_response"]
    print(f"\n{'=' * 80}\n🗺️  TripPlan (chosen by the LLM)")
    print(f"  ✈️  {plan.flight.airline}  ${plan.flight.price_total:,.2f}  {'nonstop' if plan.flight.nonstop else 'with stops'}")
    print(f"  🏨 {plan.hotel.name}  ${plan.hotel.price_per_night}/night, ${plan.hotel.total_price:,} total, rating {plan.hotel.rating}")
    for s in sorted(plan.sights, key=lambda s: s.day):
        price = "free" if s.price_usd == 0 else ("price unknown" if s.price_usd is None else f"${s.price_usd}/person")
        print(f"  📍 Day {s.day}: {s.name} ({price})")
    print(f"  💭 {plan.rationale}")

    costs = cost_breakdown(plan, request)
    print("\n💵 Cost breakdown (computed in Python)")
    for label, value in [("Flight", costs.flight), ("Hotel", costs.hotel), ("Sights", costs.sights), ("Food (est.)", costs.food_estimate)]:
        print(f"  {label:12} ${value:>9,.2f}")
    print(f"  {'TOTAL':12} ${costs.total:>9,.2f}  of ${costs.budget:,} budget  ->  "
          + (f"❌ over by ${-costs.remaining:,.2f}" if costs.over_budget else f"✅ ${costs.remaining:,.2f} left"))

    problems = verify_plan(plan, tool_results_from(final_state["messages"]))
    print("\n🔎 Evidence check: " + ("✅ every id, name, and price matches a tool result" if not problems else ""))
    for p in problems:
        print(f"  ⚠️  {p}")
    print(f"\n📊 {stats.summary()}")


if __name__ == "__main__":
    main()
