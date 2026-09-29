"""Phase 7: book an approved plan behind HumanInTheLoopMiddleware (approve / edit / reject).

    uv run python scripts/phase7_booking.py --answer approve --answer "reject: too far from the sights"
    uv run python scripts/phase7_booking.py --answer approve --answer 'edit: {"guests": 3}'
    uv run python scripts/phase7_booking.py --ledger        # show everything booked so far

Reviewer replies: approve | reject: <reason> | edit: {json of changed fields}
"""

import argparse
import sys
from datetime import date, timedelta

from langgraph.checkpoint.memory import InMemorySaver

from travel_planner.agents.booking_agent import booking_request_message, build_booking_agent
from travel_planner.agents.research_agent import build_research_agent, research_request_message
from travel_planner.bookings import list_bookings
from travel_planner.models import TripRequest
from travel_planner.trace import RunStats, run_with_interrupts


def show_ledger() -> None:
    rows = list_bookings()
    print(f"📒 {len(rows)} booking(s) in the ledger")
    for b in rows:
        print(f"  {b['confirmation']}  {b['kind']:<6}  ${b['amount_usd']:>9,.2f}  {b['summary']}")


def main() -> None:
    if "--ledger" in sys.argv:
        show_ledger()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Tokyo")
    parser.add_argument("--origin", default="SFO")
    parser.add_argument("--depart", default="2026-12-06")
    parser.add_argument("--nights", type=int, default=6)
    parser.add_argument("--adults", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--answer", action="append", default=[], help="scripted reviewer reply (repeatable)")
    args = parser.parse_args()

    depart = date.fromisoformat(args.depart)
    request = TripRequest(origin=args.origin, city=args.city, depart_date=depart,
                          return_date=depart + timedelta(days=args.nights), adults=args.adults, budget_usd=args.budget)
    stats = RunStats()

    print(f"🔍 Researching a plan for {request.summary()} ...")
    plan = build_research_agent().invoke({"messages": [research_request_message(request)]},
                                         {"callbacks": [stats], "recursion_limit": 20})["structured_response"]
    print(f"   Plan: {plan.flight.airline} ${plan.flight.price_total:,.2f} + {plan.hotel.name} ${plan.hotel.total_price:,.0f}")

    print("\n🧾 Booking (each write pauses for your decision)")
    agent = build_booking_agent(checkpointer=InMemorySaver())
    run_with_interrupts(agent, {"messages": [booking_request_message(plan, request)]},
                        {"configurable": {"thread_id": "booking-1"}}, args.answer, stats)

    print(f"\n{'=' * 80}")
    show_ledger()
    print(f"📊 {stats.summary()}")


if __name__ == "__main__":
    main()
