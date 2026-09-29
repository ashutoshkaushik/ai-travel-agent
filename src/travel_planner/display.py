"""Human-readable renderings of plans and costs, shared by review questions, chat messages and scripts.

Day numbers come from itinerary.build_itinerary(), the same code that draws the day-by-day columns,
so the chat and the itinerary can never disagree about which day something happens.
"""

from langchain.messages import AIMessage, HumanMessage

from travel_planner.itinerary import build_itinerary, price_label
from travel_planner.models import CostBreakdown, TripPlan, TripRequest


def describe_plan(plan: TripPlan, costs: CostBreakdown, request: TripRequest | None = None,
                  flight_detail: dict | None = None) -> str:
    lines = [
        f"✈️  {plan.flight.airline} ${plan.flight.price_total:,.2f} ({'nonstop' if plan.flight.nonstop else 'with stops'}, test-mode price)",
        f"🏨 {plan.hotel.name}: ${plan.hotel.price_per_night:,.0f}/night, ${plan.hotel.total_price:,.0f} total, rated {plan.hotel.rating}",
    ]
    if request is not None:
        for day in build_itinerary(request, plan, flight_detail):
            sights = [f"{e.title} ({e.detail})" for e in day.events if e.kind == "sight"]
            if sights:
                lines.append(f"📍 Day {day.number} ({day.label}): " + "; ".join(sights))
    else:  # no trip context: fall back to the plan's own relative day numbers
        for s in sorted(plan.sights, key=lambda s: s.day):
            lines.append(f"📍 Stop {s.day}: {s.name} ({price_label(s)})")
    lines.append(describe_costs(costs))
    return "\n".join(lines)


def describe_costs(costs: CostBreakdown) -> str:
    verdict = f"over by ${-costs.remaining:,.2f}" if costs.over_budget else f"${costs.remaining:,.2f} left"
    return (
        f"💵 Flight ${costs.flight:,.2f} + hotel ${costs.hotel:,.2f} + sights ${costs.sights:,.2f} "
        f"+ food ~${costs.food_estimate:,.0f} = ${costs.total:,.2f} of ${costs.budget:,} ({verdict})"
    )


def decision_messages(question: str, answer: str) -> list:
    """A human decision, as a question/answer pair tagged 'decision' so any UI can show it distinctly
    (and the router can forward it from a sub-agent into the main conversation)."""
    return [AIMessage(question, additional_kwargs={"decision": "question"}),
            HumanMessage(answer, additional_kwargs={"decision": "answer"})]


def is_decision(message) -> bool:
    return bool(getattr(message, "additional_kwargs", {}).get("decision"))
