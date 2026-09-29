"""Phase 7: the booking agent. create_agent + HumanInTheLoopMiddleware.

The middleware sits between the model and the tools: when the model requests a write tool,
the graph pauses BEFORE the tool runs and asks a human to approve, edit, or reject each call.
Unlike ask_traveler (Phase 3) or the review node (Phase 6), no interrupt() appears in our code:
the gate is declared as configuration, so it can't be forgotten in one code path.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer

from travel_planner.itinerary import stay_dates
from travel_planner.llm import agent_middleware, primary_model
from travel_planner.models import TripPlan, TripRequest
from travel_planner.prompts import BOOKING_PROMPT
from travel_planner.tools.booking import book_flight, book_hotel

DECISIONS = ["approve", "edit", "reject"]


def describe_booking(tool_call: dict, state, runtime) -> str:
    """What the reviewer sees for each paused booking: the action, the money, and what it means."""
    a = tool_call["args"]
    if tool_call["name"] == "book_flight":
        return (f"Book flight: {a['airline']} for {a['passengers']} passenger(s), ${a['price_total']:,.2f} total "
                f"(offer {a['offer_id']}). Sandbox booking, no real charge.")
    return (f"Book hotel: {a['hotel_name']}, {a['check_in']} to {a['check_out']}, {a['guests']} guest(s), "
            f"${a['total_price']:,.2f} total. Sandbox booking, no real charge.")


def build_booking_agent(llm: BaseChatModel | None = None, checkpointer: Checkpointer = None):
    gate = {"allowed_decisions": DECISIONS, "description": describe_booking}
    return create_agent(
        llm or primary_model(),
        tools=[book_flight, book_hotel],
        system_prompt=BOOKING_PROMPT,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"book_flight": gate, "book_hotel": gate})]
                   + ([] if llm else agent_middleware()),
        checkpointer=checkpointer,  # the middleware pauses with interrupt(), so state must be saved
    )


def booking_request_message(plan: TripPlan, request: TripRequest, flight_detail: dict | None = None) -> HumanMessage:
    """Everything the agent needs to book, copied from the approved plan: no searching, no choosing.
    Hotel dates come from itinerary.stay_dates(): check in on the day the flight lands."""
    check_in, check_out = stay_dates(request, flight_detail)
    return HumanMessage(
        "Book this approved plan.\n"
        f"Flight: book_flight(offer_id={plan.flight.offer_id!r}, airline={plan.flight.airline!r}, "
        f"price_total={plan.flight.price_total}, passengers={request.adults})\n"
        f"Hotel: book_hotel(hotel_id={plan.hotel.hotel_id!r}, hotel_name={plan.hotel.name!r}, "
        f"check_in='{check_in}', check_out='{check_out}', guests={request.adults}, "
        f"total_price={plan.hotel.total_price})"
    )
