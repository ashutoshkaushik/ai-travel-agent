"""WRITE tools. Unlike every tool so far, these change state, so they only ever run behind
HumanInTheLoopMiddleware (see agents/booking_agent.py) and are idempotent."""

from langchain.tools import tool

from travel_planner.bookings import create_booking
from travel_planner.tools._common import ToolInputError, ok, parse_future_date, returns_errors_as_data


@tool(parse_docstring=True)
@returns_errors_as_data
def book_flight(offer_id: str, airline: str, price_total: float, passengers: int) -> dict:
    """Book a flight offer for the travelers. Sandbox booking: no real ticket or charge.

    Args:
        offer_id: The offer_id from the approved plan.
        airline: The airline name from the approved plan.
        price_total: The offer's total price for all passengers, in USD.
        passengers: Number of adult passengers (1-4).
    """
    if not 1 <= passengers <= 4:
        raise ToolInputError("passengers must be between 1 and 4.")
    result = create_booking(
        kind="flight",
        idempotency_key=f"flight:{offer_id}",  # the same offer can only be booked once
        summary=f"{airline} for {passengers} passenger(s)",
        amount_usd=price_total,
        details={"offer_id": offer_id, "airline": airline, "passengers": passengers},
    )
    return ok(**result, booked=f"{airline} flight, ${price_total:,.2f}", note="Sandbox booking: no real ticket or charge.")


@tool(parse_docstring=True)
@returns_errors_as_data
def book_hotel(hotel_id: str, hotel_name: str, check_in: str, check_out: str, guests: int, total_price: float) -> dict:
    """Book a hotel stay for the travelers. Sandbox booking: no real reservation or charge.

    Args:
        hotel_id: The hotel_id from the approved plan.
        hotel_name: The hotel name from the approved plan.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of adult guests (1-4).
        total_price: The total price for the whole stay, in USD.
    """
    start, end = parse_future_date(check_in, "check_in"), parse_future_date(check_out, "check_out")
    if end <= start:
        raise ToolInputError("check_out must be after check_in.")
    if not 1 <= guests <= 4:
        raise ToolInputError("guests must be between 1 and 4.")
    result = create_booking(
        kind="hotel",
        idempotency_key=f"hotel:{hotel_id}:{start}:{end}",  # same hotel + same dates = same booking
        summary=f"{hotel_name}, {start} to {end}, {guests} guest(s)",
        amount_usd=total_price,
        details={"hotel_id": hotel_id, "hotel_name": hotel_name, "check_in": str(start), "check_out": str(end), "guests": guests},
    )
    return ok(**result, booked=f"{hotel_name}, {start} to {end}", note="Sandbox booking: no real reservation or charge.")
