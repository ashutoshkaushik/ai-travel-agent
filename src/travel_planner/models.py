"""Typed trip data. Two models on purpose:

- TripRequestDraft: what the LLM extracts. Every field optional, nothing validated, so the LLM
  can say "I don't know yet" (null) instead of guessing.
- TripRequest: what the rest of the system trusts. Built only by Python validation, so every
  simplifying assumption is enforced in code, not by prompt.

"The LLM extracts, the code validates."
"""

from datetime import date

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from travel_planner.cities import NEW_YORK_AIRPORTS, SUPPORTED_CITIES, US_AIRPORTS, find_city
from travel_planner.config import today

MIN_NIGHTS, MAX_NIGHTS = 2, 14
MAX_ADULTS = 4


class TripRequestDraft(BaseModel):
    """Trip details the traveler has stated so far. Use null for anything not clearly stated."""

    origin: str | None = Field(
        None, description="Departure airport IATA code. Convert a US city to its main airport (San Francisco -> SFO, Chicago -> ORD). Null if not stated."
    )
    city: str | None = Field(None, description="Destination city exactly as the traveler named it, e.g. 'Tokyo' or 'Dubai'. Null if not stated.")
    depart_date: str | None = Field(
        None, description="Departure date as YYYY-MM-DD. Only if a specific day was given; a month alone is not enough. If no year was given, use the next future occurrence."
    )
    return_date: str | None = Field(
        None, description="Return date as YYYY-MM-DD. If the traveler gave a trip length instead (e.g. '6 nights'), compute it from depart_date. Null if unknown."
    )
    adults: int | None = Field(None, description="Number of adult travelers. Null if not stated.")
    budget_usd: int | None = Field(None, description="Total budget in USD for the whole trip and all travelers. Null if not stated.")


class TripRequest(BaseModel):
    """A fully validated trip. If this object exists, every assumption holds."""

    origin: str
    city: str
    depart_date: date
    return_date: date
    adults: int = Field(ge=1, le=MAX_ADULTS)
    budget_usd: int = Field(gt=0)

    @field_validator("origin")
    @classmethod
    def origin_is_supported_us_airport(cls, v: str) -> str:
        code = v.strip().upper()
        if code not in US_AIRPORTS:
            raise ValueError(f"{v!r} is not a supported US departure airport. Supported: {', '.join(sorted(US_AIRPORTS))}")
        return code

    @field_validator("city")
    @classmethod
    def city_is_supported(cls, v: str) -> str:
        canonical = find_city(v)
        if canonical is None:
            raise ValueError(f"{v} isn't a destination I can plan yet. I can plan trips to: {', '.join(SUPPORTED_CITIES)}")
        return canonical

    @field_validator("depart_date")
    @classmethod
    def departs_in_future(cls, v: date) -> date:
        if v <= today():
            raise ValueError(f"the departure date {v} is not in the future")
        return v

    @model_validator(mode="after")
    def trip_shape_is_supported(self) -> "TripRequest":
        nights = (self.return_date - self.depart_date).days
        if nights < MIN_NIGHTS or nights > MAX_NIGHTS:
            raise ValueError(f"the trip is {nights} nights; I can plan {MIN_NIGHTS} to {MAX_NIGHTS} nights")
        if self.city == "New York" and self.origin in NEW_YORK_AIRPORTS:
            raise ValueError("the departure airport is in New York, the same city as the destination")
        return self

    @property
    def nights(self) -> int:
        return (self.return_date - self.depart_date).days

    @property
    def destination_airport(self) -> str:
        return SUPPORTED_CITIES[self.city].airport

    def summary(self) -> str:
        people = "1 adult" if self.adults == 1 else f"{self.adults} adults"
        return (
            f"{self.origin} → {self.city} ({self.destination_airport}), "
            f"{self.depart_date:%b %d} – {self.return_date:%b %d, %Y} ({self.nights} nights), "
            f"{people}, budget ${self.budget_usd:,}"
        )


# ---------- Phase 5: what the research agent returns ----------
# The LLM only CHOOSES (ids, names, the prices it saw). It never adds anything up:
# cost_breakdown() does the arithmetic, and verify_plan() checks every choice against tool results.


class FlightChoice(BaseModel):
    offer_id: str = Field(description="offer_id exactly as returned by search_flights")
    airline: str
    price_total: float = Field(description="The offer's price for all travelers, exactly as returned")
    nonstop: bool


class HotelChoice(BaseModel):
    hotel_id: str = Field(description="hotel_id exactly as returned by search_hotels")
    name: str
    price_per_night: float
    total_price: float = Field(description="The hotel's total_price for the whole stay, exactly as returned")
    rating: float


class SightChoice(BaseModel):
    name: str = Field(description="Sight name exactly as returned by search_top_sights")
    day: int = Field(description="Trip day number, starting at 1 for the arrival day")
    price_usd: float | None = Field(description="Ticket price per person as returned; 0 if free, null if unknown")


class TripPlan(BaseModel):
    """The chosen flight, hotel, and sights for the trip. Copy ids, names, and prices exactly from tool results."""

    flight: FlightChoice
    hotel: HotelChoice
    sights: list[SightChoice] = Field(description="About 2 sights per full day; 1 or none on arrival and departure days")
    rationale: str = Field(description="Two or three sentences on why these choices fit the traveler")


class CostBreakdown(BaseModel):
    flight: float
    hotel: float
    sights: float
    food_estimate: float
    total: float
    budget: int
    remaining: float

    @property
    def over_budget(self) -> bool:
        return self.remaining < 0


def cost_breakdown(plan: TripPlan, request: TripRequest) -> CostBreakdown:
    """Pure Python arithmetic. Sight prices are per person; unknown prices count as $0."""
    food_rate = SUPPORTED_CITIES[request.city].food_usd_per_person_per_day
    flight = plan.flight.price_total
    hotel = plan.hotel.total_price
    sights = sum(s.price_usd or 0 for s in plan.sights) * request.adults
    food = food_rate * request.adults * request.nights
    total = round(flight + hotel + sights + food, 2)
    return CostBreakdown(
        flight=flight, hotel=hotel, sights=round(sights, 2), food_estimate=food,
        total=total, budget=request.budget_usd, remaining=round(request.budget_usd - total, 2),
    )


def verify_plan(plan: TripPlan, tool_results: list[dict]) -> list[str]:
    """Check every choice against what the tools actually returned. Returns a list of problems.

    This catches the "hallucinated output" failure mode: an id, name, or price the LLM invented
    or mis-copied. Empty list = every choice is backed by evidence.
    """
    offers, hotels, sights = {}, {}, {}
    for r in tool_results:
        for o in r.get("offers", []):
            offers[o["offer_id"]] = o
        for h in r.get("hotels", []):
            hotels[h["hotel_id"]] = h
        for s in r.get("sights", []):
            sights[s["name"]] = s

    problems = []
    offer = offers.get(plan.flight.offer_id)
    if offer is None:
        problems.append(f"Flight offer {plan.flight.offer_id} was never returned by search_flights.")
    elif abs(offer["price"] - plan.flight.price_total) > 0.01:
        problems.append(f"Flight price {plan.flight.price_total} doesn't match the offer's {offer['price']}.")

    hotel = hotels.get(plan.hotel.hotel_id)
    if hotel is None:
        problems.append(f"Hotel {plan.hotel.name!r} was never returned by search_hotels.")
    elif abs(hotel["total_price"] - plan.hotel.total_price) > 0.01:
        problems.append(f"Hotel total {plan.hotel.total_price} doesn't match the search result's {hotel['total_price']}.")

    for s in plan.sights:
        found = sights.get(s.name)
        if found is None:
            problems.append(f"Sight {s.name!r} was never returned by search_top_sights.")
        elif found.get("price_usd") is not None and s.price_usd != found["price_usd"]:
            problems.append(f"Sight {s.name!r} price {s.price_usd} doesn't match the search result's {found['price_usd']}.")
    return problems


FIELD_LABELS = {
    "origin": "your departure airport or US city",
    "city": "your destination city",
    "depart_date": "your exact departure date",
    "return_date": "your return date or number of nights",
    "adults": "how many adults are traveling",
    "budget_usd": "your total budget in USD",
}


def validate_draft(draft: TripRequestDraft) -> tuple[TripRequest | None, list[str]]:
    """Turn a draft into a TripRequest, or explain exactly what's missing or wrong.

    Returns (request, []) on success, or (None, problems) with traveler-readable problems.
    """
    missing = [FIELD_LABELS[f] for f in FIELD_LABELS if getattr(draft, f) is None]
    if missing:
        return None, [f"I still need {m}." for m in missing]

    try:
        return TripRequest(**draft.model_dump()), []
    except ValidationError as e:
        problems = []
        for err in e.errors():
            msg = err["msg"].removeprefix("Value error, ")
            field = err["loc"][0] if err["loc"] else None
            if err["type"] in ("date_from_datetime_parsing", "date_parsing"):
                msg = f"{FIELD_LABELS.get(field, field)} isn't a valid date"
            elif err["type"] in ("less_than_equal", "greater_than_equal") and field == "adults":
                msg = f"I can plan for 1 to {MAX_ADULTS} adults"
            elif err["type"] == "greater_than" and field == "budget_usd":
                msg = "the budget must be more than $0"
            problems.append(msg[0].upper() + msg[1:] + ("" if msg.endswith(".") else "."))
        return None, problems
