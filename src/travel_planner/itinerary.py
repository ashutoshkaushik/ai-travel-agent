"""The single source of truth for WHEN things happen on a trip. Pure Python (no LLM).

Everything that shows or books dates goes through here, so the chat summary, the day-by-day
columns, the hotel booking and the budget all agree:
- stay_dates(): check in on the day the outbound flight lands, check out when the flight home leaves.
- build_itinerary(): one Day per calendar day; the plan's sight "day 1" is the first day at the
  destination (the arrival day), so its calendar day number is computed here, not by the LLM.
- fit_hotel_to_stay(): re-prices the hotel for the nights actually used.
- price_label(): one wording for sight prices everywhere ("free (usually)" for known free sights).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from travel_planner.models import SightChoice, TripPlan, TripRequest

EVENING_HOUR = 17  # departures at or after 17:00 that land the next day count as overnight flights

# Sights that are free to visit, used when the sights API returns no price. Lowercase substrings.
USUALLY_FREE = (
    "hyde park", "regent's park", "st james's park", "trafalgar square", "british museum", "national gallery",
    "tate modern", "buckingham palace (outside)", "sensō-ji", "senso-ji", "meiji jingu", "meiji shrine",
    "yoyogi park", "ueno park", "tsukiji outer market", "shibuya crossing", "fushimi inari", "trevi fountain",
    "spanish steps", "piazza navona", "pantheon", "villa borghese", "notre-dame", "sacré-cœur", "sacre-coeur",
    "jardin du luxembourg", "luxembourg gardens", "champs-élysées", "central park", "times square",
    "brooklyn bridge", "high line", "gardens by the bay", "merlion", "marina bay", "la rambla",
    "park güell (free zone)", "barceloneta",
)


def price_label(sight: SightChoice) -> str:
    if sight.price_usd == 0:
        return "free"
    if sight.price_usd is not None:
        return f"${sight.price_usd:g}/person"
    name = sight.name.lower()
    return "free (usually)" if any(free in name for free in USUALLY_FREE) else "price unknown"


@dataclass
class Event:
    kind: str  # flight | checkin | checkout | sight | free
    title: str
    detail: str = ""


@dataclass
class Day:
    number: int
    date: date
    events: list[Event] = field(default_factory=list)
    stay: str | None = None  # where you sleep that night (None on the last day)

    @property
    def label(self) -> str:
        return self.date.strftime("%a %b %-d")


def _parse(ts: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def _legs(flight_detail: dict | None) -> tuple[dict | None, dict | None]:
    slices = (flight_detail or {}).get("slices", [])
    return (slices[0] if slices else None), (slices[1] if len(slices) > 1 else None)


def stay_dates(request: TripRequest, flight_detail: dict | None) -> tuple[date, date]:
    """Check-in = the day the outbound flight lands; check-out = the day the flight home leaves."""
    outbound, inbound = _legs(flight_detail)
    arrive = _parse(outbound.get("arrive")) if outbound else None
    leave = _parse(inbound.get("depart")) if inbound else None
    check_in = arrive.date() if arrive else request.depart_date
    check_out = leave.date() if leave else request.return_date
    if check_out <= check_in:  # defensive: never produce an empty or negative stay
        check_in, check_out = request.depart_date, request.return_date
    return check_in, check_out


def fit_hotel_to_stay(plan: TripPlan, request: TripRequest, flight_detail: dict | None) -> tuple[TripPlan, str | None]:
    """Re-price the hotel for the nights actually used. Hotels are searched for the whole trip
    before the flight is chosen; if the flight lands the next day, one night would be paid for and
    never used. Returns the adjusted plan and a traveler-readable note (None if nothing changed)."""
    check_in, check_out = stay_dates(request, flight_detail)
    nights = (check_out - check_in).days
    if nights == request.nights:
        return plan, None
    old = plan.hotel.total_price
    new = round(old * nights / request.nights, 2)
    arrive = _parse(_legs(flight_detail)[0].get("arrive"))
    note = (f"Hotel check-in moved to {check_in:%a %b %-d}: your flight lands at {arrive:%H:%M} that day, so the "
            f"hotel is {nights} nights instead of {request.nights} (${old - new:,.0f} less).")
    return plan.model_copy(update={"hotel": plan.hotel.model_copy(update={"total_price": new})}), note


def _flight_event(airline: str, leg: dict | None, fallback_route: str) -> Event:
    if not leg:
        return Event("flight", f"Flight {fallback_route}", airline)
    depart, arrive = _parse(leg.get("depart")), _parse(leg.get("arrive"))
    stops = "nonstop" if leg.get("stops", 0) == 0 else f"{leg['stops']} stop(s)"
    detail = f"{airline} · {stops}"
    if depart and arrive:
        next_day = f" ({arrive:%a %b %-d})" if arrive.date() != depart.date() else ""
        detail += f" · departs {depart:%H:%M}, arrives {arrive:%H:%M}{next_day}"
    return Event("flight", f"Flight {leg.get('from', '')} → {leg.get('to', '')}", detail)


def _in_transit_label(outbound: dict | None) -> str:
    """Only an evening departure that lands the next day is an 'overnight flight'."""
    depart = _parse(outbound.get("depart")) if outbound else None
    arrive = _parse(outbound.get("arrive")) if outbound else None
    if depart and arrive and depart.hour >= EVENING_HOUR and arrive.date() > depart.date():
        return "On the overnight flight"
    return "Travelling"


def build_itinerary(request: TripRequest, plan: TripPlan, flight_detail: dict | None = None) -> list[Day]:
    outbound, inbound = _legs(flight_detail)
    check_in, check_out = stay_dates(request, flight_detail)

    days = [Day(i + 1, request.depart_date + timedelta(days=i)) for i in range(request.nights + 1)]
    by_date = {d.date: d for d in days}
    first, last = days[0], days[-1]

    first.events.append(_flight_event(plan.flight.airline, outbound, f"{request.origin} → {request.destination_airport}"))
    nights = (check_out - check_in).days
    by_date.get(check_in, first).events.append(
        Event("checkin", f"Check in: {plan.hotel.name}", f"{nights} night{'s' if nights != 1 else ''} booked"))

    for day in days[:-1]:
        day.stay = plan.hotel.name if day.date >= check_in else _in_transit_label(outbound)

    last_sight_day = days[-2] if len(days) > 1 else last
    for sight in sorted(plan.sights, key=lambda s: s.day):
        target = by_date.get(check_in + timedelta(days=sight.day - 1))  # plan day 1 = arrival day
        if target is None or target is last:
            target = last_sight_day  # never schedule sights on the flight-home day
        target.events.append(Event("sight", sight.name, price_label(sight)))

    last.events.append(Event("checkout", f"Check out: {plan.hotel.name}"))
    last.events.append(_flight_event(plan.flight.airline, inbound, f"{request.destination_airport} → {request.origin}"))

    for day in days:
        if not any(e.kind in ("sight", "flight") for e in day.events):
            day.events.append(Event("free", "Free time", "Explore at your own pace"))
    return days
