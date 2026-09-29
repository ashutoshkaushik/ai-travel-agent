"""Flight search via Duffel (test mode). Prices are synthetic; the shape is real."""

import re
from typing import Literal

from langchain.tools import tool

from travel_planner.config import duffel_token
from travel_planner.http_cache import request
from travel_planner.tools._common import ToolInputError, error, ok, parse_future_date, returns_errors_as_data

DUFFEL_API = "https://api.duffel.com"
SANDBOX_BOOKABLE_AIRLINE = "ZZ"  # Duffel Airways: the airline Duffel guarantees can be booked in test mode
OFFER_CACHE_SECONDS = 20 * 60  # offers expire ~30 minutes after search

_DURATION = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?")


def duffel_headers() -> dict:
    return {
        "Authorization": f"Bearer {duffel_token()}",
        "Duffel-Version": "v2",
        "Accept": "application/json",
    }


def _minutes(iso_duration: str | None) -> int:
    if not iso_duration:
        return 0
    days, hours, minutes = (int(x or 0) for x in _DURATION.fullmatch(iso_duration).groups())
    return days * 1440 + hours * 60 + minutes


def _trim_offer(offer: dict) -> dict:
    """Keep only what a planner needs. A raw offer is ~8 KB; this is ~0.4 KB."""
    slices = []
    for s in offer["slices"]:
        segs = s["segments"]
        slices.append({
            "from": segs[0]["origin"]["iata_code"],
            "to": segs[-1]["destination"]["iata_code"],
            "depart": segs[0]["departing_at"],
            "arrive": segs[-1]["arriving_at"],
            "duration_min": _minutes(s["duration"]),
            "stops": len(segs) - 1,
            "flights": [f"{g['marketing_carrier']['iata_code']}{g['marketing_carrier_flight_number']}" for g in segs],
        })
    return {
        "offer_id": offer["id"],
        "airline": offer["owner"]["name"],
        "price": float(offer["total_amount"]),
        "currency": offer["total_currency"],
        "slices": slices,
        "sandbox_bookable": offer["owner"]["iata_code"] == SANDBOX_BOOKABLE_AIRLINE,
        "expires_at": offer["expires_at"],
    }


@tool(parse_docstring=True)
@returns_errors_as_data
def search_flights(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    adults: int = 1,
    cabin_class: Literal["economy", "premium_economy", "business", "first"] = "economy",
    max_stops: int | None = None,
    sort_by: Literal["price", "duration"] = "price",
    max_results: int = 5,
) -> dict:
    """Search flights between two airports. Returns the best few offers with total prices for all passengers.

    Args:
        origin: 3-letter IATA airport code, e.g. 'SFO'. Use an airport, not a city name.
        destination: 3-letter IATA airport code, e.g. 'NRT' or 'HND' for Tokyo.
        depart_date: Outbound date in YYYY-MM-DD format.
        return_date: Return date in YYYY-MM-DD format for a round trip; omit for one-way.
        adults: Number of adult passengers (1-9).
        cabin_class: Cabin to search.
        max_stops: Maximum stops per direction (0 = nonstop only); omit for any.
        sort_by: 'price' for cheapest first, 'duration' for fastest first.
        max_results: How many offers to return (1-10).
    """
    origin, destination = origin.strip().upper(), destination.strip().upper()
    for field, code in (("origin", origin), ("destination", destination)):
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ToolInputError(f"{field}={code!r} is not a 3-letter IATA airport code.", "Convert city names to airport codes first, e.g. Tokyo -> NRT or HND.")
    if origin == destination:
        raise ToolInputError("origin and destination are the same airport.")
    if not 1 <= adults <= 9:
        raise ToolInputError("adults must be between 1 and 9.")

    depart = parse_future_date(depart_date, "depart_date")
    slices = [{"origin": origin, "destination": destination, "departure_date": depart.isoformat()}]
    if return_date:
        ret = parse_future_date(return_date, "return_date")
        if ret < depart:
            raise ToolInputError("return_date is before depart_date.")
        slices.append({"origin": destination, "destination": origin, "departure_date": ret.isoformat()})

    resp = request(
        "POST",
        f"{DUFFEL_API}/air/offer_requests",
        params={"return_offers": "true", "supplier_timeout": 20000},
        json_body={"data": {"slices": slices, "passengers": [{"type": "adult"}] * adults, "cabin_class": cabin_class}},
        headers=duffel_headers(),
        ttl_seconds=OFFER_CACHE_SECONDS,
        timeout=60,
    )
    if resp.status_code >= 400:
        msg = resp.data.get("errors", [{}])[0].get("message", resp.data)
        return error(f"Flight search failed: {msg}")

    offers = [_trim_offer(o) for o in resp.data["data"]["offers"]]
    if max_stops is not None:
        offers = [o for o in offers if all(s["stops"] <= max_stops for s in o["slices"])]
    if not offers:
        return error("No flights matched.", "Try allowing more stops, a nearby airport, or dates +/- 1 day.")

    key = (lambda o: o["price"]) if sort_by == "price" else (lambda o: sum(s["duration_min"] for s in o["slices"]))
    offers.sort(key=key)
    max_results = max(1, min(max_results, 10))

    return ok(
        offers_found=len(offers),
        cheapest_price=min(o["price"] for o in offers),
        note="Test-mode prices are synthetic. Only offers with sandbox_bookable=true can be booked.",
        offers=offers[:max_results],
    )
