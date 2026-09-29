"""Hotel search via SerpApi's Google Hotels engine. Real hotels and prices, read-only."""

from typing import Literal

from langchain.tools import tool

from travel_planner.config import require
from travel_planner.http_cache import request
from travel_planner.tools._common import ToolInputError, error, ok, parse_future_date, returns_errors_as_data

SERPAPI_URL = "https://serpapi.com/search.json"
SORT_CODES = {"relevance": None, "price": "3", "rating": "8"}  # SerpApi's sort_by values
DEFAULT_MIN_RATING = 4.0
# Shared-room and pod listings are never a fit for our travelers (one room for the whole party)
EXCLUDED_NAME_WORDS = ("hostel", "dorm", "capsule", "backpacker", "male only", "female only", "camping", "campground")


def _is_private_hotel(hotel: dict) -> bool:
    name = hotel["name"].lower()
    return not any(word in name for word in EXCLUDED_NAME_WORDS)


def _trim_property(p: dict) -> dict:
    gps = p.get("gps_coordinates") or {}
    return {
        "hotel_id": p.get("property_token"),
        "name": p["name"],
        "type": p.get("type"),
        "price_per_night": p.get("rate_per_night", {}).get("extracted_lowest"),
        "total_price": p.get("total_rate", {}).get("extracted_lowest"),
        "rating": p.get("overall_rating"),
        "reviews": p.get("reviews"),
        "hotel_class": p.get("extracted_hotel_class"),
        "lat": gps.get("latitude"),
        "lon": gps.get("longitude"),
        "amenities": (p.get("amenities") or [])[:6],
        "description": (p.get("description") or "")[:200] or None,  # untrusted text: guarded before the LLM sees it
    }


@tool(parse_docstring=True)
@returns_errors_as_data
def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    adults: int = 2,
    max_price_per_night: int | None = None,
    min_rating: float = DEFAULT_MIN_RATING,
    sort_by: Literal["relevance", "price", "rating"] = "relevance",
    max_results: int = 5,
) -> dict:
    """Search hotels in a city for specific dates. Prices are in USD and include every night of the stay.

    Hostels, dorms, and capsule hotels are always excluded.

    Args:
        city: City or neighborhood, e.g. 'Tokyo' or 'Shinjuku, Tokyo'.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format (after check_in).
        adults: Number of adult guests.
        max_price_per_night: Upper limit on the nightly price in USD; omit for no limit.
        min_rating: Minimum guest rating from 1.0 to 5.0. Defaults to 4.0.
        sort_by: 'relevance', 'price' (cheapest first), or 'rating' (best first).
        max_results: How many hotels to return (1-10).
    """
    start, end = parse_future_date(check_in, "check_in"), parse_future_date(check_out, "check_out")
    nights = (end - start).days
    if nights < 1:
        raise ToolInputError("check_out must be at least one day after check_in.")
    if nights > 30:
        raise ToolInputError("Stays longer than 30 nights are not supported.", "Split into multiple stays.")

    params = {
        "engine": "google_hotels",
        "q": f"{city} hotels",
        "check_in_date": start.isoformat(),
        "check_out_date": end.isoformat(),
        "adults": adults,
        "currency": "USD",
        "gl": "us",
        "hl": "en",
        "api_key": require("SERPAPI_API_KEY"),
    }
    # Price is filtered locally, not with SerpApi's max_price (which returned no results in testing).
    # Bonus: one cached search per city+dates serves every budget the agent tries while replanning.
    if SORT_CODES[sort_by]:
        params["sort_by"] = SORT_CODES[sort_by]

    resp = request("GET", SERPAPI_URL, params=params, ttl_seconds=6 * 60 * 60)
    if resp.status_code >= 400 or "error" in resp.data:
        return error(f"Hotel search failed: {resp.data.get('error', resp.status_code)}")

    hotels = [_trim_property(p) for p in resp.data.get("properties", [])]
    hotels = [h for h in hotels if h["price_per_night"] is not None]  # unpriced listings are useless for budgeting
    hotels = [h for h in hotels if _is_private_hotel(h)]
    if max_price_per_night is not None:
        hotels = [h for h in hotels if h["price_per_night"] <= max_price_per_night]
    hotels = [h for h in hotels if (h["rating"] or 0) >= min_rating]
    if not hotels:
        return error("No hotels matched.", "Raise max_price_per_night, lower min_rating, or try a nearby area.")

    max_results = max(1, min(max_results, 10))
    return ok(
        city=city,
        nights=nights,
        hotels_found=len(hotels),
        hotels=hotels[:max_results],
    )
