"""Activity search: two tools with different strengths, so the agent has to choose.

- search_top_sights (SerpApi / Google): the famous places, ranked by popularity, with ticket prices.
- search_places_by_interest (OpenTripMap): niche places for a specific interest, with coordinates,
  but not ranked by popularity and no prices.
"""

import re
from typing import Literal

from langchain.tools import tool

from travel_planner.cities import SUPPORTED_CITIES, find_city
from travel_planner.config import require
from travel_planner.http_cache import request
from travel_planner.tools._common import error, ok, returns_errors_as_data
from travel_planner.tools.geo import lookup_place

Interest = Literal["temples_shrines", "history", "museums", "food", "nature", "architecture", "entertainment"]

INTEREST_KINDS = {  # OpenTripMap category names
    "temples_shrines": "religion",
    "history": "historic",
    "museums": "museums",
    "food": "foods",
    "nature": "natural",
    "architecture": "architecture",
    "entertainment": "amusements",
}
STATUS_WORDS = ("open", "closed", "closes", "opens")  # Google puts live open/closed status in the description


def _parse_price(price: str | None) -> float | None:
    if not price:
        return None
    if price.strip().lower() == "free":
        return 0.0
    match = re.search(r"[\d,]+(?:\.\d+)?", price)
    return float(match.group().replace(",", "")) if match else None


@tool(parse_docstring=True)
@returns_errors_as_data
def search_top_sights(city: str, max_results: int = 10) -> dict:
    """Find the most popular sights in a city, ranked by popularity, with ratings and ticket prices in USD.

    Best for building the core of an itinerary. For niche interests use search_places_by_interest.

    Args:
        city: City name, e.g. 'Tokyo' or 'Kyoto'.
        max_results: How many sights to return (1-20).
    """
    # Google's top-sights panel is inconsistent: "Tokyo" has one but "Tokyo, Japan" doesn't, while
    # "Rome" has none (Rome, Georgia?) but "Rome, Italy" does. So try the bare name, then add the
    # country for supported cities. Both results are cached, so the fallback costs one search, once.
    canonical = find_city(city)
    queries = [city] + ([f"{canonical}, {SUPPORTED_CITIES[canonical].country}"] if canonical else [])
    sights = []
    for place in queries:
        resp = request(
            "GET",
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": f"top sights in {place}", "gl": "us", "hl": "en", "api_key": require("SERPAPI_API_KEY")},
            ttl_seconds=7 * 24 * 60 * 60,
        )
        if resp.status_code >= 400 or "error" in resp.data:
            return error(f"Sight search failed: {resp.data.get('error', resp.status_code)}")
        sights = resp.data.get("top_sights", {}).get("sights", [])
        if sights:
            break
    if not sights:
        return error(f"No top sights found for {city!r}.", "Use search_places_by_interest instead.")

    results = []
    for s in sights[: max(1, min(max_results, 20))]:
        desc = s.get("description") or ""
        results.append({
            "name": s["title"],
            "category": None if desc.lower().startswith(STATUS_WORDS) else desc,
            "rating": s.get("rating"),
            "reviews": s.get("reviews"),
            "price_usd": _parse_price(s.get("price")),  # None = unknown, 0.0 = free
        })
    return ok(city=city, sights=results)


@tool(parse_docstring=True)
@returns_errors_as_data
def search_places_by_interest(city: str, interests: list[Interest], radius_km: int = 10, max_results: int = 10) -> dict:
    """Find places matching specific interests near a city center, with coordinates.

    Not ranked by popularity and no prices; use search_top_sights for famous attractions.

    Args:
        city: City name, e.g. 'Tokyo'.
        interests: One or more interest categories to search.
        radius_km: Search radius around the city center in km (1-30).
        max_results: How many places to return (1-20).
    """
    center = lookup_place(city)
    kinds = ",".join(INTEREST_KINDS[i] for i in interests)
    resp = request(
        "GET",
        "https://api.opentripmap.com/0.1/en/places/radius",
        params={
            "radius": max(1, min(radius_km, 30)) * 1000,
            "lat": center["lat"],
            "lon": center["lon"],
            "kinds": kinds,
            "rate": "3",  # only the most notable places
            "format": "json",
            "limit": 500,  # results come nearest-first, so fetch wide and rank ourselves
            "apikey": require("OPENTRIPMAP_API_KEY"),
        },
        ttl_seconds=30 * 24 * 60 * 60,
    )
    if resp.status_code >= 400:
        return error(f"Place search failed: {resp.data}")

    seen, places = set(), []
    for p in sorted(resp.data, key=lambda p: (-(p.get("rate") or 0), p["dist"])):
        dedupe_key = p.get("wikidata") or p["name"].lower()  # the same temple often appears under 2 spellings
        if not p.get("name") or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        places.append({
            "name": p["name"],
            "kinds": p["kinds"].split(",")[:3],
            "lat": round(p["point"]["lat"], 5),
            "lon": round(p["point"]["lon"], 5),
            "km_from_center": round(p["dist"] / 1000, 1),
        })
        if len(places) >= max(1, min(max_results, 20)):
            break

    if not places:
        return error("No places matched.", "Increase radius_km or try other interests.")
    return ok(city=city, interests=interests, places=places)
