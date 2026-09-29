"""Geocoding (Nominatim) and weather (Open-Meteo). Both free, no API key."""

import time
from datetime import timedelta
from statistics import mean

from langchain.tools import tool

from travel_planner.http_cache import request
from travel_planner.tools._common import ToolInputError, ok, parse_date, returns_errors_as_data
from travel_planner.config import today

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "travel-agent-lab/0.1"}  # required by Nominatim policy
FORECAST_HORIZON_DAYS = 14  # Open-Meteo forecasts reach ~16 days; beyond that we use last year's weather

_last_nominatim_call = 0.0


def lookup_place(place: str) -> dict:
    """Plain helper (not a tool) used by several tools. Raises ToolInputError if not found."""
    global _last_nominatim_call
    wait = 1.0 - (time.monotonic() - _last_nominatim_call)  # Nominatim allows 1 request/second
    if wait > 0:
        time.sleep(wait)

    resp = request(
        "GET",
        NOMINATIM_URL,
        params={"q": place, "format": "json", "limit": 1, "accept-language": "en", "addressdetails": 1},
        headers=NOMINATIM_HEADERS,
        ttl_seconds=30 * 24 * 60 * 60,  # places don't move
    )
    if not resp.from_cache:
        _last_nominatim_call = time.monotonic()

    if not resp.data:
        raise ToolInputError(f"Could not find a place called {place!r}.", "Try a more specific name, e.g. 'Kyoto, Japan'.")
    hit = resp.data[0]
    return {
        "name": hit["display_name"],
        "country": hit.get("address", {}).get("country"),
        "lat": round(float(hit["lat"]), 5),
        "lon": round(float(hit["lon"]), 5),
    }


@tool(parse_docstring=True)
@returns_errors_as_data
def geocode(place: str) -> dict:
    """Look up the coordinates of a city, neighborhood, landmark, or address.

    Use this to place hotels and activities on a map or to judge distances between them.

    Args:
        place: Place name, as specific as possible, e.g. 'Asakusa, Tokyo' or 'Kyoto Station'.
    """
    return ok(**lookup_place(place))


@tool(parse_docstring=True)
@returns_errors_as_data
def get_weather(place: str, start_date: str, end_date: str) -> dict:
    """Get the daily weather for a place over a date range.

    For dates within the next two weeks this is a real forecast. For dates further out it
    returns the weather on the same dates last year as a typical-weather estimate.

    Args:
        place: City or area name, e.g. 'Tokyo, Japan'.
        start_date: First day in YYYY-MM-DD format.
        end_date: Last day in YYYY-MM-DD format (at most 31 days after start_date).
    """
    start, end = parse_date(start_date, "start_date"), parse_date(end_date, "end_date")
    if end < start:
        raise ToolInputError("end_date is before start_date.")
    if (end - start).days > 31:
        raise ToolInputError("Date range is longer than 31 days.", "Split the request into shorter ranges.")

    loc = lookup_place(place)
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }

    if start <= today() + timedelta(days=FORECAST_HORIZON_DAYS):
        mode, url, ttl = "forecast", "https://api.open-meteo.com/v1/forecast", 3 * 60 * 60
        params |= {"start_date": start.isoformat(), "end_date": end.isoformat()}
    else:
        mode, url, ttl = "typical (same dates last year)", "https://archive-api.open-meteo.com/v1/archive", 30 * 24 * 60 * 60
        shift = timedelta(days=365)
        params |= {"start_date": (start - shift).isoformat(), "end_date": (end - shift).isoformat()}

    daily = request("GET", url, params=params, ttl_seconds=ttl).data["daily"]
    highs, lows, rain = daily["temperature_2m_max"], daily["temperature_2m_min"], daily["precipitation_sum"]
    trip_days = [(start + timedelta(days=i)).isoformat() for i in range(len(highs))]

    return ok(
        place=loc["name"],
        mode=mode,
        summary={
            "avg_high_c": round(mean(highs), 1),
            "avg_low_c": round(mean(lows), 1),
            "rainy_days": sum(1 for r in rain if r and r >= 1.0),
        },
        days=[
            {"date": d, "high_c": h, "low_c": lo, "rain_mm": r}
            for d, h, lo, r in zip(trip_days, highs, lows, rain)
        ],
    )
