"""Phase 1 tests: call each tool exactly the way the agent will (tool.invoke with a dict of args).

Happy-path tests hit the real APIs once, then run from the disk cache.
Error-path tests never touch the network: validation fails first.
"""

import json
import os
from datetime import date, timedelta

import pytest

from travel_planner.tools import (
    geocode,
    get_weather,
    search_flights,
    search_hotels,
    search_places_by_interest,
    search_top_sights,
)

TRIP_START = date.today() + timedelta(days=70)
TRIP_END = TRIP_START + timedelta(days=3)
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()
# Happy paths need real (test-mode) keys; CI has placeholders only, and runs the recorded evals instead
live = pytest.mark.skipif(os.environ.get("SERPAPI_API_KEY", "offline") == "offline",
                          reason="needs real API keys (CI replays recorded responses in the evals instead)")
MAX_RESULT_CHARS = 6000  # ~1.5K tokens: trimmed results must stay small enough for the LLM


def small_enough(result: dict) -> bool:
    return len(json.dumps(result)) < MAX_RESULT_CHARS


# ---------- happy paths ----------

@live
def test_search_flights_round_trip():
    r = search_flights.invoke({
        "origin": "SFO", "destination": "NRT",
        "depart_date": TRIP_START.isoformat(), "return_date": TRIP_END.isoformat(), "adults": 2,
    })
    assert r["status"] == "ok", r
    offer = r["offers"][0]
    assert len(offer["slices"]) == 2
    assert offer["price"] == r["cheapest_price"]
    assert small_enough(r)


@live
def test_search_flights_nonstop_sorted_by_duration():
    r = search_flights.invoke({
        "origin": "SFO", "destination": "NRT", "depart_date": TRIP_START.isoformat(),
        "max_stops": 0, "sort_by": "duration",
    })
    assert r["status"] == "ok", r
    assert all(s["stops"] == 0 for o in r["offers"] for s in o["slices"])
    durations = [o["slices"][0]["duration_min"] for o in r["offers"]]
    assert durations == sorted(durations)


@live
def test_search_hotels_with_budget_filter():
    r = search_hotels.invoke({
        "city": "Tokyo", "check_in": TRIP_START.isoformat(), "check_out": TRIP_END.isoformat(),
        "max_price_per_night": 150, "min_rating": 4.0,
    })
    assert r["status"] == "ok", r
    assert r["nights"] == 3
    for h in r["hotels"]:
        assert h["price_per_night"] <= 150
        assert h["rating"] >= 4.0
    assert small_enough(r)


@live
def test_search_top_sights_has_prices_and_popularity():
    r = search_top_sights.invoke({"city": "Tokyo", "max_results": 10})
    assert r["status"] == "ok", r
    names = [s["name"] for s in r["sights"]]
    assert any("Sens" in n for n in names), names  # Senso-ji should be a top sight
    assert any(s["price_usd"] == 0.0 for s in r["sights"])  # some sights are free
    assert small_enough(r)


@live
def test_search_top_sights_falls_back_to_city_and_country():
    """'Rome' alone has no Google top-sights panel; the tool must retry as 'Rome, Italy'."""
    r = search_top_sights.invoke({"city": "Rome", "max_results": 5})
    assert r["status"] == "ok", r
    assert any("Colosseum" in s["name"] for s in r["sights"])


@live
def test_search_places_by_interest_dedupes():
    r = search_places_by_interest.invoke({"city": "Tokyo", "interests": ["temples_shrines"], "max_results": 15})
    assert r["status"] == "ok", r
    names = [p["name"] for p in r["places"]]
    assert len(names) == len(set(names))
    assert all("lat" in p and "lon" in p for p in r["places"])


def test_geocode_returns_english():
    r = geocode.invoke({"place": "Tokyo"})
    assert r["status"] == "ok", r
    assert "Tokyo" in r["name"]
    assert r["country"] == "Japan"


def test_weather_far_future_uses_typical_mode():
    r = get_weather.invoke({"place": "Tokyo", "start_date": TRIP_START.isoformat(), "end_date": TRIP_END.isoformat()})
    assert r["status"] == "ok", r
    assert r["mode"].startswith("typical")
    assert len(r["days"]) == 4
    assert r["days"][0]["date"] == TRIP_START.isoformat()  # reported on trip dates, not last year's


# ---------- error paths: returned as data, never raised ----------

@pytest.mark.parametrize("args, expected", [
    ({"origin": "San Francisco", "destination": "NRT", "depart_date": "2030-01-01"}, "IATA"),
    ({"origin": "SFO", "destination": "SFO", "depart_date": "2030-01-01"}, "same airport"),
    ({"origin": "SFO", "destination": "NRT", "depart_date": YESTERDAY}, "not in the future"),
    ({"origin": "SFO", "destination": "NRT", "depart_date": "12/10/2030"}, "not a valid date"),
    ({"origin": "SFO", "destination": "NRT", "depart_date": "2030-01-10", "return_date": "2030-01-05"}, "before depart_date"),
])
def test_search_flights_bad_input(args, expected):
    r = search_flights.invoke(args)
    assert r["status"] == "error"
    assert expected in r["reason"]


def test_search_hotels_checkout_before_checkin():
    r = search_hotels.invoke({"city": "Tokyo", "check_in": "2030-01-10", "check_out": "2030-01-10"})
    assert r["status"] == "error"
    assert "at least one day" in r["reason"]


def test_geocode_unknown_place():
    r = geocode.invoke({"place": "Xqzvbnmplace Nowhereville 99999"})
    assert r["status"] == "error"
    assert "hint" in r
