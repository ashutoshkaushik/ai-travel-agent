"""Phase 0 smoke test for the non-Duffel APIs, going through the disk cache.

Run twice: the first run hits the network, the second should say (cache) for every API.

    uv run python scripts/check_apis.py
"""

from datetime import date, timedelta

from travel_planner.config import require
from travel_planner.http_cache import request

CITY = "Tokyo"
CHECK_IN = date.today() + timedelta(days=70)
CHECK_OUT = CHECK_IN + timedelta(days=3)
NOMINATIM_HEADERS = {"User-Agent": "travel-agent-lab/0.1"}  # required by Nominatim policy


def source(resp) -> str:
    return "(cache)" if resp.from_cache else "(network)"


def check_nominatim() -> tuple[float, float]:
    resp = request(
        "GET",
        "https://nominatim.openstreetmap.org/search",
        params={"q": CITY, "format": "json", "limit": 1},
        headers=NOMINATIM_HEADERS,
    )
    print(f"\n[Nominatim] geocode {CITY!r}: HTTP {resp.status_code} {source(resp)}")
    place = resp.data[0]
    lat, lon = float(place["lat"]), float(place["lon"])
    print(f"  {place['display_name']} -> ({lat:.4f}, {lon:.4f})")
    return lat, lon


def check_open_meteo(lat: float, lon: float) -> None:
    # Forecasts only reach ~16 days ahead, so for a trip months away use the same dates
    # last year from the historical archive as a "typical weather" estimate.
    last_year_in = CHECK_IN.replace(year=CHECK_IN.year - 1)
    last_year_out = CHECK_OUT.replace(year=CHECK_OUT.year - 1)
    resp = request(
        "GET",
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": last_year_in.isoformat(),
            "end_date": last_year_out.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
        },
    )
    print(f"\n[Open-Meteo] typical weather ({last_year_in} to {last_year_out}): HTTP {resp.status_code} {source(resp)}")
    daily = resp.data["daily"]
    for day, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"], daily["temperature_2m_min"], daily["precipitation_sum"]):
        print(f"  {day}: {lo}–{hi} °C, rain {rain} mm")


def check_opentripmap(lat: float, lon: float) -> None:
    resp = request(
        "GET",
        "https://api.opentripmap.com/0.1/en/places/radius",
        params={
            "radius": 5000,
            "lat": lat,
            "lon": lon,
            "kinds": "cultural,foods",
            "rate": 3,  # 3 = most notable places
            "format": "json",
            "limit": 5,
            "apikey": require("OPENTRIPMAP_API_KEY"),
        },
    )
    print(f"\n[OpenTripMap] notable cultural/food places near {CITY}: HTTP {resp.status_code} {source(resp)}")
    if resp.status_code >= 400:
        print(f"  {resp.data}")
        return
    for place in resp.data:
        print(f"  - {place['name']}  [{place['kinds'].split(',')[0]}]")


def check_serpapi_hotels() -> None:
    resp = request(
        "GET",
        "https://serpapi.com/search.json",
        params={
            "engine": "google_hotels",
            "q": f"{CITY} hotels",
            "check_in_date": CHECK_IN.isoformat(),
            "check_out_date": CHECK_OUT.isoformat(),
            "adults": 2,
            "currency": "USD",
            "gl": "us",
            "hl": "en",
            "api_key": require("SERPAPI_API_KEY"),
        },
    )
    print(f"\n[SerpApi] Google Hotels {CITY} {CHECK_IN} to {CHECK_OUT}: HTTP {resp.status_code} {source(resp)}")
    if resp.status_code >= 400 or "error" in resp.data:
        print(f"  {resp.data.get('error', resp.data)}")
        return
    properties = resp.data.get("properties", [])
    print(f"  {len(properties)} properties on first page")
    for p in properties[:5]:
        price = p.get("rate_per_night", {}).get("lowest", "n/a")
        print(f"  - {p['name']}: {price}/night, rating {p.get('overall_rating', 'n/a')}")


if __name__ == "__main__":
    lat, lon = check_nominatim()
    check_open_meteo(lat, lon)
    check_opentripmap(lat, lon)
    check_serpapi_hotels()
