"""The simplifying assumptions, as data: which cities we plan for and which airports we fly from.

Keeping these as a fixed list (instead of geocoding anything the traveler types) removes a whole
class of failures: ambiguous places, cities without airports, and airport-code guessing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    airport: str  # the airport we search flights to (Kyoto has none, so it uses Osaka Kansai)
    country: str
    food_usd_per_person_per_day: int  # rough estimate used by the budget check (Phase 6)


SUPPORTED_CITIES: dict[str, City] = {
    "Tokyo": City("NRT", "Japan", 45),
    "Kyoto": City("KIX", "Japan", 40),
    "Paris": City("CDG", "France", 60),
    "London": City("LHR", "United Kingdom", 60),
    "Rome": City("FCO", "Italy", 50),
    "Barcelona": City("BCN", "Spain", 45),
    "New York": City("JFK", "United States", 70),
    "Singapore": City("SIN", "Singapore", 45),
}

# Major US departure airports. Origin must be one of these.
US_AIRPORTS: dict[str, str] = {
    "ATL": "Atlanta", "AUS": "Austin", "BOS": "Boston", "CLT": "Charlotte", "DEN": "Denver",
    "DFW": "Dallas", "DTW": "Detroit", "EWR": "Newark", "IAD": "Washington Dulles",
    "IAH": "Houston", "JFK": "New York JFK", "LAS": "Las Vegas", "LAX": "Los Angeles",
    "MIA": "Miami", "MSP": "Minneapolis", "OAK": "Oakland", "ORD": "Chicago O'Hare",
    "PDX": "Portland", "PHL": "Philadelphia", "PHX": "Phoenix", "SAN": "San Diego",
    "SEA": "Seattle", "SFO": "San Francisco", "SJC": "San Jose",
}

NEW_YORK_AIRPORTS = {"JFK", "EWR"}


def find_city(name: str) -> str | None:
    """Case-insensitive lookup that returns the canonical city name, e.g. 'new york' -> 'New York'."""
    normalized = name.strip().lower()
    return next((c for c in SUPPORTED_CITIES if c.lower() == normalized), None)
