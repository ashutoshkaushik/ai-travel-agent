"""The agent's tools. Each one is read-only in Phase 1; booking tools arrive in Phase 9."""

from travel_planner.tools.activities import search_places_by_interest, search_top_sights
from travel_planner.tools.flights import search_flights
from travel_planner.tools.geo import geocode, get_weather
from travel_planner.tools.hotels import search_hotels

ALL_TOOLS = [search_flights, search_hotels, search_top_sights, search_places_by_interest, geocode, get_weather]

__all__ = ["ALL_TOOLS", *[t.name for t in ALL_TOOLS]]
