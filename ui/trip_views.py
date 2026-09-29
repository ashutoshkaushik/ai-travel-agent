"""Visual building blocks for a trip: day-by-day columns, the cost bar, and the map.

Everything here only draws. The data comes from pure-Python code (itinerary.build_itinerary,
models.cost_breakdown), so what you see always matches what the agents decided and code verified.
"""


import pydeck as pdk
import streamlit as st

from travel_planner.itinerary import Day
from travel_planner.models import CostBreakdown, TripPlan
from travel_planner.tools._common import ToolInputError
from travel_planner.tools.geo import lookup_place
from ui.text import safe_html
from ui.theme import MAP_RGB

ICON = {"flight": "✈", "checkin": "⌂", "checkout": "⌂", "sight": "◉", "free": "·"}


def itinerary_columns(days: list[Day]) -> None:
    """One column per calendar day: what you do and where you sleep."""
    cols = []
    for d in days:
        events = "".join(
            f"<div class='ev {e.kind}'><div class='et'>{ICON[e.kind]} {safe_html(e.title)}</div>"
            + (f"<div class='ed'>{safe_html(e.detail)}</div>" if e.detail else "") + "</div>"
            for e in d.events)
        if d.stay is None:
            stay = "<div class='stay none'>Home</div>"
        elif d.stay in ("On the overnight flight", "Travelling"):  # in transit, not a hotel night
            stay = f"<div class='stay none'>{safe_html(d.stay)}</div>"
        else:
            stay = f"<div class='stay'>Night: {safe_html(d.stay)}</div>"
        cols.append(f"<div class='day'><div class='head'><div class='dn'>Day {d.number}</div>"
                    f"<div class='dd'>{d.label}</div></div><div class='body'>{events}</div>{stay}</div>")
    st.markdown(f"<div class='itin'>{''.join(cols)}</div>", unsafe_allow_html=True)


def cost_bar(costs: CostBreakdown) -> None:
    """Stacked bar of where the money goes, against the budget."""
    parts = [("Flight", costs.flight, "var(--flight)"), ("Hotel", costs.hotel, "var(--hotel)"),
             ("Sights", costs.sights, "var(--sight)"), ("Food (est.)", costs.food_estimate, "var(--muted)")]
    scale = max(costs.total, costs.budget) or 1
    bars = "".join(f"<div style='width:{v / scale * 100:.1f}%;background:{c}' title='{n}'></div>" for n, v, c in parts)
    legend = "".join(f"<span><i style='background:{c}'></i>{n} &#36;{v:,.0f}</span>" for n, v, c in parts)
    verdict = (f"<span class='badge no'>over budget by &#36;{-costs.remaining:,.0f}</span>" if costs.over_budget
               else f"<span class='badge ok'>&#36;{costs.remaining:,.0f} left</span>")
    st.markdown(f"<div><b>&#36;{costs.total:,.0f}</b> of &#36;{costs.budget:,} budget {verdict}</div>"
                f"<div class='costbar'>{bars}</div><div class='costlegend'>{legend}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def _geocode(place: str) -> tuple[float, float] | None:
    try:
        hit = lookup_place(place)
        return hit["lat"], hit["lon"]
    except (ToolInputError, KeyError, IndexError, OSError):
        return None


def trip_map(plan: TripPlan, city: str, hotel_detail: dict | None) -> None:
    """The hotel and every sight, numbered by trip day. Sights are geocoded once (cached)."""
    points = []
    if hotel_detail and hotel_detail.get("lat"):
        points.append({"name": f"Hotel: {plan.hotel.name}", "label": "H", "lat": hotel_detail["lat"],
                       "lon": hotel_detail["lon"], "color": MAP_RGB["hotel"], "radius": 140})
    with st.spinner("Placing sights on the map (only slow the first time)…"):
        for s in sorted(plan.sights, key=lambda s: s.day):
            where = _geocode(f"{s.name}, {city}")
            if where:
                points.append({"name": f"Day {s.day}: {s.name}", "label": str(s.day), "lat": where[0], "lon": where[1],
                               "color": MAP_RGB["sight"], "radius": 110})
    if not points:
        st.caption("No map locations available for this plan.")
        return
    lat = sum(p["lat"] for p in points) / len(points)
    lon = sum(p["lon"] for p in points) / len(points)
    layers = [
        pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]", get_fill_color="color", get_radius="radius",
                  radius_min_pixels=7, radius_max_pixels=18, pickable=True, opacity=0.9,
                  stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5),
        pdk.Layer("TextLayer", points, get_position="[lon, lat]", get_text="label", get_size=12,
                  get_color=[255, 255, 255], get_alignment_baseline="'center'", font_weight=700),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, map_style=None,  # None = follow the app's light/dark theme
                             initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=11.2),
                             tooltip={"text": "{name}"}), height=420)
    missing = len(plan.sights) - (len(points) - (1 if hotel_detail and hotel_detail.get("lat") else 0))
    st.caption("H = your hotel · numbers = trip day of each sight"
               + (f" · {missing} sight(s) couldn't be located" if missing > 0 else ""))
