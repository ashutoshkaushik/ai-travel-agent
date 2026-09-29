"""Page registry, so any page can link to any other (landing-page buttons, Lab Previous / Next).

app.py builds the st.Page objects and registers them here before running the navigation.
"""

import streamlit as st

PAGES: dict = {}

# The Agent Lab tour, in build order: each step adds one concept to the one before
TOUR = ["tools", "loop", "hitl", "structured", "planner", "approval", "rag", "router", "evals", "guardrails"]


def go_button(key: str, label: str, primary: bool = False, button_key: str | None = None) -> None:
    """A real, prominent button that opens another page."""
    if key in PAGES and st.button(label, type="primary" if primary else "secondary", width="stretch",
                                  key=button_key or f"go_{key}_{label}"):
        st.switch_page(PAGES[key])


def tour_footer(key: str) -> None:
    """Previous / Next buttons at the bottom of a Lab page."""
    i = TOUR.index(key)
    prev_key = TOUR[i - 1] if i > 0 else "home"
    next_key = TOUR[i + 1] if i < len(TOUR) - 1 else "lessons"
    st.divider()
    left, _, right = st.columns([2, 1, 2])
    with left:
        if prev_key in PAGES:
            go_button(prev_key, f"← Previous: {PAGES[prev_key].title}", button_key=f"prev_{key}")
    with right:
        if next_key in PAGES:
            go_button(next_key, f"Next: {PAGES[next_key].title} →", primary=True, button_key=f"next_{key}")
