"""Site chrome shown on every page: the author line in the sidebar and the page footer."""

import streamlit as st

AUTHOR = "Ashutosh Kaushik"
LINKEDIN_URL = "https://www.linkedin.com/in/ashutosh-kaushik/"

# LinkedIn's "in" logo (rounded square with the letters cut out), from Simple Icons (CC0). Drawn in white on
# LinkedIn blue, as LinkedIn's brand guidelines describe for linking to a profile.
LINKEDIN_LOGO = (
    "<svg class='li-logo' viewBox='0 0 24 24' aria-hidden='true'><path fill='currentColor' d='M20.447 20.452h-3.554"
    "v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 "
    "1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138"
    ".92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v"
    "11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 "
    "22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/></svg>"
)


def author_card() -> None:
    """Name and LinkedIn button, pinned to the bottom of the sidebar (see .author-card in ui/theme.py)."""
    with st.sidebar:
        st.markdown(
            f"<div class='author-card'><div class='author-name'>{AUTHOR}</div>"
            f"<a class='li-btn' href='{LINKEDIN_URL}' target='_blank' rel='noopener noreferrer' "
            f"aria-label='Connect with {AUTHOR} on LinkedIn (opens in a new tab)'>{LINKEDIN_LOGO}"
            "<span>Connect on LinkedIn</span></a></div>", unsafe_allow_html=True)


def footer() -> None:
    """Page footer, kept at the bottom of the window (see .site-footer in ui/theme.py). Phones and pages with a
    chat box show the short line only, so it stays one line."""
    st.markdown(
        "<div class='site-footer'><span class='sf-long'>Built with Streamlit, LangChain and LangGraph, using "
        "Claude Code and other AI tools, with a human in the loop. Flight prices are Duffel test-mode data; "
        "bookings are simulated.</span>"
        "<span class='sf-short'>Flight prices are Duffel test-mode data; bookings are simulated.</span></div>",
        unsafe_allow_html=True)
