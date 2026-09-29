"""Travel Agent Lab: Streamlit app.

    uv run streamlit run app.py

Structure: pages are registered with st.navigation below, in three
groups. All agent logic lives in src/travel_planner; the pages only draw state and send choices.
Restart the server after editing src/ or lab/ modules: imported modules are not reloaded.
"""

import streamlit as st

from lab.assistant_page import assistant_page
from lab.compare_page import compare_page
from lab.home import home_page
from lab.lab_pages import LAB_PAGES
from lab.nav import PAGES, TOUR
from lab.overview import diagram_page, lessons_page, usage_page
from ui import chrome, theme


def main() -> None:
    st.set_page_config(page_title="Travel Agent Lab", page_icon=":material/travel_explore:", layout="wide")
    theme.apply_theme()
    pages = {
        "home": st.Page(home_page, title="Start here", icon=":material/home:", default=True),
        "assistant": st.Page(assistant_page, title="Travel Assistant", icon=":material/flight_takeoff:", url_path="assistant"),
        "compare": st.Page(compare_page, title="Compare the levels", icon=":material/compare_arrows:", url_path="compare"),
        "diagram": st.Page(diagram_page, title="System diagram", icon=":material/account_tree:", url_path="diagram"),
        "lessons": st.Page(lessons_page, title="What broke, and the fixes", icon=":material/bug_report:", url_path="lessons"),
        "usage": st.Page(usage_page, title="Usage and cost", icon=":material/payments:", url_path="usage"),
    }
    for key, (title, icon, fn) in LAB_PAGES.items():
        pages[key] = st.Page(fn, title=title, icon=icon, url_path=key)
    PAGES.update(pages)
    page = st.navigation({
        "App": [pages["home"], pages["assistant"]],
        "Overview": [pages["compare"], pages["diagram"], pages["lessons"], pages["usage"]],
        "Agent Lab · built step by step": [pages[k] for k in TOUR],
    }, expanded=True)  # always show every page; no "View more" in the sidebar
    page.run()
    chrome.author_card()
    chrome.footer()


if __name__ == "__main__":
    main()
