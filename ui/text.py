"""Text helpers for Streamlit. Every dynamic string shown with st.markdown / st.info / st.success /
st.error / st.warning / st.write / st.caption / button labels must go through safe_md().

Streamlit renders "$...$" as LaTeX, so "$94/night ... $567" turns into garbled math. A test
(tests/test_ui_text.py) scans the UI code for Streamlit calls with an unescaped "$" string.
"""


def safe_md(text: str) -> str:
    """Escape $ (no accidental LaTeX) and keep single line breaks (Markdown would join the lines)."""
    return str(text).replace("$", "\\$").replace("\n", "  \n")


def safe_html(text) -> str:
    """For text placed inside HTML passed to st.markdown(unsafe_allow_html=True): escape HTML, and
    write $ as an entity. Streamlit still parses $...$ as LaTeX inside inline HTML such as <span>
    (seen in the live trace), so html.escape alone is not enough."""
    import html

    return html.escape(str(text)).replace("$", "&#36;")
