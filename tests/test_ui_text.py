"""Regression: dollar signs rendered as LaTeX (garbled prices on the planner Lab page and in labels).

Scans the Streamlit UI code for calls whose text argument contains an unescaped "$". Dynamic text
must go through ui.text.safe_md() (Markdown) or safe_html() (inside HTML); static text must write
"\\$" or "&#36;". HTML is NOT exempt: Streamlit parses $...$ as LaTeX inside inline HTML too
(the live trace showed "hotel798.00" rendered as math).
"""

import ast
from pathlib import Path

from ui.text import safe_html, safe_md

UI_DIRS = [Path(__file__).parent.parent / d for d in ("lab", "ui")]
RENDERED = {"markdown", "info", "success", "error", "warning", "write", "caption", "button", "title",
            "header", "subheader", "toast", "radio", "checkbox", "expander", "status", "page_link"}


def unescaped_dollar(text: str) -> bool:
    return any(ch == "$" and (i == 0 or text[i - 1] != "\\") for i, ch in enumerate(text))


def literal_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant))
    return ""


def test_no_unescaped_dollar_in_streamlit_text():
    offenders = []
    for folder in UI_DIRS:
        for path in folder.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args):
                    continue
                if node.func.attr not in RENDERED:
                    continue
                if unescaped_dollar(literal_text(node.args[0])):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"Wrap these in safe_md() or escape as \\$: {offenders}"


def test_safe_md_escapes_dollars_and_keeps_line_breaks():
    assert safe_md("$94/night · $567 total") == "\\$94/night · \\$567 total"
    assert safe_md("a\nb") == "a  \nb"


def test_safe_html_escapes_markup_and_dollars():
    assert safe_html("<b>$798</b>") == "&lt;b&gt;&#36;798&lt;/b&gt;"
