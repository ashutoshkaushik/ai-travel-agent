"""'Compare the levels': the Start page's five-level ladder, made interactive.

The same request runs at five levels of capability, and plain code scores each answer. All logic
lives in src/travel_planner/compare.py (reusing the project's tools and agents); this page only draws.
Results are one table (levels as columns, measures as rows) so every row lines up across levels.
"""

import streamlit as st

from lab.home import LADDER
from travel_planner.config import LIVE
from travel_planner.compare import (
    ESTIMATED_COST,
    LEVELS,
    decision_needed,
    is_cached,
    run_level,
    score,
    takeaway,
)
from ui.text import safe_html, safe_md

DEFAULT_REQUEST = "Plan London from SFO, March 10 to 17 2027, 2 adults, budget 1500 dollars"

CSS = """<style>
  .ctab { width: 100%; border-collapse: collapse; font-size: .84rem; table-layout: fixed; }
  .ctab th, .ctab td { padding: .45rem .55rem; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }
  .ctab thead th { border-bottom: 2px solid var(--line); vertical-align: bottom; }
  .ctab th.rowh { width: 11rem; font-weight: 600; color: var(--muted); }
  .ctab .lvl { font-family: var(--font-heading); font-size: 1.02rem; font-weight: 600; }
  .ctab .flow { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); font-weight: 400; }
  .ctab .this { color: var(--accent); }
  .ctab tr.sec th { font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                    color: var(--muted); padding-top: .9rem; }
  .ctab td.num { font-variant-numeric: tabular-nums; }
  .ctab tr.take td { background: var(--code-soft); font-size: .8rem; line-height: 1.4; }
</style>"""


def _check(ok: bool | None, yes: str, no: str, none: str) -> str:
    if ok is None:
        return f"<span class='muted'>{none}</span>"
    return f"<span class='badge ok'>✓ {yes}</span>" if ok else f"<span class='badge no'>✗ {no}</span>"


def cells(level: str, result, c) -> dict[str, str]:
    """Every row's cell for one level, as HTML."""
    if result is None:
        return {}
    if level == "L5":
        prices = _check(c.real_prices, "verified by code", "not verified", "no plan yet")
    elif c.prices_total == 0:
        prices = _check(None, "", "", "no prices given")
    else:
        prices = _check(c.real_prices, f"all {c.prices_total} from tools", f"{c.prices_traced} of {c.prices_total} traced", "")
    return {
        "LLM calls": str(result.llm_calls), "Tool calls": str(result.tool_calls), "Tokens": f"{result.tokens:,}",
        "Cost": f"&#36;{result.cost_usd:.4f}", "Latency": f"{result.seconds:.1f}s",
        "Real prices": prices,
        "Budget": _check(c.within_budget, "within budget", "over budget", "no total stated"),
        "Invented": _check(c.invented == 0, "none", f"{c.invented} untraceable", ""),
        "Asked when needed": (_check(c.asked, "asked you", "didn't ask", "") if c.needed
                              else "<span class='muted'>not needed</span>"),
        "Stopped at": f"the {result.paused} step (nothing booked)" if result.paused else "<span class='muted'>—</span>",
        "Takeaway": safe_html(takeaway(level, c)),
    }


def results_table(results: dict, checks: dict) -> None:
    head = "".join(
        f"<th><div class='lvl{' this' if lvl == 'L5' else ''}'>{lvl} · {safe_html(LEVELS[lvl])}</div>"
        f"<div class='flow'>{safe_html(LADDER[i][1])}</div></th>" for i, lvl in enumerate(LEVELS))
    by_level = {lvl: cells(lvl, results[lvl], checks.get(lvl)) for lvl in LEVELS}
    sections = [("Effort", ["LLM calls", "Tool calls", "Tokens", "Cost", "Latency"]),
                ("Checks (scored by code)", ["Real prices", "Budget", "Invented", "Asked when needed", "Stopped at"]),
                ("Takeaway", ["Takeaway"])]
    rows = []
    for title, names in sections:
        rows.append(f"<tr class='sec'><th colspan='6'>{title}</th></tr>")
        for name in names:
            cls = " class='take'" if name == "Takeaway" else ""
            tds = "".join(f"<td class='num'>{by_level[lvl].get(name, '<span class=muted>not run</span>')}</td>" for lvl in LEVELS)
            rows.append(f"<tr{cls}><th class='rowh'>{'' if name == 'Takeaway' else name}</th>{tds}</tr>")
    st.markdown(f"<div class='tablewrap'><table class='ctab'><thead><tr><th class='rowh'></th>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


def compare_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Compare the levels")
    st.markdown("<div class='lede'>The Start page's five levels, run on the same request. Each column is the "
                "project's own code at that level, and plain code (not an LLM) checks the answers: are the prices "
                "real, is it within budget, did anything get invented, and did it ask you when it should have?</div>",
                unsafe_allow_html=True)

    text = st.text_input("Trip request", DEFAULT_REQUEST)
    pending = [lvl for lvl in LEVELS if not is_cached(lvl, text)]
    estimate = sum(ESTIMATED_COST[lvl] for lvl in pending)
    a, b = st.columns([1, 3], vertical_alignment="center")
    with a:
        run = st.button("Run all", type="primary", width="stretch", icon=":material/play_arrow:", disabled=not pending or not LIVE)
    with b:
        if pending:
            st.caption(safe_md(f"Estimated cost: about ${estimate:.3f} for {len(pending)} level(s) not yet run on this "
                               "request (gpt-4o-mini). Cached levels are free. A new city may use a few SerpApi searches."))
        else:
            st.caption("All five levels are cached for this request: showing saved results, no cost.")

    if run:
        with st.status("Running the five levels…", expanded=True) as status:
            for lvl in pending:
                status.update(label=f"Running {lvl} · {LEVELS[lvl]}…")
                result = run_level(lvl, text)
                status.write(safe_md(f"{lvl} · {LEVELS[lvl]}: {result.llm_calls} LLM calls, {result.tool_calls} tool "
                                     f"calls, ${result.cost_usd:.4f}, {result.seconds:.1f}s"))
            status.update(label="All five levels done", state="complete", expanded=False)
        st.rerun()

    results = {lvl: run_level(lvl, text) if is_cached(lvl, text) else None for lvl in LEVELS}
    needed = decision_needed(results["L5"]) if results["L5"] else False
    checks = {lvl: score(r, text, needed) for lvl, r in results.items() if r}
    results_table(results, checks)

    st.markdown("#### The answers")
    cols = st.columns(5, gap="small")
    for col, lvl in zip(cols, LEVELS):
        with col, st.container(height=360, border=True):
            st.markdown(f"**{lvl} · {LEVELS[lvl]}**")
            r = results[lvl]
            st.markdown(safe_md(r.answer) if r and r.answer else "_(not run yet)_")

    st.caption("How the checks work: a price counts as real only if a tool returned it (± \\$1); totals the LLM added "
               "up itself don't count. 'Asked when needed' uses the app's own validation and budget code as the "
               "judge of whether a human decision was needed. The app (L5) runs on throwaway memory and stops at "
               "its first pause, so nothing is saved or booked.")
