"""The site's single source of truth for colour and type.

Every page gets these tokens through apply_theme(), which app.main() calls once per run. Page CSS
refers to them as var(--token). No other file should hard-code a colour or font.

Colour tokens have fixed meanings, so a colour always says the same thing:
  --accent    terracotta: buttons, links and the active nav item ONLY
  --agent     the LLM decides (agents, the classifier, extraction)
  --code      code guarantees (validation, budget math, routing rules, evidence checks)
  --human     a human decides (clarifying questions, plan review, escalation, booking approval)
  --flight / --hotel / --sight   the three kinds of thing in an itinerary and on the map
  --success / --error            within budget / over budget, verified / rejected
"""

import streamlit as st

FONT_HEADING = '"Source Serif 4", Georgia, "Times New Roman", serif'
FONT_BODY = '"Hanken Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

LIGHT = {
    "accent": "#C2603E", "on-accent": "#ffffff",
    "agent": "#2a78d6", "agent-soft": "rgba(42,120,214,.10)",
    "code": "#4a3aa7", "code-soft": "rgba(74,58,167,.10)",
    "human": "#a86a12", "human-soft": "rgba(183,121,31,.12)",
    "flight": "#d9477a", "flight-soft": "rgba(217,71,122,.10)",
    "hotel": "#1baf7a", "hotel-soft": "rgba(27,175,122,.12)",
    "sight": "#eb6834", "sight-soft": "rgba(235,104,52,.11)",
    "success": "#1f8a4c", "success-soft": "rgba(31,138,76,.13)",
    "error": "#c4521f", "error-soft": "rgba(196,82,31,.12)",
    "muted": "#6f6b62",
    "line": "rgba(31,30,29,.14)",
    "card": "#ffffff",
    "ink": "#1F1E1D",
}
DARK = {
    "accent": "#D97757", "on-accent": "#1F1E1D",
    "agent": "#3987e5", "agent-soft": "rgba(57,135,229,.16)",
    "code": "#9085e9", "code-soft": "rgba(144,133,233,.16)",
    "human": "#d9a441", "human-soft": "rgba(217,164,65,.16)",
    "flight": "#e0679a", "flight-soft": "rgba(224,103,154,.16)",
    "hotel": "#199e70", "hotel-soft": "rgba(25,158,112,.18)",
    "sight": "#e58050", "sight-soft": "rgba(229,128,80,.16)",
    "success": "#3fb67a", "success-soft": "rgba(63,182,122,.18)",
    "error": "#e66a3d", "error-soft": "rgba(230,106,61,.18)",
    "muted": "#a8a397",
    "line": "rgba(242,240,232,.16)",
    "card": "#30302E",
    "ink": "#F2F0E8",
}

# RGB versions of the travel colours for the map (pydeck needs [r, g, b])
MAP_RGB = {"hotel": [27, 175, 122], "sight": [235, 104, 52]}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tokens() -> dict[str, str]:
    return DARK if mode() == "dark" else LIGHT


def apply_theme() -> None:
    """Inject the tokens and the shared typography / component styles. Call once per run."""
    t = tokens()
    variables = "\n".join(f"    --{k}: {v};" for k, v in t.items())
    st.markdown(f"""<style>
  :root {{
{variables}
    --font-heading: {FONT_HEADING};
    --font-body: {FONT_BODY};
    --font-mono: {FONT_MONO};
  }}

  /* ---- Typography: one serif for headings, one sans for everything else */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{ font-family: var(--font-body); }}
  h1, h2, h3, h4, [data-testid="stHeading"] * {{ font-family: var(--font-heading) !important; }}
  h1 {{ font-size: 2.1rem !important; font-weight: 600 !important; line-height: 1.2 !important; }}
  h2 {{ font-size: 1.55rem !important; font-weight: 600 !important; }}
  h3 {{ font-size: 1.22rem !important; font-weight: 600 !important; }}
  h4 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
  code, pre {{ font-family: var(--font-mono); }}
  [data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-left: 1.5rem; padding-right: 1.5rem; }}
  [data-testid^="stBaseButton"] * , [data-testid="stWidgetLabel"] * {{ white-space: normal !important; }}
  [data-testid^="stBaseButton"] {{ height: auto; min-height: 2.5rem; }}

  /* ---- Shared components */
  .muted {{ color: var(--muted); }}
  .lede {{ font-size: 1.1rem; line-height: 1.6; color: var(--muted); max-width: 68ch; }}
  .eyebrow {{ font: 600 .74rem/1 var(--font-body); letter-spacing: .12em; text-transform: uppercase;
             color: var(--muted); margin-bottom: .6rem; }}
  .hero {{ font-family: var(--font-heading); font-size: clamp(2rem, 4vw, 2.8rem); line-height: 1.12;
          font-weight: 600; margin: 0 0 .8rem; text-wrap: balance; }}
  .badge {{ display: inline-block; font-size: .76rem; font-weight: 600; padding: 1px 9px; border-radius: 99px;
            white-space: nowrap; }}
  .badge.agent {{ background: var(--agent-soft); color: var(--agent); }}
  .badge.code {{ background: var(--code-soft); color: var(--code); }}
  .badge.human {{ background: var(--human-soft); color: var(--human); }}
  .badge.ok {{ background: var(--success-soft); color: var(--success); }}
  .badge.no {{ background: var(--error-soft); color: var(--error); }}
  .badge + .badge {{ margin-left: .3rem; }}
  .section-label {{ font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                    color: var(--muted); margin: .8rem 0 .35rem; }}

  /* ---- Site chrome (ui/chrome.py) */
  .author-name {{ font-family: var(--font-heading); font-size: 1.05rem; font-weight: 600; margin-bottom: .35rem; }}
  .site-footer {{ margin-top: 3rem; padding-top: .9rem; border-top: 1px solid var(--line); color: var(--muted);
                  font-size: .8rem; text-align: center; }}

  /* ---- Concept ladder (landing page): LLM -> RAG -> tools -> agent -> multi-agent */
  .ladder {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: .6rem 0 1.2rem; }}
  @media (max-width: 1100px) {{ .ladder {{ grid-template-columns: repeat(2, 1fr); }} }}
  .rung {{ border: 1px solid var(--line); border-radius: .6rem; padding: .8rem .85rem; position: relative; }}
  .rung .n {{ font-family: var(--font-heading); font-size: 1.5rem; font-weight: 600; color: var(--muted); line-height: 1; }}
  .rung .t {{ font-family: var(--font-heading); font-size: 1.08rem; font-weight: 600; margin: .25rem 0 .35rem; }}
  .rung .flow {{ font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-bottom: .45rem; }}
  .rung .adds {{ font-size: .84rem; line-height: 1.4; }}
  .rung .cant {{ font-size: .8rem; color: var(--error); margin-top: .45rem; line-height: 1.35; }}
  .rung.this {{ border: 2px solid var(--accent); background: var(--agent-soft); }}
  .rung.this .n {{ color: var(--accent); }}
  .rung .here {{ position: absolute; top: .55rem; right: .6rem; }}

  /* ---- Actor steps (who does what): agent / code / human */
  .steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  @media (max-width: 1000px) {{ .steps {{ grid-template-columns: 1fr; }} }}
  .steps > div {{ padding-top: .5rem; }}
  .steps .agent {{ border-top: 3px solid var(--agent); }}
  .steps .code {{ border-top: 3px solid var(--code); }}
  .steps .human {{ border-top: 3px solid var(--human); }}
  .steps b {{ display: block; font-size: .95rem; margin-bottom: .2rem; }}
  .steps span {{ font-size: .86rem; color: var(--muted); }}

  /* ---- Itinerary: one column per day */
  .itin {{ display: flex; gap: 10px; overflow-x: auto; padding-bottom: .5rem; margin: .3rem 0 .6rem; }}
  .day {{ flex: 1 0 185px; max-width: 260px; border: 1px solid var(--line); border-radius: .6rem;
          display: flex; flex-direction: column; background: var(--card); }}
  .day .head {{ padding: .55rem .7rem .45rem; border-bottom: 1px solid var(--line); }}
  .day .dn {{ font-family: var(--font-heading); font-size: 1.12rem; font-weight: 600; }}
  .day .dd {{ font-size: .78rem; color: var(--muted); }}
  .day .body {{ padding: .5rem .55rem; display: flex; flex-direction: column; gap: .4rem; flex: 1; }}
  .ev {{ border-left: 3px solid var(--line); padding: .25rem .45rem; border-radius: 3px; font-size: .82rem; line-height: 1.35; }}
  .ev .et {{ font-weight: 600; }}
  .ev .ed {{ color: var(--muted); font-size: .76rem; }}
  .ev.flight {{ border-color: var(--flight); background: var(--flight-soft); }}
  .ev.checkin, .ev.checkout {{ border-color: var(--hotel); background: var(--hotel-soft); }}
  .ev.sight {{ border-color: var(--sight); background: var(--sight-soft); }}
  .ev.free {{ border-color: var(--line); color: var(--muted); }}
  .day .stay {{ margin: 0 .55rem .55rem; padding: .3rem .5rem; border-radius: 99px; font-size: .74rem;
                background: var(--hotel-soft); color: var(--hotel); font-weight: 600; text-align: center; }}
  .day .stay.none {{ background: transparent; color: var(--muted); border: 1px dashed var(--line); }}

  /* ---- Cost bar */
  .costbar {{ display: flex; height: 14px; border-radius: 99px; overflow: hidden; background: var(--line); margin: .3rem 0 .2rem; }}
  .costbar > div {{ height: 100%; }}
  .costlegend {{ display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .8rem; color: var(--muted); }}
  .costlegend i {{ display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .3rem;
                   vertical-align: -1px; }}

  /* ---- "Your turn" panel: whenever the agent is waiting on the traveler */
  [class*="st-key-yourturn"] {{ border: 2px solid var(--human) !important; background: var(--human-soft); }}
  .turn-head {{ font-family: var(--font-heading); font-size: 1.15rem; font-weight: 600; color: var(--human); }}

  /* ---- Trace list */
  .trace {{ font-size: .82rem; line-height: 1.5; }}
  .trace .row {{ display: flex; gap: .5rem; align-items: baseline; border-bottom: 1px solid var(--line); padding: .2rem 0; }}
  .trace .node {{ font-family: var(--font-mono); font-size: .76rem; min-width: 15rem; }}
  .trace .what {{ color: var(--muted); }}

  /* ---- Decision bubbles: every time you answered the agent (orange = human) */
  [class*="st-key-decision_"] [data-testid="stChatMessage"] {{ border-left: 4px solid var(--human);
     background: var(--human-soft); border-radius: .6rem; }}
  .decision-label {{ font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                     color: var(--human); margin-bottom: .1rem; }}
  .trace-turn {{ margin: .9rem 0 .3rem; font-size: .9rem; }}
  .trace-node {{ font-family: var(--font-mono); font-size: .76rem; }}
  .trace-what {{ font-size: .8rem; color: var(--muted); }}
  .trace-ms {{ font-family: var(--font-mono); font-size: .74rem; color: var(--muted); }}

  /* ---- Lab "learn" + tables */
  .tablewrap {{ overflow-x: auto; }}
  .watch {{ border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: .6rem .9rem;
            min-height: 8rem; }}
  .ww-row {{ display: grid; grid-template-columns: 4.6rem 13rem 1fr; gap: .6rem; align-items: center;
             padding: .22rem 0; border-bottom: 1px dashed var(--line); opacity: 0;
             animation: ww-in .45s ease-out both; }}
  .ww-row.ww-user {{ grid-template-columns: 4.6rem 1fr; font-weight: 600; }}
  .ww-row.ww-done {{ display: block; border-bottom: 0; padding-top: .5rem; color: var(--success); }}
  @keyframes ww-in {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}
  @media (max-width: 640px) {{ .ww-row {{ grid-template-columns: 4.2rem 1fr; }} .ww-row .trace-what {{ grid-column: 2; }} }}
  @media (prefers-reduced-motion: reduce) {{ .ww-row {{ animation: none; opacity: 1; }} }}
  .learnbox {{ border: 1px solid var(--line); border-left: 3px solid var(--agent); border-radius: 8px;
               background: var(--card); padding: .7rem 1rem .4rem; margin: .8rem 0 .6rem; max-width: 72ch; }}
  .learnbox ul {{ margin: .35rem 0 .3rem 1.1rem; padding: 0; }}
  .learnbox li {{ margin: .15rem 0; }}
  .ltable tr.hl td {{ background: var(--agent-soft); }}
  .ltable {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  .ltable th, .ltable td {{ text-align: left; vertical-align: top; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }}
  .ltable thead th {{ border-bottom: 2px solid var(--line); }}
</style>""", unsafe_allow_html=True)
