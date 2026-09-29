"""'Architecture at a glance' for the System diagram page: the whole system in one layered picture.

Adapted from the architecture artifact drawn for this project, restyled with the app's own design
tokens (ui/theme.py: --agent = LLM, --code = plain code, --human = pauses for the traveler) and kept
current: the input guardrail, fenced tool output, model fallback and run caps are drawn in.
Rendered with st.html (not Markdown), so "$" and inline SVG are safe. All classes are prefixed
"ar-" so nothing collides with the rest of the app's CSS.
"""

import base64
from pathlib import Path

from ui.theme import FONT_MONO, tokens

ROOT = Path(__file__).resolve().parent.parent

CSS = """<style>
.ar { --ar-band: color-mix(in srgb, var(--muted) 9%, transparent); --ar-panel: var(--card);
      font-family: var(--font-body); font-size: 14px; line-height: 1.5; color: inherit;
      display: flex; flex-direction: column; gap: 22px; }
.ar code { font-family: var(--font-mono); font-size: .9em; }
.ar-eyebrow { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
.ar-phases { display: grid; grid-template-columns: repeat(13, minmax(0, 1fr)); gap: 3px; }
.ar-ph { height: 28px; border-radius: 3px; background: var(--ar-band); border: 1px solid var(--line);
         font-family: var(--font-mono); font-size: 11px; display: grid; place-items: center; color: var(--muted); }
.ar-ph.done { background: var(--success-soft); border-color: var(--success); color: var(--success); font-weight: 600; }
.ar-ph.now { background: var(--human-soft); border-color: var(--human); color: var(--human); font-weight: 600; }
.ar-legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 8px; }
.ar-legend span { display: inline-flex; align-items: center; gap: 6px; }
.ar-sw { width: 12px; height: 12px; border-radius: 2px; border: 1px solid; display: inline-block; }
.ar-stack { display: flex; flex-direction: column; }
.ar-layer { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 16px; background: var(--ar-band);
            border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
.ar-lab { display: flex; flex-direction: column; gap: 2px; }
.ar-lab b { font-family: var(--font-heading); font-size: 17px; font-weight: 600; }
.ar-lab small { color: var(--muted); font-size: 12px; }
.ar-row { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); }
.ar-row.wide { grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
.ar-blk { background: var(--ar-panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px;
          display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.ar-blk h3 { margin: 0; font-family: var(--font-mono) !important; font-size: 13px !important; font-weight: 600 !important;
             overflow-wrap: anywhere; }
.ar-blk .ar-file { font-family: var(--font-mono); font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }
.ar-blk p { margin: 0; font-size: 12.5px; color: var(--muted); }
.ar-blk .ar-flow { font-family: var(--font-mono); font-size: 11px; background: var(--ar-band); border-radius: 4px;
                   padding: 6px 8px; white-space: pre; overflow-x: auto; line-height: 1.55; }
.ar-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 2px; }
.ar-chip { font-family: var(--font-mono); font-size: 10.5px; padding: 1px 6px; border-radius: 3px;
           background: var(--ar-band); border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
.ar-chip.llm { color: var(--agent); border-color: var(--agent); background: var(--agent-soft); }
.ar-chip.hitl { color: var(--human); border-color: var(--human); background: var(--human-soft); }
.ar-chip.guard { color: var(--code); border-color: var(--code); background: var(--code-soft); }
.ar-blk.human { border-color: var(--human); }
.ar-blk.llm { border-color: var(--agent); }
.ar-blk.guard { border-color: var(--code); }
.ar-blk.dim { opacity: .78; border-style: dashed; }
.ar-conn { display: flex; align-items: center; gap: 10px; padding: 6px 0 6px 182px; color: var(--muted); font-size: 12px; }
.ar-conn::before { content: ""; width: 2px; height: 24px; background: var(--line); flex: none; }
.ar-group { display: flex; flex-direction: column; gap: 10px; }
.ar-gt { font-family: var(--font-mono); font-size: 11px; color: var(--muted); letter-spacing: .03em; }
.ar-graph { background: var(--ar-panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px; overflow-x: auto; }
.ar-graph svg { display: block; width: 100%; min-width: 660px; height: auto; }
.ar-cap { font-size: 12px; color: var(--muted); margin: 6px 2px 0; }
.ar-split { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(0, 1fr); gap: 12px; }
.ar-side { display: flex; flex-direction: column; gap: 10px; }
.ar svg text { font-family: var(--font-mono); font-size: 12px; fill: currentColor; }
.ar svg .lbl { font-size: 10.5px; fill: var(--muted); }
.ar svg .n-llm { fill: var(--agent-soft); stroke: var(--agent); stroke-width: 1.4; }
.ar svg .n-code { fill: var(--code-soft); stroke: var(--code); stroke-width: 1.2; }
.ar svg .n-hum { fill: var(--human-soft); stroke: var(--human); stroke-width: 1.5; }
.ar svg .term { fill: var(--ar-band); stroke: var(--muted); stroke-width: 1.2; }
.ar svg .edge { fill: none; stroke: var(--muted); stroke-width: 1.3; }
.ar svg .edge-soft { fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 4 3; }
.ar svg .ah { fill: var(--muted); }
.ar-notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; }
.ar-notes div { border-top: 2px solid var(--muted); padding-top: 8px; }
.ar-notes h2 { font-family: var(--font-heading) !important; font-size: 16px !important; margin: 0 0 4px !important; }
.ar-notes p { margin: 0; color: var(--muted); font-size: 13px; }
@media (max-width: 820px) { .ar-layer, .ar-split { grid-template-columns: 1fr; } .ar-conn { padding-left: 16px; }
  .ar-phases { grid-template-columns: repeat(7, minmax(0, 1fr)); } }
</style>"""

ROUTER_SVG = """<svg viewBox="0 0 960 300" role="img" aria-label="Router graph: START, load_profile, input_guard (blocked messages go to fallback), classify_intent, then intake and planner and save_profile, booking, guide, or fallback, then END.">
<defs><marker id="ar-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<circle class="term" cx="28" cy="150" r="18"/><text x="28" y="153" text-anchor="middle" style="font-size:8.5px">START</text>
<line class="edge" x1="46" y1="150" x2="58" y2="150" marker-end="url(#ar-a)"/>
<rect class="n-code" x="60" y="128" width="110" height="44" rx="5"/><text x="115" y="154" text-anchor="middle">load_profile</text>
<text class="lbl" x="115" y="190" text-anchor="middle">reads memory.db</text>
<line class="edge" x1="170" y1="150" x2="188" y2="150" marker-end="url(#ar-a)"/>
<rect class="n-code" x="190" y="128" width="110" height="44" rx="5"/><text x="245" y="148" text-anchor="middle">input_guard</text>
<text class="lbl" x="245" y="163" text-anchor="middle">code guardrail</text>
<line class="edge" x1="300" y1="150" x2="318" y2="150" marker-end="url(#ar-a)"/>
<rect class="n-llm" x="320" y="128" width="130" height="44" rx="5"/><text x="385" y="148" text-anchor="middle">classify_intent</text>
<text class="lbl" x="385" y="163" text-anchor="middle">1 structured call</text>
<path class="edge" d="M450,150 C478,150 472,40 498,40" marker-end="url(#ar-a)"/>
<path class="edge" d="M450,150 C478,150 472,110 498,110" marker-end="url(#ar-a)"/>
<path class="edge" d="M450,150 C478,150 472,180 498,180" marker-end="url(#ar-a)"/>
<path class="edge" d="M450,150 C478,150 472,250 498,250" marker-end="url(#ar-a)"/>
<text class="lbl" x="495" y="30" text-anchor="end">plan_trip</text><text class="lbl" x="495" y="100" text-anchor="end">book</text>
<text class="lbl" x="495" y="198" text-anchor="end">travel_question</text><text class="lbl" x="495" y="268" text-anchor="end">other</text>
<rect class="n-hum" x="500" y="20" width="110" height="40" rx="5"/><text x="555" y="44" text-anchor="middle">intake</text>
<rect class="n-hum" x="500" y="90" width="110" height="40" rx="5"/><text x="555" y="114" text-anchor="middle">booking</text>
<text class="lbl" x="555" y="146" text-anchor="middle">only if plan approved</text>
<rect class="n-llm" x="500" y="160" width="110" height="40" rx="5"/><text x="555" y="184" text-anchor="middle">guide</text>
<rect class="n-code" x="500" y="230" width="110" height="40" rx="5"/><text x="555" y="254" text-anchor="middle">fallback</text>
<path class="edge-soft" d="M245,172 C245,246 400,256 498,254" marker-end="url(#ar-a)"/>
<text class="lbl" x="268" y="224">blocked → fallback</text>
<line class="edge" x1="610" y1="40" x2="638" y2="40" marker-end="url(#ar-a)"/><text class="lbl" x="624" y="32" text-anchor="middle">valid</text>
<rect class="n-hum" x="640" y="20" width="100" height="40" rx="5"/><text x="690" y="44" text-anchor="middle">planner</text>
<line class="edge" x1="740" y1="40" x2="768" y2="40" marker-end="url(#ar-a)"/>
<rect class="n-code" x="770" y="20" width="110" height="40" rx="5"/><text x="825" y="44" text-anchor="middle">save_profile</text>
<text class="lbl" x="825" y="76" text-anchor="middle">writes after approval</text>
<path class="edge" d="M880,40 C908,40 930,90 930,128" marker-end="url(#ar-a)"/>
<path class="edge" d="M610,110 C750,110 860,140 910,146" marker-end="url(#ar-a)"/>
<path class="edge" d="M610,180 C750,180 860,158 910,154" marker-end="url(#ar-a)"/>
<path class="edge" d="M610,250 C800,250 930,220 930,172" marker-end="url(#ar-a)"/>
<path class="edge-soft" d="M555,60 C555,72 570,78 590,78 L620,78" marker-end="url(#ar-a)"/><text class="lbl" x="626" y="82">gave up → END</text>
<circle class="term" cx="930" cy="150" r="18"/><text x="930" y="153" text-anchor="middle" style="font-size:9px">END</text>
</svg>"""

PLANNER_SVG = """<svg viewBox="0 0 800 250" role="img" aria-label="Planner: research, budget_check; fits goes to review then approved; over budget or bad evidence replans; still over escalates; review can send change requests back to research.">
<defs><marker id="ar-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<text class="lbl" x="10" y="16">trip_planner.py · research decides what to pick; code checks, replans and gates</text>
<circle class="term" cx="28" cy="110" r="18"/><text x="28" y="113" text-anchor="middle" style="font-size:8.5px">START</text>
<line class="edge" x1="46" y1="110" x2="78" y2="110" marker-end="url(#ar-b)"/>
<rect class="n-llm" x="80" y="88" width="130" height="44" rx="5"/><text x="145" y="107" text-anchor="middle">research</text>
<text class="lbl" x="145" y="122" text-anchor="middle">create_agent → TripPlan</text>
<line class="edge" x1="210" y1="110" x2="268" y2="110" marker-end="url(#ar-b)"/>
<rect class="n-code" x="270" y="88" width="140" height="44" rx="5"/><text x="340" y="107" text-anchor="middle">budget_check</text>
<text class="lbl" x="340" y="122" text-anchor="middle">verify_plan + costs</text>
<line class="edge" x1="410" y1="110" x2="488" y2="110" marker-end="url(#ar-b)"/><text class="lbl" x="449" y="102" text-anchor="middle">fits</text>
<rect class="n-hum" x="490" y="88" width="120" height="44" rx="5"/><text x="550" y="107" text-anchor="middle">review</text>
<text class="lbl" x="550" y="122" text-anchor="middle">approve / pick / change</text>
<line class="edge" x1="610" y1="110" x2="668" y2="110" marker-end="url(#ar-b)"/>
<rect class="n-code" x="670" y="88" width="110" height="44" rx="5"/><text x="725" y="115" text-anchor="middle">approved</text>
<path class="edge" d="M540,88 C520,36 170,36 150,86" marker-end="url(#ar-b)"/>
<text class="lbl" x="345" y="42" text-anchor="middle">change request: feedback becomes a hard constraint, retries reset</text>
<path class="edge" d="M320,132 C310,170 270,196 232,200" marker-end="url(#ar-b)"/>
<text class="lbl" x="175" y="240" text-anchor="middle">over budget or bad evidence · ≤2 retries</text>
<text class="lbl" x="420" y="160">still over</text>
<rect class="n-code" x="120" y="182" width="110" height="40" rx="5"/><text x="175" y="206" text-anchor="middle">replan</text>
<path class="edge" d="M150,182 C140,165 140,150 142,134" marker-end="url(#ar-b)"/>
<path class="edge" d="M380,132 C400,168 430,190 468,200" marker-end="url(#ar-b)"/>
<rect class="n-hum" x="470" y="182" width="120" height="40" rx="5"/><text x="530" y="206" text-anchor="middle">escalate</text>
<text class="lbl" x="530" y="238" text-anchor="middle">raise budget · accept · cancel</text>
<path class="edge" d="M560,182 C565,165 560,150 556,134" marker-end="url(#ar-b)"/><text class="lbl" x="572" y="160">accept</text>
</svg>"""


def svg_image(svg: str) -> str:
    """st.html strips inline <svg>, so each graph is an <img> with a data URI. Page CSS can't reach
    inside an image, so the current theme's colours are written into the SVG's own <style>."""
    t = tokens()
    style = f"""<style>
      text {{ font-family: {FONT_MONO}; font-size: 12px; fill: {t['ink']}; }}
      .lbl {{ font-size: 10.5px; fill: {t['muted']}; }}
      .n-llm {{ fill: {t['agent-soft']}; stroke: {t['agent']}; stroke-width: 1.4; }}
      .n-code {{ fill: {t['code-soft']}; stroke: {t['code']}; stroke-width: 1.2; }}
      .n-hum {{ fill: {t['human-soft']}; stroke: {t['human']}; stroke-width: 1.5; }}
      .term {{ fill: {t['line']}; stroke: {t['muted']}; stroke-width: 1.2; }}
      .edge {{ fill: none; stroke: {t['muted']}; stroke-width: 1.3; }}
      .edge-soft {{ fill: none; stroke: {t['muted']}; stroke-width: 1.2; stroke-dasharray: 4 3; }}
      .ah {{ fill: {t['muted']}; }}
    </style>"""
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace("<defs>", style + "<defs>", 1)
    label = svg.split('aria-label="', 1)[1].split('"', 1)[0] if 'aria-label="' in svg else "diagram"
    data = base64.b64encode(svg.encode()).decode()
    return f'<img alt="{label}" src="data:image/svg+xml;base64,{data}" style="width:100%;min-width:660px;display:block">'


def _line_count() -> int:
    folders = [ROOT / "src", ROOT / "lab", ROOT / "ui", ROOT / "scripts", ROOT / "tests", ROOT / "evals"]
    files = [p for f in folders if f.exists() for p in f.rglob("*.py")] + [ROOT / "app.py"]
    return sum(len(p.read_text(errors="ignore").splitlines()) for p in files if p.exists())


def _phases(done_through: int = 10, in_progress: int = 11) -> str:
    cells = []
    for i in range(13):
        cls = "done" if i <= done_through else ("now" if i == in_progress else "")
        cells.append(f"<div class='ar-ph {cls}'>{i}</div>")
    return "".join(cells)


def architecture_html() -> str:
    lines = _line_count()
    return CSS + f"""
<div class="ar">
 <div>
  <div class="ar-eyebrow">travel_planner · src/ · app.py · lab/ · ui/ · scripts/ · tests/ · about {lines:,} lines of Python</div>
  <div class="ar-phases" style="margin-top:10px">{_phases()}</div>
  <div class="ar-legend">
   <span><i class="ar-sw" style="background:var(--success-soft);border-color:var(--success)"></i>Built: phases 0–10</span>
   <span><i class="ar-sw" style="background:var(--human-soft);border-color:var(--human)"></i>Now: 11 robustness, guardrails, evals</span>
   <span><i class="ar-sw" style="background:var(--agent-soft);border-color:var(--agent)"></i>Calls the LLM</span>
   <span><i class="ar-sw" style="background:var(--code-soft);border-color:var(--code)"></i>Plain code / guardrail</span>
   <span><i class="ar-sw" style="background:var(--human-soft);border-color:var(--human)"></i>Pauses for the traveler</span>
  </div>
 </div>

 <div class="ar-stack">
  <section class="ar-layer">
   <div class="ar-lab"><b>Front ends</b><small>app.py, lab/, ui/, scripts/, tests/</small></div>
   <div class="ar-row">
    <div class="ar-blk llm"><h3>Streamlit app</h3><span class="ar-file">app.py → lab/assistant_page.py</span>
     <p>Trip form, option pickers, approve and book buttons. Every click becomes a message or a <code>Command(resume)</code>. The trip id is kept in the URL.</p>
     <div class="ar-meta"><span class="ar-chip">Phase 10</span></div></div>
    <div class="ar-blk"><h3>Overview and Agent Lab</h3><span class="ar-file">lab/compare_page · overview · lab_pages</span>
     <p>Compare L1–L5, this diagram, lessons, usage and cost, and the step-by-step Agent Lab tour.</p></div>
    <div class="ar-blk llm"><h3>phase9_assistant.py</h3><span class="ar-file">scripts/</span>
     <p>Terminal chat with the same graph. <code>--thread</code> resumes a saved pause; <code>--profile</code> shows long-term memory.</p></div>
    <div class="ar-blk"><h3>tests/</h3>
     <p>Every component of <code>build_assistant</code> is injectable, so the whole router runs with fakes and no API calls. Failures are mocked too.</p>
     <div class="ar-meta"><span class="ar-chip">pytest</span></div></div>
   </div>
  </section>
  <div class="ar-conn"><span><code>build_assistant(open_checkpointer(), open_store())</code> · config carries <code>thread_id</code> and <code>user_id</code> · per-run token/$ caps</span></div>

  <section class="ar-layer">
   <div class="ar-lab"><b>Router graph</b><small>agents/assistant.py · Phase 9</small></div>
   <div class="ar-group">
    <div><div class="ar-graph">{svg_image(ROUTER_SVG)}</div>
     <p class="ar-cap">One checkpointer sits on this graph. Sub-agents are compiled without one and inherit it, so a pause deep inside the planner or booking agent still resumes after a restart.</p></div>
    <div class="ar-row">
     <div class="ar-blk guard"><h3>input_guard</h3><span class="ar-file">guardrails.check_input</span>
      <p>Code, before any LLM: blocks abusive, injection-style and off-topic messages, and booking anything but the approved plan.</p>
      <div class="ar-meta"><span class="ar-chip guard">guardrail</span></div></div>
     <div class="ar-blk llm"><h3>IntentClassification</h3><span class="ar-file">CLASSIFIER_PROMPT</span>
      <p><code>plan_trip | book | travel_question | other</code>, plus the city. It only routes; <code>book</code> is refused in code unless a plan is approved.</p></div>
     <div class="ar-blk"><h3>Three kinds of memory</h3>
      <p>Working: this turn. Session: the checkpointed thread. Long-term: the profile (home airport, usual travelers), filled into intake.</p></div>
    </div>
   </div>
  </section>
  <div class="ar-conn"><span>Each route node calls a compiled sub-agent with <code>.invoke(state, config)</code>. Its interrupts bubble up through the router.</span></div>

  <section class="ar-layer">
   <div class="ar-lab"><b>Sub-agents</b><small>agents/ · phases 4–8</small></div>
   <div class="ar-group">
    <div class="ar-graph">{svg_image(PLANNER_SVG)}</div>
    <p class="ar-cap">A replan changes only the hotel, capped at the nightly price that fits the budget; code carries the verified flight and sights over. <code>cancelled</code>, <code>failed</code> and <code>stopped</code> (run cap) are left out of the drawing.</p>
    <div class="ar-row wide">
     <div class="ar-blk human"><h3>Trip intake</h3><span class="ar-file">trip_intake.py · Phase 4</span>
      <div class="ar-flow">extract ─▶ validate ─▶ confirm
   ▲          └─▶ ask ⏸ (≤3 rounds)
   └───────────────┘   └─▶ give_up</div>
      <p>The LLM extracts a <code>TripRequestDraft</code>; <code>validate_draft</code> enforces every assumption in code, and code decides when to ask.</p>
      <div class="ar-meta"><span class="ar-chip llm">structured output</span><span class="ar-chip hitl">interrupt</span></div></div>
     <div class="ar-blk llm"><h3>Research agent</h3><span class="ar-file">research_agent.py · Phase 5</span>
      <div class="ar-flow">create_agent(model ⇄ tools)
  + ToolOutputGuard (fence, strip)
  + retry ×2 → fallback model
  response: ToolStrategy(TripPlan)</div>
      <p>Every tool result is kept as evidence for <code>verify_plan</code>, and treated as data, never instructions.</p>
      <div class="ar-meta"><span class="ar-chip llm">create_agent</span><span class="ar-chip guard">guarded</span></div></div>
     <div class="ar-blk human"><h3>Booking agent</h3><span class="ar-file">booking_agent.py · Phase 7</span>
      <div class="ar-flow">create_agent(model ⇄ tools)
  + HumanInTheLoopMiddleware
    book_flight ⏸  book_hotel ⏸
    approve · edit · reject</div>
      <p>The gate is configuration, not an <code>interrupt()</code> in our code. Rejections are logged with a reason.</p>
      <div class="ar-meta"><span class="ar-chip llm">create_agent</span><span class="ar-chip hitl">middleware</span></div></div>
     <div class="ar-blk llm"><h3>Guide agent</h3><span class="ar-file">guide_agent.py · Phase 8</span>
      <div class="ar-flow">create_agent(model ⇄ search_travel_guide)
  + ToolOutputGuard
  response: GuideAnswer
    answer · citations · from_guides</div>
      <p>Agentic RAG. Code verifies each citation was retrieved, appends the visa note, and the output guard checks prices.</p>
      <div class="ar-meta"><span class="ar-chip llm">create_agent</span><span class="ar-chip">RAG</span></div></div>
     <div class="ar-blk dim"><h3>ReAct agent</h3><span class="ar-file">react_agent.py · phases 2–3</span>
      <p>The hand-wired reason ⇄ action loop. Not in the app's path; used by the Lab tour and as level L4 in <code>compare.py</code>.</p></div>
    </div>
   </div>
  </section>
  <div class="ar-conn"><span>tools return <code>{{"status": "ok" | "error"}}</code> · tool text is fenced as untrusted · code checks every number the LLM produces</span></div>

  <section class="ar-layer">
   <div class="ar-lab"><b>Tools and domain code</b><small>tools/ and plain Python modules</small></div>
   <div class="ar-group">
    <div class="ar-gt">TOOLS · wrapped by @returns_errors_as_data</div>
    <div class="ar-row">
     <div class="ar-blk"><h3>Read tools</h3><span class="ar-file">flights · hotels · activities · geo</span>
      <p><code>search_flights</code>, <code>search_hotels</code>, <code>search_top_sights</code>, <code>get_weather</code>. Also <code>search_places_by_interest</code> and <code>geocode</code> for the ReAct agent.</p></div>
     <div class="ar-blk human"><h3>Write tools</h3><span class="ar-file">tools/booking.py</span>
      <p><code>book_flight</code>, <code>book_hotel</code>: only behind the approval middleware, and idempotent.</p></div>
     <div class="ar-blk"><h3>search_travel_guide</h3><span class="ar-file">guide_agent.make_guide_tool</span>
      <p>City-filtered vector search; the confidence band decides proceed, confirm or stop.</p></div>
    </div>
    <div class="ar-gt">DOMAIN AND SAFETY CODE · no LLM</div>
    <div class="ar-row">
     <div class="ar-blk"><h3>models.py</h3><p><code>TripRequest</code>, <code>TripPlan</code>, <code>CostBreakdown</code>; <code>validate_draft</code>, <code>verify_plan</code>, <code>cost_breakdown</code>.</p></div>
     <div class="ar-blk"><h3>itinerary.py</h3><p><code>stay_dates</code> checks in the day the flight lands; <code>fit_hotel_to_stay</code> re-prices; <code>build_itinerary</code> lays out the days.</p></div>
     <div class="ar-blk guard"><h3>guardrails.py</h3><p>Input check, tool-output fencing and injection stripping, output PII and price checks.</p></div>
     <div class="ar-blk guard"><h3>limits.py</h3><p>Per-run token/$ caps and per-session budgets, enforced in code.</p></div>
     <div class="ar-blk"><h3>compare.py · display.py</h3><p>Five levels scored by code; shared plan, cost and decision text.</p></div>
    </div>
   </div>
  </section>
  <div class="ar-conn"><span><code>request(…)</code> with retries and a circuit breaker · <code>llm.py</code> models with fallback · SQLite · <code>usage.record(…)</code></span></div>

  <section class="ar-layer">
   <div class="ar-lab"><b>Storage and infrastructure</b><small>all local files</small></div>
   <div class="ar-row">
    <div class="ar-blk guard"><h3>http_cache.py</h3><span class="ar-file">cache/&lt;hash&gt;.json</span>
     <p>Cache for every provider call; retries with backoff and jitter on 429/5xx; circuit breaker; never raises.</p></div>
    <div class="ar-blk llm"><h3>llm.py</h3><span class="ar-file">gpt-4o-mini → gpt-4.1-mini</span>
     <p>Timeouts, two attempts on the primary model, then the fallback model; the trace shows which answered.</p></div>
    <div class="ar-blk"><h3>rag.py</h3><span class="ar-file">guides/*.md · 8 cities</span>
     <p>Split on headings, <code>InMemoryVectorStore</code>, cached embeddings, cutoffs at 0.30 and 0.45.</p></div>
    <div class="ar-blk human"><h3>persistence.py</h3><span class="ar-file">checkpoints.db · memory.db</span>
     <p><code>SqliteSaver</code> for every step of every thread; <code>SqliteStore</code> for traveler profiles.</p></div>
    <div class="ar-blk"><h3>bookings.py</h3><span class="ar-file">bookings.db · feedback.jsonl</span>
     <p>Simulated ledger with idempotency keys; rejections kept as feedback.</p></div>
    <div class="ar-blk"><h3>usage.py · trace.py</h3><span class="ar-file">usage.jsonl</span>
     <p>One line per API or LLM call; <code>RunStats</code> counts tokens and cost and enforces the run cap.</p></div>
   </div>
  </section>
  <div class="ar-conn"><span>HTTPS</span></div>

  <section class="ar-layer">
   <div class="ar-lab"><b>External services</b><small>network</small></div>
   <div class="ar-row">
    <div class="ar-blk llm"><h3>OpenAI</h3><p><code>gpt-4o-mini</code> for every agent and the classifier, <code>gpt-4.1-mini</code> as fallback, <code>text-embedding-3-small</code> for the guides.</p></div>
    <div class="ar-blk"><h3>Duffel</h3><p>Test-mode flight offers. Synthetic prices.</p></div>
    <div class="ar-blk"><h3>SerpApi</h3><p>Google Hotels and top sights (untrusted text). About 250 free searches a month.</p></div>
    <div class="ar-blk"><h3>Open-Meteo</h3><p>Forecast and archive weather. No key.</p></div>
    <div class="ar-blk dim"><h3>OpenTripMap · Nominatim</h3><p>Behind the ReAct-only tools and the map.</p></div>
   </div>
  </section>
 </div>

 <section class="ar-notes">
  <div><h2>LLM decides, code guarantees</h2><p>Agents choose flights, hotels and sights. Code validates the request, checks each id and price against tool evidence, does the budget math and enforces every limit.</p></div>
  <div><h2>Four ways to pause</h2><p>Intake asks at a fixed node. The planner pauses to escalate and to review. Booking pauses through middleware before each write. All resume through the router's one checkpointer.</p></div>
  <div><h2>Fail safely</h2><p>Retries, a circuit breaker and a fallback model absorb outages; guardrails treat tool text as data; run caps stop runaway spend with a clear message.</p></div>
 </section>
</div>"""
