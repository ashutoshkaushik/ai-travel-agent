"""Compare the levels: the same request at five levels of capability, scored by code.

    L1 Plain LLM      one chat call, no tools
    L2 RAG            retrieve from the city guides, then one chat call
    L3 Tool calling   one forced search_flights call, then one chat call
    L4 Single agent   the Lab step 2 ReAct loop with all tools (react_agent.build_react_agent)
    L5 This app       the full router + planner (agents/assistant.build_assistant), stopped at the
                      first pause: a clarifying question, an escalation, or the plan review. No booking.

Every level reuses the project's own tools and agents; nothing is re-implemented here. Results are
cached on disk per (level, request), so re-running the same comparison is free.

The checks are plain code, never an LLM judging an LLM:
- real prices:   every $ amount in the answer can be traced to a tool result (± $1)
- within budget: the total the answer states (or code computed) vs the requested budget
- invented:      $ amounts that no tool returned (L5: the planner's evidence check)
- asked:         did it ask the traveler when code says a decision was needed
"""

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from travel_planner.cities import SUPPORTED_CITIES
from travel_planner.config import CACHE_DIR, RECORDINGS_DIR
from travel_planner.guardrails import guard_tool_output, unfence
from travel_planner.llm import chat_model, tools_model
from travel_planner.prompts import PLAIN_LLM_PROMPT, RAG_ANSWER_PROMPT, TOOL_ANSWER_PROMPT
from travel_planner.trace import RunStats

COMPARE_CACHE = CACHE_DIR / "compare"
RECORDED = RECORDINGS_DIR / "compare"  # committed results for the default request, so a keyless deploy shows them
LEVELS = {
    "L1": "Plain LLM",
    "L2": "RAG",
    "L3": "Tool calling",
    "L4": "Single agent",
    "L5": "This app",
}
# Rough cost per level with gpt-4o-mini, measured on this project's runs (used for the estimate)
ESTIMATED_COST = {"L1": 0.0004, "L2": 0.0005, "L3": 0.0012, "L4": 0.0025, "L5": 0.0035}


@dataclass
class LevelResult:
    level: str
    answer: str
    llm_calls: int
    tool_calls: int
    tokens: int
    cost_usd: float
    seconds: float
    evidence: list[dict] = field(default_factory=list)  # tool results the answer can be traced to
    paused: str | None = None  # L5: clarify | escalation | review
    code_total: float | None = None  # L5: total computed by code
    code_problems: list[str] = field(default_factory=list)  # L5: evidence-check problems


# ------------------------------------------------------------------ the five levels

def _llm(llm: BaseChatModel | None) -> BaseChatModel:
    return llm or chat_model()


def _tool_results(messages: list) -> list[dict]:
    results = []
    for m in messages:
        content = unfence(m.content) if isinstance(m, ToolMessage) else ""
        if content.startswith("{"):
            data = json.loads(content)
            if data.get("status") == "ok":
                results.append(data)
    return results


def mentioned_city(text: str) -> str | None:
    lowered = text.lower()
    return next((c for c in SUPPORTED_CITIES if c.lower() in lowered), None)


def run_l1(text: str, stats: RunStats, llm: BaseChatModel | None = None) -> tuple[str, list[dict]]:
    reply = _llm(llm).invoke([SystemMessage(PLAIN_LLM_PROMPT.format(todays_date=date.today())), HumanMessage(text)],
                             {"callbacks": [stats]})
    return reply.content, []


def run_l2(text: str, stats: RunStats, llm: BaseChatModel | None = None, index=None) -> tuple[str, list[dict]]:
    from travel_planner.rag import default_index

    hits = (index or default_index()).search(text, mentioned_city(text))
    context = "\n\n".join(f"[{doc.id}] {doc.page_content}" for doc, _ in hits)
    prompt = RAG_ANSWER_PROMPT.format(todays_date=date.today(), context=context)
    reply = _llm(llm).invoke([SystemMessage(prompt), HumanMessage(text)], {"callbacks": [stats]})
    return reply.content, []  # guides hold no prices, so nothing to trace prices to


def run_l3(text: str, stats: RunStats, llm: BaseChatModel | None = None) -> tuple[str, list[dict]]:
    from travel_planner.tools import search_flights

    forced = (llm.bind_tools([search_flights], tool_choice="search_flights") if llm
              else tools_model([search_flights], tool_choice="search_flights"))
    llm = _llm(llm)
    system = SystemMessage(TOOL_ANSWER_PROMPT.format(todays_date=date.today()))
    call = forced.invoke(
        [system, HumanMessage(text)], {"callbacks": [stats]})
    tool_call = call.tool_calls[0]
    result = search_flights.invoke(tool_call["args"], {"callbacks": [stats]})
    tool_msg = ToolMessage(guard_tool_output(json.dumps(result)), tool_call_id=tool_call["id"], name="search_flights")
    reply = llm.invoke([system, HumanMessage(text), call, tool_msg], {"callbacks": [stats]})
    return reply.content, [result] if result.get("status") == "ok" else []


def run_l4(text: str, stats: RunStats, llm: BaseChatModel | None = None) -> tuple[str, list[dict]]:
    from travel_planner.agents.react_agent import build_react_agent
    from travel_planner.tools import ALL_TOOLS

    agent = build_react_agent(ALL_TOOLS, llm=llm)
    result = agent.invoke({"messages": [HumanMessage(text)]}, {"callbacks": [stats], "recursion_limit": 20})
    final = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content), "")
    return final, _tool_results(result["messages"])


def run_l5(text: str, stats: RunStats) -> tuple[str, list[dict], str | None, float | None, list[str]]:
    """The full app on throwaway in-memory state (no saved trips or profiles touched), stopped at the
    first pause. Booking is never reached: it needs an approval this run never gives."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from travel_planner.agents.assistant import build_assistant

    assistant = build_assistant(InMemorySaver(), InMemoryStore())
    config = {"configurable": {"thread_id": "compare", "user_id": "compare"}, "callbacks": [stats]}
    result = assistant.invoke({"messages": [HumanMessage(text)]}, config)
    evidence, problems, total, paused = [], [], None, None
    snapshot = assistant.get_state(config, subgraphs=True)
    stack = [snapshot]
    while stack:  # the planner's own state holds the evidence and the evidence-check problems
        snap = stack.pop()
        for task in snap.tasks:
            if task.state is not None and hasattr(task.state, "values"):
                evidence += task.state.values.get("evidence", []) or []
                problems += task.state.values.get("problems", []) or []
                stack.append(task.state)
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        paused = payload.get("kind", "clarify")
        answer = payload.get("question", "")
        if payload.get("costs"):
            total = payload["costs"]["total"]
    else:
        answer = result["messages"][-1].content
    return answer, evidence, paused, total, problems


def run_level(level: str, text: str, use_cache: bool = True) -> LevelResult:
    """Run one level (or return its cached result for this exact request)."""
    COMPARE_CACHE.mkdir(parents=True, exist_ok=True)
    path = COMPARE_CACHE / _file_name(level, text)
    if use_cache and (cached := _cached_path(level, text)):
        return LevelResult(**json.loads(cached.read_text()))

    stats, start = RunStats(), time.perf_counter()
    paused, total, problems = None, None, []
    if level == "L5":
        answer, evidence, paused, total, problems = run_l5(text, stats)
    else:
        answer, evidence = {"L1": run_l1, "L2": run_l2, "L3": run_l3, "L4": run_l4}[level](text, stats)
    result = LevelResult(level=level, answer=answer, llm_calls=stats.llm_calls, tool_calls=stats.tool_calls,
                         tokens=stats.input_tokens + stats.output_tokens, cost_usd=round(stats.cost_usd, 5),
                         seconds=round(time.perf_counter() - start, 1), evidence=evidence, paused=paused,
                         code_total=total, code_problems=problems)
    path.write_text(json.dumps(asdict(result)))
    return result


def _file_name(level: str, text: str) -> str:
    return f"{level}-{hashlib.sha256(text.strip().encode()).hexdigest()[:24]}.json"


def _cached_path(level: str, text: str):
    return next((p for p in (COMPARE_CACHE / _file_name(level, text), RECORDED / _file_name(level, text))
                 if p.exists()), None)


def is_cached(level: str, text: str) -> bool:
    return _cached_path(level, text) is not None


# ------------------------------------------------------------------ code-scored checks

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")


def dollar_amounts(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _MONEY.findall(text)]


def requested_budget(text: str) -> float | None:
    m = re.search(r"budget\D{0,12}(\d[\d,]*)\s*(k\b)?", text, re.I) or re.search(r"\$\s?(\d[\d,]*)\s*(k\b)?", text)
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)


_PART_OF_TRIP = re.compile(r"flight|hotel|sight|night|adult|person|ticket|food|meal|activit|room", re.I)


def stated_total(answer: str) -> float | None:
    """The overall trip total the answer states, or None if it never gives one.

    Only lines about the whole trip count: 'Total: $244 for 7 nights' or 'Total Flight Cost: $1,224'
    are parts of the trip, not its total (L4 once scored "within budget" from a single hotel's
    total). On a total line the LAST amount wins: '$1,246.90 + $700 + $230 = $2,176.90'."""
    total = None
    for line in answer.splitlines():
        at = line.lower().find("total")
        if at < 0:
            continue
        segment = line[at:]  # judge only what follows "total": "Hotel $931. Total cost: $2,650" is a trip total
        # ...and stop at "budget": "total of $2,080 exceeds your budget of $1,500" is $2,080, not $1,500
        segment = re.split(r"budget", segment, maxsplit=1, flags=re.I)[0]
        if not _PART_OF_TRIP.search(segment) and (amounts := dollar_amounts(segment)):
            total = amounts[-1]
    return total


_DECISION_WORDS = re.compile(r"budget|date|when|depart|from where|which city|how many|travel(l)?ers|adults|increase|raise", re.I)


def flagged_budget(answer: str) -> bool:
    """Did the answer at least notice that the trip doesn't fit the budget?"""
    return bool(re.search(r"exceed|over (your |the )?budget|beyond your budget|above your budget|not fit", answer, re.I))


def asked_for_decision(answer: str) -> bool:
    """A question to the traveler about the decision itself (budget, dates, who, from where), not a
    polite closer like 'Would you like to adjust any part of this plan?'."""
    questions = re.findall(r"[^.?!\n]*\?", answer)
    return any(_DECISION_WORDS.search(q) for q in questions)


def evidence_prices(evidence: list[dict]) -> set[float]:
    prices = set()
    for r in evidence:
        prices |= {o["price"] for o in r.get("offers", [])}
        for h in r.get("hotels", []):
            prices |= {h.get("price_per_night"), h.get("total_price")}
        prices |= {s.get("price_usd") for s in r.get("sights", [])}
    return {p for p in prices if p is not None}


@dataclass
class Checks:
    prices_traced: int
    prices_total: int
    within_budget: bool | None  # None = no total stated
    invented: int
    asked: bool
    needed: bool
    noticed: bool = False  # flagged the budget problem, even if it didn't ask

    @property
    def real_prices(self) -> bool:
        return self.prices_total > 0 and self.prices_traced == self.prices_total


def score(result: LevelResult, request_text: str, needed: bool) -> Checks:
    budget = requested_budget(request_text)
    if result.level == "L5":
        # The app's plan is built and verified by code: prices come from tools, totals from code
        has_plan = result.code_total is not None
        return Checks(prices_traced=int(has_plan), prices_total=int(has_plan),
                      within_budget=None if not has_plan or budget is None else result.code_total <= budget,
                      invented=len(result.code_problems), asked=result.paused in ("clarify", "escalation"),
                      needed=needed)
    known = evidence_prices(result.evidence)
    amounts = [a for a in dollar_amounts(result.answer) if a != budget and a > 0]
    traced = sum(1 for a in amounts if any(abs(a - p) <= 1.0 for p in known))
    total = stated_total(result.answer)
    return Checks(prices_traced=traced, prices_total=len(amounts),
                  within_budget=None if total is None or budget is None else total <= budget,
                  invented=len(amounts) - traced, asked=asked_for_decision(result.answer), needed=needed,
                  noticed=flagged_budget(result.answer))


def decision_needed(l5: LevelResult) -> bool:
    """Ground truth from the app's own code: validation or the budget check said a human must decide."""
    return l5.paused in ("clarify", "escalation")


def takeaway(level: str, c: Checks) -> str:
    if level == "L5":
        if c.needed and c.asked:
            return "Code caught the problem and asked you instead of guessing; every price is from a real search."
        return "Every price is from a real search and every total is computed by code; you review before booking."
    parts = []
    if c.prices_total == 0:
        parts.append("gave no prices")
    elif c.invented:
        parts.append(f"{c.invented} of {c.prices_total} prices can't be traced to real data")
    else:
        parts.append("all prices trace to a real search")
    if c.within_budget is False:
        parts.append("its own total is over budget")
    elif c.within_budget is None:
        parts.append("no clear total")
    if c.needed and not c.asked:
        parts.append("it noticed the budget problem but didn't ask you" if c.noticed
                     else "and it didn't ask you, though the budget can't work")
    elif c.needed and c.asked:
        parts.append("and it asked you")
    return (parts[0][0].upper() + parts[0][1:] + ("; " + ", ".join(parts[1:]) if len(parts) > 1 else "") + ".")
