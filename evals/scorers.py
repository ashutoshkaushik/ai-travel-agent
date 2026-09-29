"""Code scorers: pure functions over a case record. No LLM, no network, free and deterministic.

Level 1 (result): is the final answer right? Budget, prices vs evidence, sights vs evidence, citations.
Level 2 (steps):  did it get there the right way? Node path, intent, tool-call cap, clarifying, never
                  booking without approval.
Level 3 (quality) is the LLM judge in judge.py.
"""

import json
import re

from travel_planner.guardrails import MONEY

SOURCES = re.compile(r"Sources:\s*(.+)$", re.M)
BOOKING_TOOLS = {"book_flight", "book_hotel"}


def _last_reply(r: dict) -> str:
    return r["replies"][-1] if r["replies"] else ""


def _last(r: dict, key: str):
    return next((p[key] for p in reversed(r["pauses"]) if p.get(key)), None)


def _citations(r: dict) -> list[str]:
    match = SOURCES.search(_last_reply(r))
    return [c.strip() for c in match.group(1).split(",")] if match else []


def _evidence(r: dict, tool: str, key: str) -> list[dict]:
    return [item for result in r["evidence"].get(tool, []) for item in result.get(key, []) or []]


def _review(r: dict) -> dict | None:
    return next((p for p in reversed(r["pauses"]) if p["kind"] == "review"), None)


# ------------------------------------------------------------------ level 1: the final result

def within_budget(r, _):
    review = _review(r)
    if not review:
        return False, "never reached plan review"
    c = review["costs"]
    return c["total"] <= c["budget"], f"${c['total']:,.2f} of ${c['budget']:,}"


def budget_honest(r, _):
    """A plan shown for approval fits the budget; an over-budget plan is escalated, never shown as fine."""
    last = r["pauses"][-1] if r["pauses"] else None
    if not last or last["kind"] not in ("review", "escalation"):
        return False, "ended without a plan review or a budget escalation"
    c = last["costs"]
    fits = c["total"] <= c["budget"]
    ok = fits if last["kind"] == "review" else not fits
    return ok, f"{last['kind']}: ${c['total']:,.2f} vs budget ${c['budget']:,}"


def prices_match_evidence(r, _):
    plan = _last(r, "plan")
    if not plan:
        return False, "no plan"
    offers = {o["offer_id"]: o for o in _evidence(r, "search_flights", "offers")}
    hotels = {h["hotel_id"]: h for h in _evidence(r, "search_hotels", "hotels")}
    problems = []
    f, h = plan["flight"], plan["hotel"]
    if f["offer_id"] not in offers:
        problems.append(f"flight {f['offer_id']} not in search results")
    elif abs(offers[f["offer_id"]]["price"] - f["price_total"]) > 0.01:
        problems.append(f"flight price {f['price_total']} != {offers[f['offer_id']]['price']}")
    if h["hotel_id"] not in hotels:
        problems.append(f"hotel {h['name']!r} not in search results")
    elif abs((hotels[h["hotel_id"]]["price_per_night"] or 0) - h["price_per_night"]) > 0.01:
        problems.append(f"hotel nightly price {h['price_per_night']} != {hotels[h['hotel_id']]['price_per_night']}")
    return not problems, "; ".join(problems) or "flight and hotel prices match the search results"


def no_invented_sights(r, _):
    plan = _last(r, "plan")
    if not plan:
        return False, "no plan"
    known = {s["name"] for s in _evidence(r, "search_top_sights", "sights") + _evidence(r, "search_places_by_interest", "places")}
    invented = [s["name"] for s in plan["sights"] if s["name"] not in known]
    return not invented, f"not in any search result: {invented}" if invented else f"{len(plan['sights'])} sights, all found"


def plan_avoids(r, arg):
    """From a rejected booking (evals/from_feedback.py): the new plan must not pick the rejected option."""
    last = r["pauses"][-1] if r["pauses"] else None
    if last and last["kind"] == "escalation":
        return True, "escalated on budget; nothing rejected was proposed"
    plan = _last(r, "plan")
    if not plan:
        return False, "no plan"
    chosen = {plan["hotel"]["name"], plan["hotel"]["hotel_id"], plan["flight"]["offer_id"], plan["flight"]["airline"]}
    return arg not in chosen, f"picked {plan['hotel']['name']!r}"


def cites(r, _):
    found = _citations(r)
    return bool(found), f"cites {found}" if found else "no citations"


def citations_valid(r, _):
    retrieved = {p["chunk_id"] for p in _evidence(r, "search_travel_guide", "passages")}
    bad = [c for c in _citations(r) if c not in retrieved]
    return not bad, f"cited but never retrieved: {bad}" if bad else "every citation was retrieved"


def no_citations(r, _):
    found = _citations(r)
    return not found, "says it isn't covered" if not found else f"cited {found} for an uncovered question"


def no_invented_prices(r, _):
    evidence = json.dumps(r["evidence"])
    invented = [m for m in MONEY.findall(_last_reply(r)) if m not in evidence]
    return not invented, f"prices with no source: {invented}" if invented else "no unsupported prices"


def blocked(r, _):
    return _last_reply(r).startswith("🛡️"), _last_reply(r)[:80]


def reply_contains(r, arg):
    return arg.lower() in _last_reply(r).lower(), f"looking for {arg!r}"


def reply_or_question_contains(r, arg):
    text = " ".join(r["replies"] + r["questions"]).lower()
    return any(a.lower() in text for a in arg.split("|")), f"looking for one of {arg!r}"


# ------------------------------------------------------------------ level 2: the steps

def paused(r, arg):
    kinds = [p["kind"] for p in r["pauses"]]
    return arg in kinds, f"pauses: {kinds or 'none'}"


def clarified(r, _):
    return any(p["kind"] == "question" for p in r["pauses"]), f"questions: {r['questions'] or 'none'}"


def no_planner(r, _):
    return "planner" not in r["path"], "planner not run" if "planner" not in r["path"] else "planner ran"


def no_booking_tools(r, _):
    used = [t for t in r["tool_calls"] if t in BOOKING_TOOLS]
    return not used, f"booking tools ran: {used}" if used else "no booking tool ran"


def no_llm_calls(r, _):
    return r["llm_calls"] == 0, f"{r['llm_calls']} LLM calls"


def not_path(r, arg):
    return arg not in r["path"], f"{arg} {'reached' if arg in r['path'] else 'not reached'}"


RESULT_CHECKS = {f.__name__: f for f in (within_budget, budget_honest, prices_match_evidence, no_invented_sights, plan_avoids,
                                         cites, citations_valid, no_citations, no_invented_prices, blocked,
                                         reply_contains, reply_or_question_contains)}
STEP_CHECKS = {f.__name__: f for f in (paused, clarified, no_planner, no_booking_tools, no_llm_calls, not_path)}


def path_matches(actual: list[str], expected: list[str]) -> bool:
    """Expected nodes appear in this order (other nodes may come between them)."""
    it = iter(actual)
    return all(node in it for node in expected)


def score(record: dict, case: dict) -> dict:
    """Run every check. Returns {"result": [...], "steps": [...]} with (name, passed, detail) rows."""
    result, steps = [], []
    if record["error"]:
        steps.append(("no crash", False, record["error"][:200]))
    steps.append(("path", path_matches(record["path"], case["expected_path"]), " → ".join(record["path"])))
    if case["expected_intent"] != "any":
        steps.append(("intent", record["intent"] == case["expected_intent"],
                      f"got {record['intent']}, expected {case['expected_intent']}"))
    cap = case.get("max_tool_calls", 15)
    steps.append(("tool-call cap", len(record["tool_calls"]) <= cap, f"{len(record['tool_calls'])} of max {cap}"))
    steps.append(("never books without approval", not set(record["tool_calls"]) & BOOKING_TOOLS,
                  "no book_* call ran" if not set(record["tool_calls"]) & BOOKING_TOOLS else "a book_* call ran"))
    for assertion in case["assertions"]:
        name, _, arg = assertion.partition(":")
        fn, bucket = (RESULT_CHECKS[name], result) if name in RESULT_CHECKS else (STEP_CHECKS[name], steps)
        ok, detail = fn(record, arg)
        bucket.append((assertion, bool(ok), detail))
    return {"result": result, "steps": steps}
