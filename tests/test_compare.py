"""'Compare the levels': the code-scored checks and the cache (no API calls)."""

from fakes import ScriptedLLM
from langchain.messages import AIMessage

from travel_planner import compare
from travel_planner.compare import (
    LevelResult,
    decision_needed,
    dollar_amounts,
    asked_for_decision,
    requested_budget,
    score,
    stated_total,
    takeaway,
)
from travel_planner.trace import RunStats

REQUEST = "Plan London from SFO, March 10 to 17 2027, 2 adults, budget 1500 dollars"
EVIDENCE = [{"offers": [{"offer_id": "o1", "price": 1227.0}]},
            {"hotels": [{"hotel_id": "h1", "price_per_night": 133, "total_price": 931.0}]},
            {"sights": [{"name": "Tower of London", "price_usd": 49.03}]}]


def result(level, answer, **kw) -> LevelResult:
    return LevelResult(level=level, answer=answer, llm_calls=1, tool_calls=0, tokens=10, cost_usd=0.0,
                       seconds=0.1, **kw)


def test_parsing_helpers():
    assert requested_budget(REQUEST) == 1500
    assert requested_budget("Tokyo, budget 4k") == 4000
    assert dollar_amounts("Flight $1,227.00, hotel $133/night") == [1227.0, 133.0]
    assert stated_total("Hotel $931. **Total estimated cost:** $2,650") == 2650
    assert stated_total("hotel $931 total, Tower of London $49.03. Total: $2,207") == 2207  # regression


def test_plain_llm_prices_are_untraceable_and_over_budget():
    answer = "Flight: $850. Hotel: $180/night. Total: $2,300. Enjoy London!"
    c = score(result("L1", answer), REQUEST, needed=True)
    assert (c.prices_traced, c.prices_total, c.invented) == (0, 3, 3)
    assert c.within_budget is False and not c.asked
    assert takeaway("L1", c).startswith("3 of 3 prices can't be traced")
    assert "didn't ask you" in takeaway("L1", c)


def test_tool_prices_are_traced_but_llm_arithmetic_is_not():
    answer = "Flight $1,227.00, hotel $931 total, Tower of London $49.03. Total: $2,207"  # the sum is the LLM's
    c = score(result("L4", answer, evidence=EVIDENCE), REQUEST, needed=True)
    assert (c.prices_traced, c.prices_total) == (3, 4)
    assert c.within_budget is False


def test_the_app_is_scored_from_code_not_text():
    l5 = result("L5", "I couldn't fit this trip…", paused="escalation", code_total=3328.8, code_problems=[])
    needed = decision_needed(l5)
    c = score(l5, REQUEST, needed)
    assert needed and c.asked and c.real_prices and c.invented == 0
    assert c.within_budget is False  # over budget, and it asked instead of pretending
    assert takeaway("L5", c).startswith("Code caught the problem")


def test_results_are_cached_per_request(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "COMPARE_CACHE", tmp_path)
    calls, real_run_l1 = [], compare.run_l1

    def fake_l1(text, stats: RunStats, llm=None):
        calls.append(text)
        return real_run_l1(text, stats, ScriptedLLM(messages=iter([AIMessage("Plan: $900 total")])))

    monkeypatch.setitem(compare.__dict__, "run_l1", fake_l1)
    first = compare.run_level("L1", REQUEST)
    second = compare.run_level("L1", REQUEST)
    assert first == second and len(calls) == 1  # the rerun came from the cache
    assert compare.is_cached("L1", REQUEST)



def test_part_totals_are_not_the_trip_total():
    """Regression: L4 listed 'Total: $244 for 7 nights' (one hotel) and was scored within budget."""
    answer = "- Price: $35 per night (Total: $244 for 7 nights)\n- **Total Flight Cost**: $1224.50"
    assert stated_total(answer) is None
    assert stated_total("**Total Trip Cost:** $1246.90 + $700 + $230 = **$2176.90**") == 2176.90


def test_only_decision_questions_count_as_asking():
    """Regression: 'Would you like to adjust any part of this plan?' was scored as asking."""
    assert not asked_for_decision("Here is your plan. Would you like to adjust any part of this plan?")
    assert asked_for_decision("This exceeds your budget. Would you like to increase your budget to $3,400?")
    assert asked_for_decision("What dates are you travelling?")



def test_the_budget_is_not_mistaken_for_the_total():
    """Regression: 'total of $2080 exceeds your budget of $1500' was read as a $1,500 total."""
    answer = "Unfortunately, the total estimated cost of $2080 exceeds your budget of $1500."
    assert stated_total(answer) == 2080
    c = score(result("L2", answer), REQUEST, needed=True)
    assert c.within_budget is False and c.noticed and not c.asked
    assert "noticed the budget problem but didn't ask you" in takeaway("L2", c)
