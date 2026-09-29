"""How much have we used? Local ledger (every API + LLM call) plus live SerpApi quota.

    uv run python scripts/usage_report.py
"""

import os
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from travel_planner.config import require
from travel_planner.usage import USAGE_LOG, read_events

# What each provider allows (as used in this project). Only SerpApi exposes a live quota API.
LIMITS = {
    "duffel": "test mode: free, no quota",
    "serpapi": "free plan: 250 searches/month (live numbers below)",
    "opentripmap": "free tier with a daily cap (see your OpenTripMap dashboard)",
    "nominatim": "free, max 1 request/second",
    "open-meteo": "free for non-commercial use",
}


def serpapi_account() -> dict | None:
    """SerpApi's Account API does not count toward the search quota."""
    try:
        r = httpx.get("https://serpapi.com/account.json", params={"api_key": require("SERPAPI_API_KEY")}, timeout=20)
        return r.json() if r.is_success else None
    except (httpx.HTTPError, RuntimeError):
        return None


def main() -> None:
    events = read_events()
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    today = now.strftime("%Y-%m-%d")

    calls = defaultdict(lambda: {"network_today": 0, "network_month": 0, "network_all": 0, "cache_hits": 0})
    llm = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "cost_month": 0.0})
    for e in events:
        if e["provider"] == "openai":
            m = llm[e["model"]]
            m["calls"] += 1
            m["tokens_in"] += e["input_tokens"]
            m["tokens_out"] += e["output_tokens"]
            m["cost"] += e["cost_usd"]
            if e["ts"].startswith(month):
                m["cost_month"] += e["cost_usd"]
            continue
        c = calls[e["provider"]]
        if e.get("cached"):
            c["cache_hits"] += 1
        else:
            c["network_all"] += 1
            c["network_month"] += e["ts"].startswith(month)
            c["network_today"] += e["ts"].startswith(today)

    first = events[0]["ts"][:10] if events else "n/a"
    print(f"📒 Usage ledger: {USAGE_LOG.name} ({len(events)} events since {first})\n")

    print(f"{'API':<13}{'today':>7}{'month':>7}{'all':>7}{'cache hits':>12}   limits")
    for provider in LIMITS:
        c = calls[provider]
        print(f"{provider:<13}{c['network_today']:>7}{c['network_month']:>7}{c['network_all']:>7}{c['cache_hits']:>12}   {LIMITS[provider]}")

    account = serpapi_account()
    if account:
        print(f"\n🔎 SerpApi (live): {account['this_month_usage']} used this month, "
              f"{account['plan_searches_left']} of {account['searches_per_month']} left "
              f"({account['last_hour_searches']} in the last hour)")

    print(f"\n🤖 OpenAI (estimated from token counts at list price)")
    total = 0.0
    for model, m in llm.items():
        total += m["cost"]
        print(f"  {model}: {m['calls']} calls, {m['tokens_in']:,} in / {m['tokens_out']:,} out tokens, "
              f"${m['cost']:.4f} total (${m['cost_month']:.4f} this month)")
    if not llm:
        print("  no LLM calls logged yet")

    credit = os.environ.get("OPENAI_CREDIT_USD")
    if credit:
        print(f"  💳 Credit remaining ≈ ${float(credit) - total:.2f} (${float(credit):.2f} when logging started − ${total:.4f})")
    print("  Source of truth for billing: https://platform.openai.com/usage")


if __name__ == "__main__":
    main()
