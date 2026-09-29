"""Run the evals: python -m evals.run [--category planner] [--case id] [--no-judge] [--offline]

Deterministic and nearly free:
- API responses come from evals/cache (TRAVEL_FROZEN_CACHE=1: any age is fine). A miss is recorded
  once, reusing the app's own cache first (TRAVEL_CACHE_SEED_DIR) to save SerpApi quota.
- LLM responses come from a disk cache keyed by the exact prompt; only misses cost money.
- Today's date is pinned, so prompts that mention it never change.
--offline (CI) never calls a paid API: a missing recording fails the case with a clear message.
Every run appends a summary to evals/results/history.jsonl (pass rate, cost, cache hits) for the trend chart.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINNED_TODAY = "2026-09-27"
os.environ.setdefault("TRAVEL_CACHE_DIR", str(ROOT / "cache"))
os.environ.setdefault("TRAVEL_CACHE_SEED_DIR", str(ROOT.parent / "cache"))
os.environ["TRAVEL_FROZEN_CACHE"] = "1"
os.environ["TRAVEL_TODAY"] = PINNED_TODAY
os.environ.setdefault("TRAVEL_USAGE_LOG", str(ROOT / "results" / "usage.jsonl"))  # keep the app's ledger clean

from langchain_core.globals import set_llm_cache  # noqa: E402

from evals.runner import build_graph, run_case  # noqa: E402
from evals.scorers import score  # noqa: E402
from travel_planner.llm_cache import DiskLLMCache  # noqa: E402

DATASET = ROOT / "dataset.jsonl"
RESULTS = ROOT / "results"
LEVELS = ("result", "steps", "quality")


def load_cases(category: str | None = None, case_id: str | None = None) -> list[dict]:
    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    return [c for c in cases if (not category or c["category"] == category) and (not case_id or c["id"] == case_id)]


def evaluate(cases: list[dict], use_judge: bool, offline: bool, progress=print) -> dict:
    if offline:
        os.environ.setdefault("OPENAI_API_KEY", "offline-replay")  # never used: every call is a cache hit or an error
        os.environ.setdefault("DUFFEL_ACCESS_TOKEN", "duffel_test_offline")
        os.environ.setdefault("SERPAPI_API_KEY", "offline")
    cache = DiskLLMCache(Path(os.environ["TRAVEL_CACHE_DIR"]) / "llm", offline=offline)
    set_llm_cache(cache)
    graph = build_graph()
    rows = []
    for i, case in enumerate(cases, 1):
        record = run_case(graph, case)
        checks = score(record, case)
        quality = None
        if use_judge and case.get("judge") and not record["error"]:
            from evals.judge import judge
            try:
                quality = judge(record)
            except Exception as e:
                quality = {"passed": False, "reasoning": f"judge failed: {type(e).__name__}: {e}"[:200]}
        level_pass = {
            "result": all(ok for _, ok, _ in checks["result"]),
            "steps": all(ok for _, ok, _ in checks["steps"]),
            "quality": None if quality is None else bool(quality["passed"]),
        }
        passed = all(v for v in level_pass.values() if v is not None)
        rows.append({"id": case["id"], "category": case["category"], "input": " / ".join(case["turns"]),
                     "passed": passed, "levels": level_pass, "checks": checks, "quality": quality,
                     "path": record["path"], "steps": record["steps"], "intent": record["intent"], "tool_calls": len(record["tool_calls"]),
                     "llm_calls": record["llm_calls"], "seconds": record["seconds"], "error": record["error"],
                     "reply": (record["replies"][-1] if record["replies"] else "")[:400]})
        failed = [name for level in ("result", "steps") for name, ok, _ in checks[level] if not ok]
        progress(f"[{i}/{len(cases)}] {'PASS' if passed else 'FAIL'} {case['category']:<10} {case['id']}"
                 + (f"  ✗ {', '.join(failed)}" if failed else "")
                 + ("" if quality is None else f"  judge {quality.get('average', '?')}"))
    set_llm_cache(None)
    return summarize(rows, cache, use_judge, offline)


def summarize(rows: list[dict], cache: DiskLLMCache, use_judge: bool, offline: bool) -> dict:
    by_category = {}
    for row in rows:
        c = by_category.setdefault(row["category"], {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += row["passed"]
    level_rates = {}
    for level in LEVELS:
        scored = [r["levels"][level] for r in rows if r["levels"][level] is not None]
        level_rates[level] = round(sum(scored) / len(scored), 3) if scored else None
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(rows), "passed": sum(r["passed"] for r in rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3) if rows else 0.0,
        "by_category": by_category, "level_rates": level_rates,
        "judge": use_judge, "offline": offline, **cache.stats(), "rows": rows,
    }


def save(summary: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(summary, indent=1))
    with (RESULTS / "history.jsonl").open("a") as f:
        f.write(json.dumps({k: v for k, v in summary.items() if k != "rows"}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the travel agent evals.")
    parser.add_argument("--category", help="Only this category (planner, budget, intake, guide, booking, guardrails, router)")
    parser.add_argument("--case", help="Only this case id")
    parser.add_argument("--no-judge", action="store_true", help="Code checks only (levels 1 and 2)")
    parser.add_argument("--offline", action="store_true", help="Recorded responses only; never call a paid API (CI)")
    parser.add_argument("--no-save", action="store_true", help="Don't write results/latest.json or history")
    args = parser.parse_args()

    cases = load_cases(args.category, args.case)
    if not cases:
        print("No cases match.")
        return 2
    summary = evaluate(cases, use_judge=not args.no_judge, offline=args.offline)
    if not args.no_save:
        save(summary)
    print(f"\n{summary['passed']}/{summary['cases']} passed ({summary['pass_rate']:.0%}) · "
          f"levels {summary['level_rates']} · new LLM spend ${summary['new_cost_usd']:.4f} · "
          f"LLM cache {summary['llm_cache_hits']} hits / {summary['llm_cache_misses']} misses")
    for category, c in summary["by_category"].items():
        print(f"  {category:<11} {c['passed']}/{c['total']}")
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
