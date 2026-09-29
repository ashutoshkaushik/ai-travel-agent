"""Turn human rejections (feedback.jsonl) into eval cases: every "no" becomes a regression test.

    uv run python -m evals.from_feedback            # write candidates to evals/feedback_cases.jsonl
    uv run python -m evals.from_feedback --append   # also add new ones to evals/dataset.jsonl

A rejection at the booking gate says "the agent proposed X and the traveler didn't want it, because Y".
The case replays the same trip with the reason as a stated preference, and asserts the plan avoids X.
Rejections logged before the trip request was attached can't be replayed and are skipped.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEEDBACK = ROOT.parent / "feedback.jsonl"
CANDIDATES = ROOT / "feedback_cases.jsonl"
DATASET = ROOT / "dataset.jsonl"


def case_from(event: dict) -> dict | None:
    request = event.get("request")
    if not request or event.get("decision") != "reject":
        return None
    args = event["args"]
    rejected = args.get("hotel_name") or args.get("hotel_id") if event["action"] == "book_hotel" else args.get("offer_id")
    if not rejected:
        return None
    text = (f"Plan {request['city']} from {request['origin']}, {request['depart_date']} to {request['return_date']}, "
            f"{request['adults']} adults, budget {request['budget_usd']} dollars. {event['reason'].strip().rstrip('.')}.")
    digest = hashlib.sha1(f"{event['action']}|{rejected}|{event['reason']}".encode()).hexdigest()[:6]
    return {"id": f"feedback_{request['city'].lower().replace(' ', '_')}_{digest}", "category": "feedback",
            "turns": [text], "answers": [], "expected_intent": "plan_trip",
            "expected_path": ["classify_intent", "intake", "planner", "research", "budget_check"],
            "assertions": [f"plan_avoids:{rejected}", "budget_honest", "prices_match_evidence"],
            "max_tool_calls": 12, "judge": True, "source": {"ts": event["ts"], "reason": event["reason"]}}


def build(feedback: Path = FEEDBACK) -> tuple[list[dict], int]:
    events = [json.loads(line) for line in feedback.read_text().splitlines() if line.strip()] if feedback.exists() else []
    cases, seen = [], set()
    for event in events:
        case = case_from(event)
        if case and case["id"] not in seen:
            seen.add(case["id"])
            cases.append(case)
    return cases, len(events) - len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--append", action="store_true", help="Add new cases to evals/dataset.jsonl")
    args = parser.parse_args()
    cases, skipped = build()
    CANDIDATES.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"{len(cases)} case(s) from feedback ({skipped} rejection(s) skipped: no trip request attached) "
          f"→ {CANDIDATES.relative_to(ROOT.parent)}")
    if args.append:
        existing = {json.loads(line)["id"] for line in DATASET.read_text().splitlines() if line.strip()}
        new = [c for c in cases if c["id"] not in existing]
        with DATASET.open("a") as f:
            f.writelines(json.dumps(c) + "\n" for c in new)
        print(f"Appended {len(new)} new case(s) to {DATASET.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
