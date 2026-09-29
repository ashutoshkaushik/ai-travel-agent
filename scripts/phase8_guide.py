"""Phase 8: ask the travel-guide agent. Answers only from the guides, with citations.

    uv run python scripts/phase8_guide.py                                  # demo questions
    uv run python scripts/phase8_guide.py "Do I need a visa?" --city Tokyo
    uv run python scripts/phase8_guide.py --calibrate                      # retrieval scores only, no LLM
"""

import argparse

from travel_planner.agents.guide_agent import build_guide_agent, final_answer_text, verify_citations
from travel_planner.rag import CONFIRM_BELOW, STOP_BELOW, confidence_band, default_index
from travel_planner.trace import RunStats

DEMO = [
    ("Do I need a visa?", "Tokyo"),
    ("How do I get from the airport into the city?", "Rome"),
    ("Where should we stay?", "Barcelona"),
    ("Can I chew gum there?", "Singapore"),
    ("How much is a Shinkansen ticket to Osaka?", "Tokyo"),
    ("Do I need any vaccinations?", "Singapore"),
]

CALIBRATION = DEMO + [
    ("should I tip at restaurants", "New York"),
    ("dress code for churches", "Rome"),
    ("is the Louvre open on Tuesday", "Paris"),
    ("best vegan ramen shop", "Tokyo"),
    ("what is the population", "London"),
    ("recipe for carbonara", "Rome"),
]


def calibrate() -> None:
    index = default_index()
    print(f"bands: stop < {STOP_BELOW} <= confirm < {CONFIRM_BELOW} <= proceed\n")
    for question, city in CALIBRATION:
        doc, score = index.search(question, city)[0]
        print(f"{score:.3f}  {confidence_band(score):8} {city:10} {question!r:48} -> {doc.id}")


def ask(agent, question: str, city: str, stats: RunStats) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": f"[City: {city}] {question}"}]},
                          {"callbacks": [stats], "recursion_limit": 10})
    answer = result["structured_response"]
    problems = verify_citations(answer, result["messages"])
    icon = "📖" if answer.answered_from_guides else "🙅"
    print(f"\n❓ {city}: {question}\n{icon} {final_answer_text(answer)}")
    print(f"   citations: {answer.citations or 'none'}" + ("" if not problems else f"  ⚠️ {' '.join(problems)}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--city", default="Tokyo")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
        return

    agent, stats = build_guide_agent(), RunStats()
    for question, city in ([(args.question, args.city)] if args.question else DEMO):
        ask(agent, question, city, stats)
    print(f"\n📊 {stats.summary()}")


if __name__ == "__main__":
    main()
