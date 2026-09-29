"""Phase 3: the agent pauses to ask the traveler, then resumes the SAME run.

Interactive (you type the answers):
    uv run python scripts/phase3_interrupt.py "Plan me a trip to Japan"

Scripted (answers supplied up front, handy for repeatable demos):
    uv run python scripts/phase3_interrupt.py "Plan me a trip to Japan" \
        --answer "From SFO, Dec 6-12 2026, 2 adults, hotels under 150/night" \
        --followup "Now find the cheapest nonstop option instead"

After the agent answers, you can keep chatting on the same thread: the checkpointer remembers.
"""

import argparse
import sys

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from travel_planner.agents.react_agent import build_react_agent
from travel_planner.tools import ALL_TOOLS
from travel_planner.tools.human import ask_traveler
from travel_planner.trace import RunStats, run_with_interrupts

checkpointer = InMemorySaver()  # snapshots live in RAM: gone when this process exits (Phase 10 fixes that)
agent = build_react_agent(ALL_TOOLS + [ask_traveler], checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--answer", action="append", default=[], help="scripted answer to an interrupt (repeatable)")
    parser.add_argument("--followup", action="append", default=[], help="scripted follow-up message (repeatable)")
    parser.add_argument("--thread", default="trip-1")
    args = parser.parse_args()

    config = {"configurable": {"thread_id": args.thread}, "recursion_limit": 25}
    stats = RunStats()
    message = args.request

    while message:
        print(f"\n🧳 {message}")
        run_with_interrupts(agent, {"messages": [HumanMessage(message)]}, config, args.answer, stats)

        if args.followup:
            message = args.followup.pop(0)
        elif sys.stdin.isatty():
            message = input("\n💭 Follow-up on the same trip (blank to quit): ").strip()
        else:
            message = None

    snapshot = agent.get_state(config)
    history = list(agent.get_state_history(config))
    print(f"\n{'=' * 80}")
    print(f"🧠 Thread {args.thread!r}: {len(snapshot.values['messages'])} messages in state, {len(history)} checkpoints saved")
    print(f"📊 {stats.summary()}")


if __name__ == "__main__":
    main()
