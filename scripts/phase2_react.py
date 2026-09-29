"""Phase 2: run the hand-wired ReAct agent and watch each lap of the loop.

    uv run python scripts/phase2_react.py
    uv run python scripts/phase2_react.py "What's the weather like in Kyoto next week?"
    uv run python scripts/phase2_react.py --graph      # print the graph as Mermaid
"""

import sys

from langchain.messages import HumanMessage

from travel_planner.agents.react_agent import build_react_agent
from travel_planner.tools import ALL_TOOLS
from travel_planner.trace import RunStats, print_update

agent = build_react_agent(ALL_TOOLS)  # no checkpointer: every run starts from nothing

DEFAULT_REQUEST = (
    "I'm planning a trip from San Francisco to Tokyo, December 6 to 12, 2026, for 2 adults. "
    "Find round-trip flights, 3 well-rated hotels under $150 a night, and the top 5 sights."
)

if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()

    request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUEST
    print(f"🧳 {request}")

    stats = RunStats()
    # recursion_limit caps total node executions: a safety net against a runaway loop burning credits
    for chunk in agent.stream(
        {"messages": [HumanMessage(request)]},
        config={"recursion_limit": 15, "callbacks": [stats]},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            print_update(node_name, update, stats)

    print(f"\n{'=' * 80}\n📊 {stats.summary()}")
