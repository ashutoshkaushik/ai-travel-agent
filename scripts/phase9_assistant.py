"""Phase 9: the whole travel assistant behind one front door. Conversations persist in SQLite.

Interactive chat (type messages; answer its questions when it pauses):
    uv run python scripts/phase9_assistant.py --thread my-trip

Scripted (each --say is a new message; --answer replies to pauses, in order):
    uv run python scripts/phase9_assistant.py --thread demo \
        --say "Plan Rome from ORD, Nov 20-27, 2 people, 3000 dollars" --answer approve
    uv run python scripts/phase9_assistant.py --thread demo --say "book it" --answer approve --answer approve
    uv run python scripts/phase9_assistant.py --thread demo --say "do I need a visa?"

Paused threads resume where they left off: run again with the same --thread and an --answer.

Other:
    --graph      print the router graph (Mermaid)
    --profile    show the saved traveler profile (long-term memory)
"""

import argparse
import sys

from langchain.messages import HumanMessage

from travel_planner.agents.assistant import build_assistant, profile_namespace
from travel_planner.persistence import open_checkpointer, open_store
from travel_planner.trace import RunStats, pending_interrupt, resume_command, run_with_interrupts


def print_reply(assistant, config: dict) -> None:
    """The assistant's latest reply, unless it's still waiting on a question (shown above)."""
    if not pending_interrupt(assistant, config):
        print(f"\n🤖 {assistant.get_state(config).values['messages'][-1].content}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="default-trip")
    parser.add_argument("--user", default="me")
    parser.add_argument("--say", action="append", default=[], help="a message to send (repeatable)")
    parser.add_argument("--answer", action="append", default=[], help="reply to a pause (repeatable)")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    store = open_store()
    config = {"configurable": {"thread_id": args.thread, "user_id": args.user}}
    if args.profile:
        item = store.get(profile_namespace(config), "travel")
        print(f"🧠 Profile for {args.user!r}: {item.value if item else 'nothing saved yet'}")
        return

    assistant = build_assistant(open_checkpointer(), store)
    if args.graph:
        print(assistant.get_graph().draw_mermaid())
        return

    stats = RunStats()
    pending = pending_interrupt(assistant, config)
    if pending:
        # A pause saved by an earlier run: answer it first
        print("⏯️  This conversation is paused, waiting for your reply:")
        command = resume_command(pending, args.answer)
        if command is not None:
            run_with_interrupts(assistant, command, config, args.answer, stats, subgraphs=True)
            print_reply(assistant, config)

    messages = list(args.say)
    while True:
        if messages:
            text = messages.pop(0)
        elif sys.stdin.isatty() and not args.say:
            text = input("\n💭 You (blank to quit): ").strip()
        else:
            break
        if not text:
            break
        print(f"\n🧳 {text}")
        run_with_interrupts(assistant, {"messages": [HumanMessage(text)]}, config, args.answer, stats, subgraphs=True)
        print_reply(assistant, config)

    state = assistant.get_state(config).values
    print(f"\n{'=' * 80}\n🧵 Thread {args.thread!r}: trip_status={state.get('trip_status')}, "
          f"{len(state.get('messages', []))} messages saved | pending: {assistant.get_state(config).next or 'nothing'}")
    print(f"📊 {stats.summary()}")


if __name__ == "__main__":
    main()
