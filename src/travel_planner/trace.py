"""Pretty-print a streamed agent run so you can read it as laps around the loop, and drive
runs that pause on interrupts."""

import json
import sys

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command

from travel_planner.bookings import parse_decision
from travel_planner.guardrails import unfence
from travel_planner.limits import RunLimits
from travel_planner.usage import record

# gpt-4o-mini list prices per 1M tokens (USD). Update if OpenAI changes pricing.
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60


class RunStats(UsageMetadataCallbackHandler):
    """Counts every LLM call and its tokens via a LangChain callback.

    A callback sees all model calls, including ones that never become a message in state
    (e.g. structured-output extraction). Pass it in config: {"callbacks": [stats]}.
    """

    def __init__(self, limits: RunLimits | None = None):
        super().__init__()
        self.llm_calls = 0
        self.tool_calls = 0
        self.limits = limits
        self.raise_error = limits is not None  # let RunLimitExceeded stop the run

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        super().on_llm_end(response, **kwargs)
        # Persist to the usage ledger so spend is tracked across runs, not just this one
        try:
            message = response.generations[0][0].message
            usage = message.usage_metadata or {}
            model = message.response_metadata.get("model_name", "unknown")
        except (IndexError, AttributeError):
            return
        if usage:
            tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            record("openai", model=model, input_tokens=tokens_in, output_tokens=tokens_out,
                   cost_usd=round((tokens_in * PRICE_PER_M_INPUT + tokens_out * PRICE_PER_M_OUTPUT) / 1_000_000, 6))
        if self.limits:
            self.limits.check(self.input_tokens + self.output_tokens, self.cost_usd)

    def on_tool_end(self, output, **kwargs):
        self.tool_calls += 1  # sees tools inside nested agents too, unlike counting ToolMessages

    @property
    def input_tokens(self) -> int:
        return sum(u.get("input_tokens", 0) for u in self.usage_metadata.values())

    @property
    def output_tokens(self) -> int:
        return sum(u.get("output_tokens", 0) for u in self.usage_metadata.values())

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_PER_M_INPUT + self.output_tokens * PRICE_PER_M_OUTPUT) / 1_000_000

    def summary(self) -> str:
        return (
            f"LLM calls: {self.llm_calls} | tool calls: {self.tool_calls} | "
            f"tokens in/out: {self.input_tokens:,}/{self.output_tokens:,} | est. cost: ${self.cost_usd:.4f}"
        )


def _short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text):,} chars]"


def print_update(node_name: str, update: dict, stats: RunStats) -> None:
    print(f"\n{'─' * 80}\n📍 {node_name}\n{'─' * 80}")
    update = update or {}
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"🤖 requests {len(msg.tool_calls)} tool call(s):")
                for call in msg.tool_calls:
                    print(f"   • {call['name']}({json.dumps(call['args'])})")
            if msg.content:
                print(f"💬 {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"🙋 {msg.content}")
        elif isinstance(msg, ToolMessage):
            if getattr(msg, "status", None) == "error" and not msg.content.startswith("{"):
                icon = "🛑"  # e.g. a write the human rejected
            elif msg.name == "ask_traveler":
                icon = "🙋"  # the traveler's answer
            elif not unfence(msg.content).startswith("{"):
                icon = "📦"  # e.g. create_agent's structured-response confirmation
            else:
                icon = "✅" if json.loads(unfence(msg.content)).get("status") == "ok" else "⚠️"
            print(f"{icon} {msg.name} -> {_short(msg.content)}")
    # Non-message state fields (e.g. draft, problems, request): show what the node wrote
    for key, value in update.items():
        if key != "messages":
            print(f"📝 {key} = {_short(json.dumps(value, default=str), 400)}")


def _label(namespace: tuple, node_name: str) -> str:
    """('intake:<task id>',) + 'extract' -> 'intake › extract'."""
    return " › ".join([part.split(":")[0] for part in namespace] + [node_name])


def run_with_interrupts(agent, agent_input, config: dict, answers: list[str], stats: RunStats,
                        subgraphs: bool = False) -> None:
    """Stream the graph; whenever it pauses on an interrupt, get an answer and resume.

    Answers come from `answers` (scripted) first, then from the keyboard if interactive.
    The resume value is always {"answer": ...}, the contract every interrupt in this project uses.
    """
    config = {**config, "callbacks": [stats]}
    while True:
        paused_with = None
        for item in agent.stream(agent_input, config=config, stream_mode="updates", subgraphs=subgraphs):
            namespace, chunk = item if subgraphs else ((), item)
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    paused_with = update[0].value
                else:
                    print_update(_label(namespace, node_name), update, stats)

        if paused_with is None:
            return  # reached END

        print(f"\n⏸️  PAUSED. Next node on resume: {agent.get_state(config).next}")
        agent_input = resume_command(paused_with, answers)
        if agent_input is None:
            return


def pending_interrupt(agent, config: dict) -> dict | None:
    """The value of a saved, unanswered interrupt on this thread (e.g. from a previous run), if any."""
    for task in agent.get_state(config).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def resume_command(paused_with: dict, answers: list[str]) -> Command | None:
    """Show a pause and collect the human's reply. None = no human available (stay paused)."""
    if "action_requests" in paused_with:
        # HumanInTheLoopMiddleware: one decision per paused tool call
        decisions = []
        for action in paused_with["action_requests"]:
            print(f"🛂 {action['description']}")
            decision = _get_decision(action, answers)
            if decision is None:
                return None
            decisions.append(decision)
        return Command(resume={"decisions": decisions})
    print(f"❓ {paused_with['question']}")
    answer = _get_answer(answers)
    return None if answer is None else Command(resume={"answer": answer})


def _get_answer(answers: list[str]) -> str | None:
    """Scripted answer first, then keyboard. None = no human available."""
    if answers:
        answer = answers.pop(0)
        print(f"🙋 (scripted) {answer}")
        return answer
    if sys.stdin.isatty():
        return input("🙋 Your answer: ")
    # Never invent an answer: leave the graph paused (its state is checkpointed) and stop.
    print("⏹️  No answer available: leaving the run paused.")
    return None


def _get_decision(action: dict, answers: list[str]) -> dict | None:
    """Ask until the reply is a valid approve / edit / reject decision."""
    while True:
        answer = _get_answer(answers)
        if answer is None:
            return None
        try:
            return parse_decision(answer, action)
        except ValueError as e:
            print(f"⚠️  {e}")
