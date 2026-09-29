"""Spending limits, enforced in code.

- RunLimits: a token and dollar cap for ONE run (one message and everything it triggers), on top of
  LangGraph's recursion_limit. Checked by RunStats (trace.py) after every LLM call.
- SessionBudget: a cap per browser session (planner runs and total LLM spend), checked by the UI
  before each run and shown on the Usage page.

RunLimitExceeded derives from BaseException on purpose: retries, model fallbacks and our own
"errors as data" handlers catch Exception, and none of them may swallow a spending stop.
"""

from dataclasses import dataclass, field


class RunLimitExceeded(BaseException):
    """This run hit its token or dollar cap. Caught by the planner (clean stop) or the app."""


@dataclass(frozen=True)
class RunLimits:
    max_tokens: int = 80_000
    max_cost_usd: float = 0.03

    def check(self, tokens: int, cost_usd: float) -> None:
        if tokens > self.max_tokens:
            raise RunLimitExceeded(f"this request used {tokens:,} tokens, over the {self.max_tokens:,}-token limit per run")
        if cost_usd > self.max_cost_usd:
            raise RunLimitExceeded(f"this request cost ${cost_usd:.4f}, over the ${self.max_cost_usd:.2f} limit per run")


DEFAULT_RUN_LIMITS = RunLimits()


@dataclass
class SessionBudget:
    max_planner_runs: int = 8
    max_spend_usd: float = 0.15
    planner_runs: int = 0
    spend_usd: float = 0.0
    history: list = field(default_factory=list)

    def blocked_reason(self) -> str | None:
        """Why the next run is not allowed, or None if it is."""
        if self.planner_runs >= self.max_planner_runs:
            return (f"This session has used its {self.max_planner_runs} trip plans. Start a new browser session "
                    "to plan more (a limit that keeps a public demo affordable).")
        if self.spend_usd >= self.max_spend_usd:
            return f"This session has used its ${self.max_spend_usd:.2f} LLM budget."
        return None

    def add(self, cost_usd: float, planner_ran: bool) -> None:
        self.spend_usd += cost_usd
        self.planner_runs += int(planner_ran)
        self.history.append({"cost_usd": round(cost_usd, 5), "planner": planner_ran})
