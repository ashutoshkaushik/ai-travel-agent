"""The one place models are created: timeouts, retries, and a fallback model.

- Every model call has a timeout, and the OpenAI client itself retries 429/5xx with backoff.
- If the primary model still fails twice, the same step is retried on a fallback model.
  Which model answered is on every AIMessage (response_metadata["model_name"]), so traces show it.
- For create_agent agents this is middleware (ModelFallbackMiddleware around ModelRetryMiddleware);
  for hand-built nodes it's LangChain's .with_retry().with_fallbacks().
- simulate_outage(n) makes the next n primary-model calls fail ("Break it" mode and tests).
"""

import threading

from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_openai import ChatOpenAI

from travel_planner import config  # noqa: F401  (loads .env)

PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4.1-mini"
TIMEOUT_S = 30
PRIMARY_ATTEMPTS = 2

_lock = threading.Lock()
_outages = {"count": 0}


class SimulatedOutage(RuntimeError):
    """Raised by the primary model while a simulated outage is active."""


def simulate_outage(calls: int) -> None:
    with _lock:
        _outages["count"] = calls


def _take_outage() -> bool:
    with _lock:
        if _outages["count"] > 0:
            _outages["count"] -= 1
            return True
        return False


class PrimaryChatModel(ChatOpenAI):
    """The primary model, with a hook to simulate an outage."""

    def _generate(self, *args, **kwargs):
        if _take_outage():
            raise SimulatedOutage(f"Simulated outage: {self.model_name} returned 503")
        return super()._generate(*args, **kwargs)


def primary_model() -> ChatOpenAI:
    return PrimaryChatModel(model=PRIMARY_MODEL, temperature=0, timeout=TIMEOUT_S, max_retries=2)


def fallback_model() -> ChatOpenAI:
    return ChatOpenAI(model=FALLBACK_MODEL, temperature=0, timeout=TIMEOUT_S, max_retries=2)


def resilient(primary_runnable, fallback_runnable):
    """Two attempts on the primary (exponential backoff + jitter), then the fallback."""
    return primary_runnable.with_retry(stop_after_attempt=PRIMARY_ATTEMPTS, wait_exponential_jitter=True,
                                       exponential_jitter_params={"initial": 0.5, "max": 4}).with_fallbacks(
        [fallback_runnable])


def chat_model():
    """A plain chat model (no tools) with retry + fallback."""
    return resilient(primary_model(), fallback_model())


def structured_model(schema):
    """Structured output (function calling) with retry + fallback."""
    return resilient(primary_model().with_structured_output(schema, method="function_calling"),
                     fallback_model().with_structured_output(schema, method="function_calling"))


def tools_model(tools, **bind_kwargs):
    """A tool-calling model with retry + fallback (for hand-wired agents like react_agent)."""
    return resilient(primary_model().bind_tools(tools, **bind_kwargs), fallback_model().bind_tools(tools, **bind_kwargs))


def agent_middleware() -> list:
    """For create_agent: fallback is the OUTER middleware, so it runs after the retry gives up."""
    return [ModelFallbackMiddleware(fallback_model()),
            ModelRetryMiddleware(max_retries=PRIMARY_ATTEMPTS - 1, on_failure="error", initial_delay=0.5, max_delay=4.0)]
