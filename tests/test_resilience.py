"""Resilience tests with mocked failures: HTTP retries/backoff/breaker, model fallback, run caps."""

import httpx
import pytest
from fakes import ScriptedLLM
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from travel_planner import http_cache
from travel_planner.limits import RunLimitExceeded, RunLimits, SessionBudget
from travel_planner.llm import resilient
from travel_planner.trace import RunStats

URL = "https://api.duffel.com/air/offer_requests"


@pytest.fixture(autouse=True)
def isolated_http(tmp_path, monkeypatch):
    monkeypatch.setattr(http_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(http_cache, "_sleep", lambda s: None)  # no real waiting in tests
    http_cache.clear_faults()
    http_cache.reset_breakers()
    http_cache.RETRY_EVENTS.clear()
    yield
    http_cache.clear_faults()
    http_cache.reset_breakers()


def scripted_http(monkeypatch, *outcomes):
    """Each call returns the next outcome: an int status (JSON body) or an exception to raise."""
    calls = []

    def fake(method, url, **kw):
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, json={"data": "ok"} if outcome < 400 else {"errors": [{"message": "no"}]})

    monkeypatch.setattr(http_cache.httpx, "request", fake)
    return calls


# ---------- HTTP ----------

def test_503_twice_then_success_is_retried_with_backoff(monkeypatch):
    calls = scripted_http(monkeypatch, 503, 503, 200)
    resp = http_cache.request("POST", URL, json_body={"q": 1})
    assert resp.status_code == 200 and len(calls) == 3
    assert [e[2] for e in http_cache.RETRY_EVENTS] == [503, 503]


def test_429_and_timeouts_are_retried(monkeypatch):
    scripted_http(monkeypatch, 429, httpx.ReadTimeout("slow"), 200)
    assert http_cache.request("GET", URL).status_code == 200


def test_4xx_is_not_retried(monkeypatch):
    calls = scripted_http(monkeypatch, 400)
    assert http_cache.request("GET", URL).status_code == 400 and len(calls) == 1


def test_final_failure_returns_error_data_and_never_raises(monkeypatch):
    scripted_http(monkeypatch, 503, 503, 503)
    resp = http_cache.request("GET", URL)
    assert resp.status_code == 503 and "failed after 3 attempts" in resp.data["error"]


def test_circuit_breaker_opens_then_fails_fast(monkeypatch):
    calls = scripted_http(monkeypatch, *([503] * 9))
    for _ in range(http_cache.BREAKER_THRESHOLD):
        http_cache.request("GET", URL, params={"n": _})
    before = len(calls)
    resp = http_cache.request("GET", URL, params={"n": "new"})
    assert len(calls) == before  # no network call while open
    assert "circuit breaker open" in resp.data["error"]


def test_open_breaker_serves_a_stale_cached_copy(monkeypatch):
    scripted_http(monkeypatch, 200, *([503] * 9))
    http_cache.request("GET", URL, params={"q": "cached"}, ttl_seconds=0)  # cached, instantly stale
    for i in range(http_cache.BREAKER_THRESHOLD):
        http_cache.request("GET", URL, params={"q": i})
    resp = http_cache.request("GET", URL, params={"q": "cached"}, ttl_seconds=0)
    assert resp.status_code == 200 and resp.from_cache


def test_injected_faults_are_retried_like_real_ones(monkeypatch):
    calls = scripted_http(monkeypatch, 200)
    http_cache.inject_faults("duffel", 2, 503)
    resp = http_cache.request("GET", URL)
    assert resp.status_code == 200 and len(calls) == 1  # two simulated failures, then the real call
    assert all(e[4] for e in http_cache.RETRY_EVENTS)  # flagged as simulated in the trace


def test_frozen_cache_ignores_age(monkeypatch):
    scripted_http(monkeypatch, 200)
    http_cache.request("GET", URL, ttl_seconds=0)
    monkeypatch.setenv("TRAVEL_FROZEN_CACHE", "1")
    assert http_cache.request("GET", URL, ttl_seconds=0).from_cache


def test_tools_turn_a_dead_provider_into_error_data(monkeypatch):
    from travel_planner.tools import search_top_sights
    monkeypatch.setenv("SERPAPI_API_KEY", "test")
    scripted_http(monkeypatch, 503, 503, 503)
    result = search_top_sights.invoke({"city": "Tokyo"})
    assert result["status"] == "error"


# ---------- model fallback ----------

def test_primary_fails_twice_then_fallback_answers():
    attempts = []

    def flaky(_):
        attempts.append(1)
        raise RuntimeError("503")

    chain = resilient(RunnableLambda(flaky), RunnableLambda(lambda _: "fallback answer"))
    assert chain.invoke("hi") == "fallback answer"
    assert len(attempts) == 2


class FailingModel(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("simulated 503")


def test_create_agent_middleware_retries_then_falls_back():
    fallback = ScriptedLLM(messages=iter([AIMessage("answered by the fallback model",
                                                   response_metadata={"model_name": "gpt-4.1-mini"})]))
    agent = create_agent(FailingModel(messages=iter([])), tools=[], middleware=[
        ModelFallbackMiddleware(fallback),
        ModelRetryMiddleware(max_retries=1, on_failure="error", initial_delay=0, jitter=False)])
    result = agent.invoke({"messages": [HumanMessage("hi")]})
    last = result["messages"][-1]
    assert last.content == "answered by the fallback model"
    assert last.response_metadata["model_name"] == "gpt-4.1-mini"  # the trace can show who answered


# ---------- spending caps ----------

def llm_result(tokens_in: int, tokens_out: int) -> LLMResult:
    msg = AIMessage("x", usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                         "total_tokens": tokens_in + tokens_out},
                    response_metadata={"model_name": "gpt-4o-mini"})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_run_stats_stop_the_run_at_the_token_cap():
    stats = RunStats(limits=RunLimits(max_tokens=1_000, max_cost_usd=1.0))
    stats.on_llm_end(llm_result(400, 100))
    with pytest.raises(RunLimitExceeded, match="1,000-token limit"):
        stats.on_llm_end(llm_result(400, 200))


def test_run_limit_is_not_swallowed_by_retries_or_fallbacks():
    def over_budget(_):
        raise RunLimitExceeded("cap")

    with pytest.raises(RunLimitExceeded):
        resilient(RunnableLambda(over_budget), RunnableLambda(lambda _: "should not run")).invoke("x")


def test_planner_stops_cleanly_at_the_cap():
    from datetime import date, timedelta

    from travel_planner.agents.trip_planner import build_trip_planner, initial_state
    from travel_planner.models import TripRequest

    def research(req, constraints, config):
        raise RunLimitExceeded("this request used 90,000 tokens, over the 80,000-token limit per run")

    d = date.today() + timedelta(days=70)
    req = TripRequest(origin="SFO", city="Tokyo", depart_date=d, return_date=d + timedelta(days=6), adults=2, budget_usd=4000)
    result = build_trip_planner(research, InMemorySaver()).invoke(initial_state(req), {"configurable": {"thread_id": "cap"}})
    assert result["status"] == "stopped"
    assert "stopped planning because" in result["messages"][-1].content


def test_session_budget_blocks_after_limits():
    budget = SessionBudget(max_planner_runs=2, max_spend_usd=1.0)
    budget.add(0.01, planner_ran=True)
    assert budget.blocked_reason() is None
    budget.add(0.01, planner_ran=True)
    assert "2 trip plans" in budget.blocked_reason()
