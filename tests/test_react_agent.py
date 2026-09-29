"""Phase 2 tests for the loop mechanics, driven by a scripted fake LLM (no API cost)."""

import json

from fakes import ScriptedLLM, tool_call
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from travel_planner.agents.react_agent import build_react_agent, route_after_reason
from travel_planner.guardrails import unfence
from travel_planner.tools import search_places_by_interest


def run(*script: AIMessage) -> list:
    agent = build_react_agent([search_places_by_interest], llm=ScriptedLLM(messages=iter(script)))
    return agent.invoke({"messages": [HumanMessage("go")]})["messages"]


def tool_results(messages: list) -> list[dict]:
    return [json.loads(unfence(m.content)) for m in messages if isinstance(m, ToolMessage)]  # tool data is fenced


def test_invalid_tool_args_become_error_observation():
    messages = run(
        tool_call("search_places_by_interest", {"city": "Tokyo", "interests": ["shopping"]}),
        AIMessage(content="Sorry, shopping isn't a supported interest."),
    )
    [result] = tool_results(messages)
    assert result["status"] == "error"
    assert "shopping" in result["reason"]
    assert messages[-1].content.startswith("Sorry")  # the loop survived and the LLM got a turn


def test_unknown_tool_becomes_error_observation():
    [result] = tool_results(run(tool_call("book_spaceship", {}), AIMessage(content="ok")))
    assert result["status"] == "error"


def test_router_ends_on_plain_answer_and_acts_on_tool_calls():
    assert route_after_reason({"messages": [AIMessage(content="done")]}) == "__end__"
    assert route_after_reason({"messages": [tool_call("geocode", {"place": "Tokyo"})]}) == "action_node"
