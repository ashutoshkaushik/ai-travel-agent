"""Phase 3 tests: pause / resume with a scripted fake LLM (no API cost, fully deterministic)."""

from fakes import ScriptedLLM, tool_call
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_planner.agents.react_agent import build_react_agent
from travel_planner.tools.human import ask_traveler


def ask(question: str) -> AIMessage:
    return tool_call("ask_traveler", {"question": question})


def make_agent(*script: AIMessage):
    return build_react_agent([ask_traveler], checkpointer=InMemorySaver(), llm=ScriptedLLM(messages=iter(script)))


def test_pauses_and_resumes_with_answer():
    agent = make_agent(ask("Where are you flying from?"), AIMessage(content="Great, searching from SFO."))
    config = {"configurable": {"thread_id": "t1"}}

    first = agent.invoke({"messages": [HumanMessage("Plan me a trip to Japan")]}, config)
    assert first["__interrupt__"][0].value == {"question": "Where are you flying from?"}
    assert agent.get_state(config).next == ("action_node",)  # paused inside action_node

    final = agent.invoke(Command(resume={"answer": "SFO"}), config)
    tool_msg = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "SFO"  # the human's answer arrives as an ordinary tool result
    assert final["messages"][-1].content == "Great, searching from SFO."


def test_interrupt_is_not_swallowed_by_error_handling():
    """Regression: action_node's `except Exception` must re-raise GraphInterrupt."""
    agent = make_agent(ask("Dates?"), AIMessage(content="done"))
    result = agent.invoke({"messages": [HumanMessage("trip")]}, {"configurable": {"thread_id": "t2"}})
    assert "__interrupt__" in result  # if swallowed, we'd get an error ToolMessage and no pause


def test_tools_before_interrupt_rerun_on_resume():
    """Proven: resuming restarts the interrupted NODE from the top.

    The LLM requests two tools in one turn: a lookup, then ask_traveler. The lookup runs, the
    interrupt pauses the node, and on resume action_node starts over, so the lookup runs AGAIN.
    Harmless for a cached search; a double charge if it were a booking. (Phase 9 handles this.)
    """
    from langchain.tools import tool

    calls = []

    @tool
    def lookup(city: str) -> str:
        """Look something up."""
        calls.append(city)
        return "found"

    parallel_turn = AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {"city": "Tokyo"}, "id": "c1"},
        {"name": "ask_traveler", "args": {"question": "Dates?"}, "id": "c2"},
    ])
    agent = build_react_agent([lookup, ask_traveler], checkpointer=InMemorySaver(),
                              llm=ScriptedLLM(messages=iter([parallel_turn, AIMessage(content="done")])))
    config = {"configurable": {"thread_id": "rerun"}}

    agent.invoke({"messages": [HumanMessage("trip")]}, config)
    assert calls == ["Tokyo"]
    agent.invoke(Command(resume={"answer": "Dec 6-12"}), config)
    assert calls == ["Tokyo", "Tokyo"]  # ran twice


def test_threads_are_isolated():
    agent = make_agent(ask("Dates?"))
    agent.invoke({"messages": [HumanMessage("trip")]}, {"configurable": {"thread_id": "a"}})
    assert agent.get_state({"configurable": {"thread_id": "a"}}).next == ("action_node",)
    assert agent.get_state({"configurable": {"thread_id": "b"}}).values == {}  # nothing to resume


def test_follow_up_on_same_thread_keeps_history():
    agent = make_agent(AIMessage(content="first answer"), AIMessage(content="second answer"))
    config = {"configurable": {"thread_id": "t3"}}
    agent.invoke({"messages": [HumanMessage("hello")]}, config)
    final = agent.invoke({"messages": [HumanMessage("and again")]}, config)
    assert [m.content for m in final["messages"]] == ["hello", "first answer", "and again", "second answer"]
