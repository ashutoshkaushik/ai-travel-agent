"""Phases 2-3: a ReAct agent wired by hand.

    START -> reason_node --(tool_calls?)--> action_node --> back to reason_node
                        +--(no tool_calls)--> END

The LLM never executes anything. It only emits tool-call requests; action_node runs them.

Phase 2: build_react_agent(ALL_TOOLS)
Phase 3: build_react_agent(ALL_TOOLS + [ask_traveler], checkpointer=InMemorySaver())
         -> the same loop, but it can now pause mid-run and resume with the traveler's answer.
"""

import json
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Checkpointer

from travel_planner.guardrails import guard_tool_output
from travel_planner.llm import tools_model
from travel_planner.prompts import REACT_PROMPT_TEMPLATE
from travel_planner.config import today



# ---------- State ----------
# `messages` uses the add_messages reducer: a node returning [new_msg] APPENDS it, not replaces.
# We call it `messages` because create_agent uses `messages`,
# which will let us compose this with prebuilt agents in later phases without glue code.
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def route_after_reason(state: GraphState) -> str:
    """Tool calls requested -> go act. Plain text -> the model is done."""
    return "action_node" if state["messages"][-1].tool_calls else END


def build_react_agent(tools: list[BaseTool], checkpointer: Checkpointer = None, llm: BaseChatModel | None = None):
    """Wire the reason/act loop around any set of tools.

    Args:
        tools: The tools the LLM may request.
        checkpointer: Saves state after every step. Required for interrupt() to pause and resume.
        llm: Override the model (tests pass a fake one so they cost nothing).
    """
    # sends all tool schemas with every call; real runs get retry + a fallback model (llm.py)
    llm_with_tools = llm.bind_tools(tools) if llm else tools_model(tools)
    tools_by_name = {t.name: t for t in tools}

    def reason_node(state: GraphState) -> dict:
        """The thinking step: the LLM sees the prompt + history and either answers or requests tools."""
        system = SystemMessage(REACT_PROMPT_TEMPLATE.format(todays_date=today().isoformat()))
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    def action_node(state: GraphState, config: RunnableConfig) -> dict:
        """The doing step: plain Python runs every tool the LLM asked for. No LLM involved.

        If a tool calls interrupt(), execution stops here. On resume, this node restarts FROM
        THE TOP, so every tool call before the interrupting one runs again.
        """
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                observation = {"status": "error", "reason": f"Unknown tool {call['name']!r}."}
            else:
                try:
                    observation = tool.invoke(call["args"], config)  # pass config: callbacks + tracing see the call
                except GraphBubbleUp:
                    # interrupt() pauses the graph by raising GraphInterrupt (a GraphBubbleUp).
                    # It is control flow, not an error: it must reach LangGraph untouched.
                    raise
                except Exception as e:
                    # Argument validation (e.g. a value outside a Literal) happens BEFORE the tool's
                    # own error handling runs, so catch it here and let the LLM correct its arguments.
                    observation = {"status": "error", "reason": f"Invalid arguments: {e}", "hint": "Fix the arguments and retry."}
            # External tool data is untrusted: sanitize + fence it. A plain string is the traveler's own
            # answer (ask_traveler), which is the user speaking, so it passes through as-is.
            content = observation if isinstance(observation, str) else guard_tool_output(json.dumps(observation))
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    graph = StateGraph(GraphState)
    graph.add_node("reason_node", reason_node)
    graph.add_node("action_node", action_node)
    graph.add_edge(START, "reason_node")
    graph.add_conditional_edges("reason_node", route_after_reason, ["action_node", END])
    graph.add_edge("action_node", "reason_node")  # the back edge: this is what makes it a loop
    return graph.compile(checkpointer=checkpointer)
