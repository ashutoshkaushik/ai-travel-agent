"""Phase 5: the research agent, rebuilt with create_agent.

create_agent is a pre-assembled LangGraph: the same model -> tools -> model loop we wired by
hand in Phase 2, plus a structured-output step. Compare this file with react_agent.py.
"""

import json

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from travel_planner.cities import SUPPORTED_CITIES
from travel_planner.guardrails import ToolOutputGuard, unfence
from travel_planner.llm import agent_middleware, primary_model
from travel_planner.models import TripPlan, TripRequest
from travel_planner.prompts import RESEARCH_PROMPT
from travel_planner.tools import get_weather, search_flights, search_hotels, search_top_sights

RESEARCH_TOOLS = [search_flights, search_hotels, search_top_sights, get_weather]  # only what this job needs


def build_research_agent(llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None):
    return create_agent(
        llm or primary_model(),
        tools=tools or RESEARCH_TOOLS,
        system_prompt=RESEARCH_PROMPT,
        # ToolStrategy: the final answer is a call to a "TripPlan" tool, validated by Pydantic.
        # handle_errors=True: if validation fails, the error goes back to the LLM to fix and retry.
        # Works with any tool-calling model (e.g. Nebius later), not only OpenAI's native JSON mode.
        response_format=ToolStrategy(TripPlan, handle_errors=True),
        # Tool output is untrusted: sanitize + fence it (guardrails.py). Real runs also get retry + fallback.
        middleware=[ToolOutputGuard()] + ([] if llm else agent_middleware()),
    )


def research_request_message(request: TripRequest, constraints: list[str] | None = None) -> HumanMessage:
    """Turn a validated TripRequest into the agent's task. Everything here is already validated,
    so the agent never has to interpret free text or guess airport codes."""
    city = SUPPORTED_CITIES[request.city]
    lines = [
        f"Plan this trip: {request.summary()}.",
        f"Search flights {request.origin} -> {request.destination_airport}, departing {request.depart_date}, "
        f"returning {request.return_date}, {request.adults} adults.",
        f"Search hotels in {request.city} from {request.depart_date} to {request.return_date} for {request.adults} adults.",
        f"Food is estimated separately at ${city.food_usd_per_person_per_day}/person/day; leave room for it in the budget.",
    ]
    if constraints:
        lines.append("Hard constraints from the previous attempt:")
        lines += [f"- {c}" for c in constraints]
    return HumanMessage("\n".join(lines))


def tool_results_from(messages: list) -> list[dict]:
    """Collect the parsed JSON of every successful tool result: the evidence verify_plan() checks against."""
    results = []
    for m in messages:
        content = unfence(m.content) if isinstance(m, ToolMessage) else ""
        if content.startswith("{"):
            data = json.loads(content)
            if data.get("status") == "ok":
                results.append(data)
    return results
