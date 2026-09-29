"""Phase 4: turn a free-text request into a validated TripRequest.

    START -> extract -> validate --(valid)--------------------> confirm -> END
                ^          |
                |          +--(problems, rounds left)--> ask ---+
                |          +--(problems, no rounds left)--> give_up -> END
                +----------------------------------------------+

Contrast with Phase 3: there the LLM decided when to ask (interrupt inside a tool).
Here CODE decides when to ask (interrupt at a fixed node), because "is the request complete
and within our assumptions?" is a rule, not a judgment call.
"""

from typing import Annotated, Literal, TypedDict

from langchain.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Checkpointer, interrupt

from travel_planner.display import decision_messages
from travel_planner.llm import structured_model
from travel_planner.models import TripRequest, TripRequestDraft, validate_draft
from travel_planner.prompts import EXTRACT_PROMPT_TEMPLATE
from travel_planner.config import today

MAX_ASK_ROUNDS = 3


class IntakeState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: dict | None  # latest TripRequestDraft, as a dict (plain JSON checkpoints cleanly)
    problems: list[str]  # overwritten each validation, no reducer
    request: dict | None  # the validated TripRequest, as a dict
    ask_rounds: int


def build_question(problems: list[str]) -> str:
    """Deterministic, no LLM: the traveler sees exactly what the validator found."""
    if len(problems) == 1:
        return f"To plan your trip: {problems[0]}"
    return "To plan your trip:\n" + "\n".join(f"- {p}" for p in problems)


def build_trip_intake(extractor: Runnable | None = None, checkpointer: Checkpointer = None):
    """Args:
        extractor: messages -> TripRequestDraft. Defaults to gpt-4o-mini with structured output;
            tests pass a scripted fake.
        checkpointer: required for the ask node's interrupt to pause and resume.
    """
    if extractor is None:
        # function_calling handles Optional fields without OpenAI strict-schema requirements
        extractor = structured_model(TripRequestDraft)  # retry + fallback model (llm.py)

    def extract(state: IntakeState) -> dict:
        """LLM reads the WHOLE conversation, so answers from earlier rounds accumulate."""
        system = SystemMessage(EXTRACT_PROMPT_TEMPLATE.format(todays_date=today().isoformat()))
        draft = extractor.invoke([system] + state["messages"])
        return {"draft": draft.model_dump()}

    def validate(state: IntakeState) -> dict:
        """Pure Python. Every assumption is enforced here, never in a prompt."""
        request, problems = validate_draft(TripRequestDraft(**state["draft"]))
        return {"request": request.model_dump(mode="json") if request else None, "problems": problems}

    def route_after_validate(state: IntakeState) -> Literal["confirm", "ask", "give_up"]:
        if state["request"]:
            return "confirm"
        return "ask" if state.get("ask_rounds", 0) < MAX_ASK_ROUNDS else "give_up"

    def ask(state: IntakeState) -> dict:
        # Everything above interrupt() re-runs on resume, so keep it side-effect free.
        question = build_question(state["problems"])
        reply = interrupt({"question": question, "problems": state["problems"]})
        return {
            "messages": decision_messages(question, reply["answer"]),
            "ask_rounds": state.get("ask_rounds", 0) + 1,
        }

    def confirm(state: IntakeState) -> dict:
        return {"messages": [AIMessage(f"Got it: {TripRequest(**state['request']).summary()}.")]}

    def give_up(state: IntakeState) -> dict:
        issues = " ".join(state["problems"])
        return {"messages": [AIMessage(f"I couldn't pin down the trip details after {MAX_ASK_ROUNDS} tries. {issues} Please start over with those details.")]}

    graph = StateGraph(IntakeState)
    graph.add_node("extract", extract)
    graph.add_node("validate", validate)
    graph.add_node("ask", ask)
    graph.add_node("confirm", confirm)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges("validate", route_after_validate, ["confirm", "ask", "give_up"])
    graph.add_edge("ask", "extract")  # the loop: re-extract with the new answer in the conversation
    graph.add_edge("confirm", END)
    graph.add_edge("give_up", END)
    return graph.compile(checkpointer=checkpointer)
