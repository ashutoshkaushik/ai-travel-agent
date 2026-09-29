"""Phase 9 tests: the whole assistant (router + every sub-agent) with scripted fakes, across a restart."""

from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from fakes import ScriptedLLM, tool_call
from test_booking import BOTH_BOOKINGS
from test_guide import StubIndex
from test_trip_planner import ScriptedResearch, backed, plan
from test_trip_request import draft
from travel_planner.agents.assistant import IntentClassification, build_assistant
from travel_planner.agents.booking_agent import build_booking_agent
from travel_planner.agents.guide_agent import build_guide_agent
from travel_planner.agents.trip_intake import build_trip_intake
from travel_planner.agents.trip_planner import build_trip_planner
from travel_planner.bookings import list_bookings
from travel_planner.persistence import open_checkpointer


def intent(name: str, city: str | None = None) -> IntentClassification:
    return IntentClassification(intent=name, city=city, reasoning="test")


class Recorder:
    """A scripted runnable that also records what it was given."""

    def __init__(self, *outputs):
        self.outputs, self.inputs = list(outputs), []

    def __call__(self, messages):
        self.inputs.append(messages)
        return self.outputs.pop(0)


def make_assistant(checkpoints, store, intents, extractor=None, research=None, booking_llm=None, guide_llm=None):
    return build_assistant(
        open_checkpointer(checkpoints), store,
        classifier=RunnableLambda(Recorder(*intents)),
        intake=build_trip_intake(RunnableLambda(extractor or Recorder(draft()))),
        planner=build_trip_planner(research or ScriptedResearch(backed(plan(100)))),
        booking=build_booking_agent(booking_llm or ScriptedLLM(messages=iter([]))),
        guide=build_guide_agent(StubIndex(0.6), guide_llm or ScriptedLLM(messages=iter([]))),
    )


def say(assistant, text, thread="trip", user="ash"):
    return assistant.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread, "user_id": user}})


def resume(assistant, value, thread="trip", user="ash"):
    return assistant.invoke(Command(resume=value), {"configurable": {"thread_id": thread, "user_id": user}})


def test_off_topic_goes_to_fallback(tmp_path):
    assistant = make_assistant(tmp_path / "c.db", InMemoryStore(), [intent("other")])
    assert "I can plan trips to" in say(assistant, "what's 2+2?")["messages"][-1].content


def test_booking_without_an_approved_plan_is_refused_in_code(tmp_path):
    assistant = make_assistant(tmp_path / "c.db", InMemoryStore(), [intent("book")])
    assert "no approved plan" in say(assistant, "book it")["messages"][-1].content
    assert list_bookings() == []


def test_plan_approve_restart_book_then_ask(tmp_path):
    checkpoints, store = tmp_path / "checkpoints.db", InMemoryStore()

    # 1) plan a trip: intake -> planner -> review pause (inside a sub-agent) -> approve
    assistant = make_assistant(checkpoints, store, [intent("plan_trip", "Tokyo")])
    paused = say(assistant, "Plan Tokyo from SFO, Dec 6-12, 2 people, $4k")
    assert paused["__interrupt__"][0].value["kind"] == "review"
    done = resume(assistant, {"answer": "approve"})
    assert done["trip_status"] == "approved"
    assert "Say 'book it'" in done["messages"][-1].content

    # long-term memory written back after approval
    profile = store.get(("profiles", "ash"), "travel").value
    assert profile["home_airport"] == "SFO" and profile["trips_planned"] == 1

    # 2) "restart": a brand-new assistant object on the same checkpoint file
    assistant = make_assistant(checkpoints, store, [intent("book"), intent("travel_question")],
                               booking_llm=ScriptedLLM(messages=iter([BOTH_BOOKINGS, AIMessage("Booked both.")])),
                               guide_llm=ScriptedLLM(messages=iter([
                                   tool_call("search_travel_guide", {"query": "visa", "city": "Tokyo"}),
                                   tool_call("GuideAnswer", {"answer": "Visa-free for 90 days.",
                                                             "citations": ["tokyo/visa-and-entry"],
                                                             "answered_from_guides": True}, "c2")])))
    paused = say(assistant, "book it")
    assert [a["name"] for a in paused["__interrupt__"][0].value["action_requests"]] == ["book_flight", "book_hotel"]
    booked = resume(assistant, {"decisions": [{"type": "approve"}, {"type": "approve"}]})
    assert booked["trip_status"] == "booked"
    assert len(list_bookings()) == 2

    # 3) a travel question with no city in it uses the saved trip's city
    answer = say(assistant, "do I need a visa?")["messages"][-1].content
    assert "Visa-free" in answer and "tokyo/visa-and-entry" in answer
    assert "official government sources" in answer  # added by code, not by the model


def test_returning_traveler_profile_reaches_intake(tmp_path):
    store = InMemoryStore()
    store.put(("profiles", "ash"), "travel", {"home_airport": "SFO", "usual_travelers": 2})
    extractor = Recorder(draft())
    assistant = make_assistant(tmp_path / "c.db", store, [intent("plan_trip", "Rome")], extractor=extractor)
    say(assistant, "Rome in November for a week", thread="new-trip")
    sent = extractor.inputs[0][-1].content
    assert "home airport SFO" in sent and "Rome in November" in sent


def test_travel_question_without_any_city_asks_which(tmp_path):
    assistant = make_assistant(tmp_path / "c.db", InMemoryStore(), [intent("travel_question")])
    assert "Which city" in say(assistant, "do I need a visa?")["messages"][-1].content


def test_human_decisions_appear_in_the_main_chat(tmp_path):
    """Regression: an escalation answered with $2,700 and the review approval never reached the chat,
    so the chat said $1,500 while the trip card said $2,700."""
    assistant = make_assistant(tmp_path / "c.db", InMemoryStore(), [intent("plan_trip", "Tokyo"), intent("book")],
                               extractor=Recorder(draft(budget_usd=1500)), research=ScriptedResearch(backed(plan(150))),
                               booking_llm=ScriptedLLM(messages=iter([BOTH_BOOKINGS, AIMessage("Done.")])))
    say(assistant, "Plan Tokyo for $1,500", thread="decisions")
    resume(assistant, {"answer": "$2,700"}, thread="decisions")
    done = resume(assistant, {"answer": "approve"}, thread="decisions")
    decisions = [m.content for m in done["messages"] if m.additional_kwargs.get("decision")]
    assert decisions[0].startswith("Over budget by $") and decisions[1] == "Raised budget to $2,700"
    assert decisions[2].startswith("Review the plan") and decisions[3] == "Approved the plan"

    say(assistant, "book it", thread="decisions")
    booked = resume(assistant, {"decisions": [{"type": "approve"}, {"type": "reject", "message": "too far out"}]},
                    thread="decisions")
    answer = [m.content for m in booked["messages"] if m.additional_kwargs.get("decision") == "answer"][-1]
    assert "Approved the flight" in answer and "Rejected the hotel: too far out" in answer
