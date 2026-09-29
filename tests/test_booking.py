"""Phase 7 tests: idempotent ledger, reviewer decisions, and the approval middleware (scripted LLM)."""

import json
from datetime import date, timedelta

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fakes import ScriptedLLM
from travel_planner import bookings
from travel_planner.agents.booking_agent import build_booking_agent
from travel_planner.bookings import create_booking, list_bookings, parse_decision
from travel_planner.tools.booking import book_flight, book_hotel

CHECK_IN = date.today() + timedelta(days=70)
CHECK_OUT = CHECK_IN + timedelta(days=6)
FLIGHT_ARGS = {"offer_id": "off_1", "airline": "Duffel Airways", "price_total": 1217.46, "passengers": 2}
HOTEL_ARGS = {"hotel_id": "h_1", "hotel_name": "Hamacho Hotel", "check_in": str(CHECK_IN),
              "check_out": str(CHECK_OUT), "guests": 2, "total_price": 762.0}


# ---------- ledger ----------

def test_booking_the_same_item_twice_returns_the_first_booking():
    first = create_booking("flight", "flight:off_1", "x", 100.0, {})
    second = create_booking("flight", "flight:off_1", "x", 100.0, {})
    assert second == {"confirmation": first["confirmation"], "already_booked": True}
    assert len(list_bookings()) == 1


def test_booking_tools_write_to_the_ledger():
    flight = book_flight.invoke(FLIGHT_ARGS)
    hotel = book_hotel.invoke(HOTEL_ARGS)
    assert flight["status"] == hotel["status"] == "ok"
    assert flight["confirmation"].startswith("FL-") and hotel["confirmation"].startswith("HO-")
    assert {b["kind"] for b in list_bookings()} == {"flight", "hotel"}


def test_booking_tool_validates_before_writing():
    result = book_hotel.invoke(HOTEL_ARGS | {"check_out": str(CHECK_IN)})
    assert result["status"] == "error"
    assert list_bookings() == []


# ---------- reviewer decisions ----------

ACTION = {"name": "book_hotel", "args": HOTEL_ARGS}


def test_parse_approve_and_edit():
    assert parse_decision("approve", ACTION) == {"type": "approve"}
    edit = parse_decision('edit: {"guests": 3}', ACTION)
    assert edit["type"] == "edit"
    assert edit["edited_action"]["args"] == HOTEL_ARGS | {"guests": 3}


def test_reject_requires_a_reason_and_logs_it_as_feedback():
    with pytest.raises(ValueError, match="reason"):
        parse_decision("reject", ACTION)
    assert parse_decision("reject: too far from the center", ACTION) == {"type": "reject", "message": "too far from the center"}
    logged = json.loads(bookings.FEEDBACK_LOG.read_text().splitlines()[0])
    assert logged["action"] == "book_hotel" and logged["reason"] == "too far from the center"


@pytest.mark.parametrize("answer, message", [
    ('edit: {"stars": 5}', "Unknown field"),
    ("edit: guests=3", "must be JSON"),
    ("maybe", "Reply 'approve'"),
])
def test_unusable_replies_are_explained(answer, message):
    with pytest.raises(ValueError, match=message):
        parse_decision(answer, ACTION)


# ---------- the middleware ----------

BOTH_BOOKINGS = AIMessage(content="", tool_calls=[
    {"name": "book_flight", "args": FLIGHT_ARGS, "id": "c1"},
    {"name": "book_hotel", "args": HOTEL_ARGS, "id": "c2"},
])


def start_booking(thread: str):
    agent = build_booking_agent(ScriptedLLM(messages=iter([BOTH_BOOKINGS, AIMessage(content="Summary.")])),
                                checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    return agent, config, agent.invoke({"messages": [HumanMessage("Book it")]}, config)


def test_writes_pause_for_review_before_anything_is_booked():
    agent, config, result = start_booking("gate")
    requests = result["__interrupt__"][0].value["action_requests"]
    assert [r["name"] for r in requests] == ["book_flight", "book_hotel"]
    assert "Hamacho Hotel" in requests[1]["description"] and "$762.00" in requests[1]["description"]
    assert list_bookings() == []  # paused BEFORE the tools ran


def test_approve_both():
    agent, config, _ = start_booking("approve")
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config)
    assert {b["kind"] for b in list_bookings()} == {"flight", "hotel"}


def test_approve_flight_reject_hotel():
    agent, config, _ = start_booking("reject")
    final = agent.invoke(Command(resume={"decisions": [
        {"type": "approve"}, {"type": "reject", "message": "too far from the center"}]}), config)
    assert [b["kind"] for b in list_bookings()] == ["flight"]
    rejection = next(m for m in final["messages"] if isinstance(m, ToolMessage) and m.name == "book_hotel")
    assert "too far from the center" in rejection.content  # the agent is told why


def test_edit_runs_the_tool_with_the_reviewers_args():
    agent, config, _ = start_booking("edit")
    edited = {"type": "edit", "edited_action": {"name": "book_hotel", "args": HOTEL_ARGS | {"guests": 3}}}
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}, edited]}), config)
    hotel = next(b for b in list_bookings() if b["kind"] == "hotel")
    assert json.loads(hotel["details"])["guests"] == 3


def test_rebooking_an_approved_plan_does_not_double_book():
    for thread in ("first", "second"):
        agent, config, _ = start_booking(thread)
        agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config)
    assert len(list_bookings()) == 2  # one flight + one hotel, not four
